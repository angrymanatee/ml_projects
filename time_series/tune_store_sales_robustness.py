"""Architecture robustness check: is the phase-1 tuning ranking signal or noise?

Phase-1 architecture search (tune_store_sales_encoder_only) trains each candidate for
few epochs, so trial-to-trial ranking can be dominated by training noise rather than
true architecture quality. This script pulls the finished child trials of a phase-1
MLflow parent run, deduplicates by (d_model, num_layers, dim_feedforward), selects the
single best architecture plus N-1 diverse candidates (farthest-point sampling over the
hyperparameter grid), and re-trains each candidate multiple times for many more epochs.
If rankings are stable under this longer/repeated regime, phase-1's conclusion was signal;
if they scramble, phase-1 was mostly noise.

Uses GridSampler so every candidate runs exactly n_repeats times. All features are
enabled (matching what phase-1 tuned against). Each trial is a nested MLflow run.

Run with:
    uv run python -m time_series.tune_store_sales_robustness
    uv run python -m time_series.tune_store_sales_robustness --parent-run-name store_sales_allfeatures_tune \\
        --n-candidates 5 --epochs 150 --n-repeats 3
"""

import argparse
import dataclasses

import optuna
import torch

import mlflow
from common.git import get_branch, get_sha
from common.model_registry import TRACKING_URI
from time_series.main_store_sales_encoder_only import PoolingMode, train_and_eval
from time_series.store_sales import HOLIDAY_FEATURE_COLS, STORE_FEATURE_COLS, StoreData

_TUNE_EXPERIMENT_NAME = "StoreSales_EncoderOnly_Tune"
_ROBUSTNESS_EXPERIMENT_NAME = "StoreSales_ArchRobustness"


@dataclasses.dataclass(frozen=True)
class ArchCandidate:
    """One (d_model, num_layers, dim_feedforward, lr) architecture pulled from phase-1."""

    d_model: int
    num_layers: int
    dim_feedforward: int
    lr: float
    phase1_best_val_loss: float

    @property
    def key(self) -> tuple[int, int, int]:
        """Dedup key — architecture shape, ignoring lr."""
        return (self.d_model, self.num_layers, self.dim_feedforward)

    @property
    def name(self) -> str:
        return f"d{self.d_model}_l{self.num_layers}_ff{self.dim_feedforward}"

    def as_config(self) -> dict:
        return {
            "lr": self.lr,
            "d_model": self.d_model,
            "nhead": 2,
            "num_layers": self.num_layers,
            "batch_size": 64,
            "pooling_mode": PoolingMode.ALL,
            "dim_feedforward": self.dim_feedforward,
        }


def fetch_phase1_candidates(
    parent_run_name: str, experiment_name: str = _TUNE_EXPERIMENT_NAME
) -> list[ArchCandidate]:
    """Pull finished child trials of a phase-1 tuning run, deduped by architecture shape.

    When multiple trials share the same (d_model, num_layers, dim_feedforward), keeps
    the one with the lowest best_val_loss (its lr and reported loss are used).

    Args:
        parent_run_name: MLflow run name of the phase-1 parent run (the study_name
            passed to tune_store_sales_encoder_only.tune).
        experiment_name: MLflow experiment the phase-1 run was logged under.

    Returns:
        List of distinct ArchCandidate, one per unique architecture shape.
    """
    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment {experiment_name!r} not found")

    parent_runs = client.search_runs(
        [experiment.experiment_id],
        filter_string=f"tags.mlflow.runName = '{parent_run_name}'",
        order_by=["start_time DESC"],
    )
    if not parent_runs:
        raise ValueError(f"No parent run named {parent_run_name!r} found")
    parent_run_id = parent_runs[0].info.run_id

    child_runs = client.search_runs(
        [experiment.experiment_id],
        filter_string=f"tags.mlflow.parentRunId = '{parent_run_id}'",
    )

    best_by_key: dict[tuple[int, int, int], ArchCandidate] = {}
    for run in child_runs:
        val_loss = run.data.metrics.get("best_val_loss")
        if val_loss is None:
            continue  # trial still running or failed
        params = run.data.params
        candidate = ArchCandidate(
            d_model=int(params["d_model"]),
            num_layers=int(params["num_layers"]),
            dim_feedforward=int(params["dim_feedforward"]),
            lr=float(params["lr"]),
            phase1_best_val_loss=val_loss,
        )
        existing = best_by_key.get(candidate.key)
        if (
            existing is None
            or candidate.phase1_best_val_loss < existing.phase1_best_val_loss
        ):
            best_by_key[candidate.key] = candidate

    if not best_by_key:
        raise ValueError(
            f"No completed child trials found under parent run {parent_run_name!r}"
        )
    return list(best_by_key.values())


def _normalized_distance(
    a: ArchCandidate, b: ArchCandidate, ranges: dict[str, float]
) -> float:
    """Euclidean distance over (d_model, num_layers, dim_feedforward), each scaled by its range."""
    terms = [
        (a.d_model - b.d_model) / ranges["d_model"] if ranges["d_model"] else 0.0,
        (
            (a.num_layers - b.num_layers) / ranges["num_layers"]
            if ranges["num_layers"]
            else 0.0
        ),
        (
            (a.dim_feedforward - b.dim_feedforward) / ranges["dim_feedforward"]
            if ranges["dim_feedforward"]
            else 0.0
        ),
    ]
    return sum(t**2 for t in terms) ** 0.5


def select_diverse_candidates(
    candidates: list[ArchCandidate], n_candidates: int
) -> list[ArchCandidate]:
    """Select the best architecture plus diverse others via farthest-point sampling.

    Always includes the single lowest-loss candidate first, then greedily adds whichever
    remaining candidate maximizes the minimum normalized distance to the selected set —
    spreading the picks across the explored (d_model, num_layers, dim_feedforward) space
    instead of clustering near the best trial.

    Args:
        candidates: deduplicated architecture candidates (see fetch_phase1_candidates).
        n_candidates: how many to select; returns all of them if fewer are available.

    Returns:
        List of n_candidates ArchCandidate (or len(candidates) if that's fewer).
    """
    if n_candidates >= len(candidates):
        return list(candidates)

    ranges: dict[str, float] = {
        "d_model": float(
            max(c.d_model for c in candidates) - min(c.d_model for c in candidates)
        ),
        "num_layers": float(
            max(c.num_layers for c in candidates)
            - min(c.num_layers for c in candidates)
        ),
        "dim_feedforward": float(
            max(c.dim_feedforward for c in candidates)
            - min(c.dim_feedforward for c in candidates)
        ),
    }

    remaining = list(candidates)
    best = min(remaining, key=lambda c: c.phase1_best_val_loss)
    selected = [best]
    remaining.remove(best)

    while len(selected) < n_candidates and remaining:
        next_candidate = max(
            remaining,
            key=lambda c: min(_normalized_distance(c, s, ranges) for s in selected),
        )
        selected.append(next_candidate)
        remaining.remove(next_candidate)

    return selected


def objective(
    trial: optuna.Trial,
    store_data: StoreData,
    device: torch.device,
    epochs: int,
    split: float,
    candidates: list[ArchCandidate],
    n_repeats: int,
) -> float:
    """Optuna objective: train one candidate architecture, return best val MSLE.

    Must be called inside an active mlflow.start_run() context. Starts a nested child
    run so each trial appears as a collapsible entry under the parent study run.

    n_repeats is used only to size the repeat_idx grid dimension: GridSampler treats
    (candidate_idx, repeat_idx) pairs as distinct grid points, otherwise it stops the
    study after visiting each candidate_idx once and silently ignores any requested repeats.
    """
    candidate_idx: int = trial.suggest_categorical(
        "candidate_idx", list(range(len(candidates)))
    )
    trial.suggest_categorical("repeat_idx", list(range(n_repeats)))
    candidate = candidates[candidate_idx]

    run_name = f"trial_{trial.number}_{candidate.name}"
    with mlflow.start_run(nested=True, run_name=run_name):
        mlflow.log_params(
            {
                "candidate_name": candidate.name,
                "trial_number": trial.number,
                "phase1_best_val_loss": candidate.phase1_best_val_loss,
                "epochs": epochs,
                "split": split,
                **{k: str(v) for k, v in candidate.as_config().items()},
            }
        )
        val_loss, *_ = train_and_eval(
            config=candidate.as_config(),
            store_data=store_data,
            device=device,
            epochs=epochs,
            split=split,
            save_checkpoints=False,
            log_metrics=True,
        )
        mlflow.log_metric("best_val_loss", val_loss)

    return val_loss


def run_study(
    parent_run_name: str,
    n_candidates: int,
    epochs: int,
    n_repeats: int,
    split: float,
    study_name: str,
    candidate_keys: list[tuple[int, int, int]] | None = None,
) -> None:
    """Run the architecture robustness study and log everything to MLflow.

    Pulls candidates from a completed phase-1 tuning run, selects a diverse subset,
    then evaluates each candidate n_repeats times at the given epoch budget using
    GridSampler (every candidate runs exactly n_repeats times). All feature groups
    are enabled, matching the dataset phase-1 was tuned against.

    Args:
        parent_run_name: MLflow run name of the phase-1 parent run to pull candidates from.
        n_candidates: number of diverse architecture candidates to evaluate. Ignored
            when candidate_keys is given.
        epochs: training epochs per trial (should exceed phase-1's epoch budget).
        n_repeats: number of times to re-run each candidate.
        split: train/val split fraction.
        study_name: Optuna study name and MLflow parent run name for this study.
        candidate_keys: explicit (d_model, num_layers, dim_feedforward) tuples to
            evaluate, bypassing diversity-based selection. Useful for confirming a
            small set of promising candidates (e.g. from a prior robustness run) at
            a larger epoch budget. Raises if any key isn't found among phase-1 trials.
    """
    mlflow.set_tracking_uri(TRACKING_URI)

    all_candidates = fetch_phase1_candidates(parent_run_name)
    if candidate_keys is not None:
        candidates_by_key = {c.key: c for c in all_candidates}
        missing = [key for key in candidate_keys if key not in candidates_by_key]
        if missing:
            raise ValueError(f"Candidate key(s) not found in phase-1 trials: {missing}")
        selected = [candidates_by_key[key] for key in candidate_keys]
    else:
        selected = select_diverse_candidates(all_candidates, n_candidates)
    print(
        f"Selected {len(selected)} candidates from {len(all_candidates)} unique architectures:"
    )
    for candidate in selected:
        print(
            f"  {candidate.name}: phase1_best_val_loss={candidate.phase1_best_val_loss:.6f}"
        )

    store_data = StoreData(
        dtype=torch.float32,
        include_oil=True,
        include_onpromotion=True,
        store_feature_cols=list(STORE_FEATURE_COLS),
        holiday_features=list(HOLIDAY_FEATURE_COLS),
    )
    device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )

    n_configs = len(selected)
    # repeat_idx makes (candidate_idx, repeat_idx) pairs distinct grid points; GridSampler
    # otherwise stops the study after visiting each candidate_idx once, silently ignoring
    # any requested repeats.
    grid_search_space = {
        "candidate_idx": list(range(n_configs)),
        "repeat_idx": list(range(n_repeats)),
    }
    sampler = optuna.samplers.GridSampler(grid_search_space, seed=42)
    study = optuna.create_study(
        direction="minimize", sampler=sampler, study_name=study_name
    )

    mlflow.set_experiment(_ROBUSTNESS_EXPERIMENT_NAME)
    mlflow.set_experiment_tags(
        {
            "dataset": "store-sales-kaggle",
            "task": "time-series-forecasting",
            "loss": "RMSLE",
        }
    )

    n_trials = n_configs * n_repeats

    with mlflow.start_run(
        run_name=study_name,
        tags={
            "git_branch": get_branch(),
            "git_sha": get_sha(),
            "n_trials": str(n_trials),
            "epochs_per_trial": str(epochs),
            "n_repeats": str(n_repeats),
            "phase1_parent_run": parent_run_name,
            "candidates": ",".join(c.name for c in selected),
        },
    ):
        study.optimize(
            lambda trial: objective(
                trial, store_data, device, epochs, split, selected, n_repeats
            ),
            n_trials=n_trials,
        )

        # Summarize mean/std/best val loss per candidate.
        losses_by_candidate: dict[str, list[float]] = {c.name: [] for c in selected}
        for completed_trial in study.trials:
            if completed_trial.value is None:
                continue
            idx = completed_trial.params["candidate_idx"]
            losses_by_candidate[selected[idx].name].append(completed_trial.value)

        summary: dict[str, tuple[float, float, float]] = {}
        for name, losses in losses_by_candidate.items():
            if not losses:
                continue
            mean = sum(losses) / len(losses)
            variance = sum((x - mean) ** 2 for x in losses) / len(losses)
            std = variance**0.5
            summary[name] = (mean, std, min(losses))
            mlflow.log_metric(f"mean_val_loss_{name}", mean)
            mlflow.log_metric(f"std_val_loss_{name}", std)
            mlflow.log_metric(f"best_val_loss_{name}", min(losses))

    print("\nArchitecture robustness results:")
    print(
        f"{'Candidate':<20} {'Mean MSLE':>12} {'Std':>10} {'Best':>10} {'Phase1':>10}"
    )
    print("-" * 66)
    candidate_by_name = {c.name: c for c in selected}
    for name, (mean, std, best) in sorted(summary.items(), key=lambda kv: kv[1][0]):
        phase1_loss = candidate_by_name[name].phase1_best_val_loss
        print(
            f"{name:<20} {mean:>12.6f} {std:>10.6f} {best:>10.6f} {phase1_loss:>10.6f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-evaluate diverse phase-1 architecture candidates for longer to check ranking stability"
    )
    parser.add_argument(
        "--parent-run-name",
        type=str,
        default="store_sales_allfeatures_tune",
        help="MLflow run name of the phase-1 parent tuning run to pull candidates from",
    )
    parser.add_argument(
        "--n-candidates",
        type=int,
        default=5,
        help="number of diverse architecture candidates to evaluate (default: 5)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=150,
        help="training epochs per trial (default: 150)",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=3,
        help="number of times to re-run each candidate (default: 3)",
    )
    parser.add_argument(
        "--split",
        type=float,
        default=0.9,
        metavar="FRAC",
        help="train/val split fraction (default: 0.9)",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default="store_sales_arch_robustness",
    )
    parser.add_argument(
        "--candidate-keys",
        nargs="*",
        default=None,
        metavar="D_MODEL:NUM_LAYERS:DIM_FEEDFORWARD",
        help=(
            "explicit architecture(s) to evaluate, e.g. '64:3:64 64:2:512', "
            "bypassing diversity-based selection (overrides --n-candidates)"
        ),
    )
    return parser.parse_args()


def _parse_candidate_key(spec: str) -> tuple[int, int, int]:
    d_model_str, num_layers_str, dim_feedforward_str = spec.split(":")
    return (int(d_model_str), int(num_layers_str), int(dim_feedforward_str))


def main() -> None:
    """Entry point: parse args and run the architecture robustness study."""
    args = parse_args()
    candidate_keys = (
        [_parse_candidate_key(spec) for spec in args.candidate_keys]
        if args.candidate_keys
        else None
    )
    run_study(
        parent_run_name=args.parent_run_name,
        n_candidates=args.n_candidates,
        epochs=args.epochs,
        n_repeats=args.n_repeats,
        split=args.split,
        study_name=args.study_name,
        candidate_keys=candidate_keys,
    )


if __name__ == "__main__":
    main()
