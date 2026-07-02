"""Tests for tune_store_sales_robustness candidate selection."""

from time_series.tune_store_sales_robustness import (
    ArchCandidate,
    select_diverse_candidates,
)


def _candidate(
    d_model: int, num_layers: int, dim_feedforward: int, loss: float
) -> ArchCandidate:
    return ArchCandidate(
        d_model=d_model,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        lr=1e-3,
        phase1_best_val_loss=loss,
    )


def test_select_diverse_candidates_always_includes_best() -> None:
    candidates = [
        _candidate(64, 2, 128, loss=1.5),
        _candidate(64, 2, 256, loss=1.2),
        _candidate(128, 4, 512, loss=1.8),
    ]
    selected = select_diverse_candidates(candidates, n_candidates=2)
    assert any(c.phase1_best_val_loss == 1.2 for c in selected)


def test_select_diverse_candidates_returns_requested_count() -> None:
    candidates = [
        _candidate(64, 2, 64, loss=1.0),
        _candidate(64, 2, 128, loss=1.1),
        _candidate(64, 4, 128, loss=1.2),
        _candidate(128, 2, 256, loss=1.3),
        _candidate(128, 4, 512, loss=1.4),
    ]
    selected = select_diverse_candidates(candidates, n_candidates=3)
    assert len(selected) == 3


def test_select_diverse_candidates_returns_all_when_fewer_than_requested() -> None:
    candidates = [
        _candidate(64, 2, 128, loss=1.0),
        _candidate(128, 4, 512, loss=1.2),
    ]
    selected = select_diverse_candidates(candidates, n_candidates=5)
    assert len(selected) == 2


def test_select_diverse_candidates_no_duplicates() -> None:
    candidates = [
        _candidate(64, 2, 64, loss=1.0),
        _candidate(64, 2, 128, loss=1.1),
        _candidate(64, 4, 128, loss=1.2),
        _candidate(128, 2, 256, loss=1.3),
        _candidate(128, 4, 512, loss=1.4),
    ]
    selected = select_diverse_candidates(candidates, n_candidates=4)
    keys = [c.key for c in selected]
    assert len(keys) == len(set(keys))


def test_select_diverse_candidates_spreads_across_extremes() -> None:
    candidates = [
        _candidate(64, 2, 64, loss=1.0),  # best, smallest
        _candidate(64, 2, 128, loss=1.05),  # near-duplicate of best
        _candidate(64, 3, 128, loss=1.1),
        _candidate(128, 4, 512, loss=1.5),  # largest, farthest from best
    ]
    selected = select_diverse_candidates(candidates, n_candidates=2)
    selected_keys = {c.key for c in selected}
    assert (64, 2, 64) in selected_keys
    assert (128, 4, 512) in selected_keys


def test_arch_candidate_config_has_required_keys() -> None:
    candidate = _candidate(64, 2, 128, loss=1.0)
    config = candidate.as_config()
    required = {
        "lr",
        "d_model",
        "nhead",
        "num_layers",
        "batch_size",
        "pooling_mode",
        "dim_feedforward",
    }
    assert required <= config.keys()


def test_arch_candidate_name_is_descriptive() -> None:
    candidate = _candidate(64, 2, 128, loss=1.0)
    assert candidate.name == "d64_l2_ff128"
