# Store Sales Transformer — Notes & Ideas

## Current Architecture

Transformer encoder-decoder (`nn.Transformer`) mapping a `window_lags`-day input window
to an `output_lags`-day forecast over all 54 stores × 33 families.

## Known Issues

### Constant output over time (bug)
The decoder receives an all-zeros target with no positional encoding. Since
`nn.Transformer` adds no PE by default, all `n_output_steps` decoder positions
are identical vectors. Self-attention and cross-attention both collapse to the
same output repeated across time steps.

**Fix:** Switch to encoder-only architecture (see below).

## Ideas to Try

### Encoder-only architecture
Replace `nn.Transformer` with `nn.TransformerEncoder` + `nn.TransformerEncoderLayer`.
Read out the last encoder hidden state (or mean-pool across time) and project to
`n_output_steps * n_stores * n_families` in one linear layer.

Pros: simpler, no decoder pathology, appropriate for fixed-horizon forecasting.

### Positional encoding
`nn.Transformer` adds none by default — required to distinguish time steps.

- **Sinusoidal (Attention Is All You Need):** fixed, generalizes beyond training length
- **Learned embeddings (`nn.Embedding`):** one line, often performs similarly within training distribution
- **RoPE (Rotary Position Embedding):** applied inside attention (rotates Q/K before QK^T),
  encodes relative position directly. Not supported natively by `nn.TransformerEncoderLayer` —
  requires subclassing or a custom attention layer. Good learning exercise.

### Pooling strategy for encoder output
Options for collapsing `[batch, window_lags, d_model]` → `[batch, d_model]`:
- Last token `[:, -1, :]` — convention, biased toward most recent step
- Mean pool `x.mean(dim=1)` — treats all positions equally, probably more principled
- CLS token — prepend a learned token, read from position 0 (BERT-style)
- Flatten + project — retains all info but fixes `window_lags` in the weight shape

### Loss / training
- Current loss: MSLE (competition metric is RMSLE)
- Try per-family loss weighting — some families (PRODUCE, BEVERAGES) dominate volume
- Learning rate schedule (cosine decay, warmup)

### Features
- Currently only uses raw sales values; no exogenous features
- Oil price, holidays, store type/cluster are all available in the dataset
- Day-of-week / month embeddings as additional input channels

## Feature Study — Architecture Robustness Check (2026-07-01)

Goal: tune architecture on the full feature set (oil, onpromotion, store metadata,
holidays all enabled), then use the winning architecture for a feature ablation study.
Scripts: `time_series/tune_store_sales_encoder_only.py` (arch search),
`time_series/tune_store_sales_robustness.py` (this check),
`time_series/tune_store_sales_features.py` (ablation, not yet rerun — see bug below).

### Bug: `optuna.samplers.GridSampler` silently ignores repeats

`GridSampler` stops the study once every grid point has been visited **once**,
regardless of the `n_trials` passed to `study.optimize()`. A grid space of
`{"feature_config_idx": [0..5]}` with `n_trials=18` (intended as 3 repeats × 6 configs)
actually only ran 6 trials — repeats were silently dropped. This affected an earlier
`n-repeats 3` feature-ablation run and the first version of the robustness script;
both reported spurious `std=0.0` (only one sample per group) or, in the ablation case,
just re-ran a second independent single-shot study without anyone noticing.

**Fix:** add a `repeat_idx` dimension to the grid (`{"feature_config_idx": [...],
"repeat_idx": range(n_repeats)}`) so `(config, repeat)` pairs are distinct grid points.
`trial.suggest_categorical("repeat_idx", ...)` is called but its value is otherwise
unused — it exists purely to force GridSampler to actually repeat. Applied in both
`tune_store_sales_features.py` and `tune_store_sales_robustness.py`. Verify trial counts
in MLflow (`len(child_runs)`) match `n_configs * n_repeats` after any GridSampler-based
study — don't trust the requested `n_trials` alone.

### Phase-1 architecture search was noisy (as suspected)

Phase 1 (20 trials, 50 epochs each, one shot per architecture) picked `d64_l2_ff512`
as the winner (val MSLE 1.151) and ranked `d128_l4_ff128` worst (1.828) among 5 diverse
candidates later re-tested. Re-running those 5 candidates for 150 epochs × 3 real
repeats (after the GridSampler fix) gave:

| Candidate       | Mean MSLE | Std   | Best   | Phase-1 (1 run, 50 epochs) |
|-----------------|-----------|-------|--------|------------------------------|
| `d64_l3_ff64`   | **1.166** | 0.135 | 1.063  | 1.378                        |
| `d64_l2_ff512`  | 1.208     | 0.096 | 1.075  | 1.151 (phase-1 "winner")     |
| `d64_l4_ff512`  | 1.370     | 0.020 | 1.356  | 1.343                        |
| `d128_l4_ff128` | 1.409     | 0.051 | 1.338  | 1.828 (phase-1 "worst")      |
| `d128_l2_ff256` | 1.585     | 0.122 | 1.473  | 1.698                        |

Phase-1's best and worst picks both moved — `d128_l4_ff128` went from worst to
mid-pack. **Single-shot 50-epoch trials are not reliable for ranking architectures**;
std (0.02–0.14) is large relative to the spread between candidates. Any future
architecture search should budget for repeats, not just more trials.

New leading candidate: **`d64_l3_ff64`** — small `d_model` (64), moderate depth (3
layers), and notably the *narrowest* feedforward width (64) of the five candidates,
despite phase-1 sampling from `{64, 128, 256, 512}`. Worth trying narrower
`dim_feedforward` values in a future search. All `d_model=64` variants beat their
`d_model=128` counterparts here — bigger `d_model` isn't paying off at this data scale.

### Convergence check: are larger models just undertrained?

Hypothesis was that `d128_*` models look worse only because 150 epochs isn't enough
for their larger parameter count. Checked via the `best_epoch` MLflow tag (epoch of
the best validation loss so far) across all 15 trials, against a rough parameter-count
estimate (`params ≈ num_layers * (4*d_model² + 2*d_model*dim_feedforward) + d_model²`):

| Config          | Params  | best_epoch samples (of 150) |
|-----------------|---------|------------------------------|
| `d64_l3_ff64`   | 77,824  | 138, 148, 147                |
| `d64_l2_ff512`  | 167,936 | 144, 101, 145                |
| `d128_l2_ff256` | 278,528 | 105, 147, 127                |
| `d64_l4_ff512`  | 331,776 | 131, 148, 136                |
| `d128_l4_ff128` | 409,600 | 135, 125, 141                |

Pearson correlation between param count and `best_epoch`: **-0.16** (negligible,
wrong sign for the hypothesis). Larger models are *not* disproportionately
undertrained relative to smaller ones.

However: mean `best_epoch` across **all** 15 trials, regardless of size, is
**134.5 / 150 (90% of budget)**, and 47% of trials were still improving as late as
epoch 140+. This is the more important finding — every architecture tested, big or
small, was still improving near the end of the 150-epoch run. **150 epochs is likely
not enough to fairly rank any of these architectures**, not just the large ones.
Before trusting a final architecture choice, rerun the leading candidates (at least
`d64_l3_ff64` and `d64_l2_ff512`) for more epochs (300+?) to confirm the ranking holds
once training actually converges.

### CLI addition: `--candidate-keys` for targeted confirmation runs

Added `--candidate-keys` to `tune_store_sales_robustness.py` to bypass diversity-based
candidate selection and evaluate specific architectures directly, e.g.:

```bash
uv run python -m time_series.tune_store_sales_robustness \
    --parent-run-name store_sales_allfeatures_tune \
    --candidate-keys 64:3:64 64:2:512 \
    --epochs 300 --n-repeats 3 \
    --study-name store_sales_arch_robustness_confirm
```

Each key is `d_model:num_layers:dim_feedforward`; lr is pulled from the matching
phase-1 trial. Raises if a key isn't among the phase-1 candidates. Useful for
confirming a short list of promising architectures at a larger epoch budget without
re-running the (possibly different) diversity selection.

### 300-epoch confirmation run — ranking flipped AGAIN, still not converged (completed 2026-07-01)

Ran to check whether `d64_l3_ff64` still beats `d64_l2_ff512` once training is closer
to converged (150 epochs wasn't enough — mean best_epoch was 134.5/150 at that budget).
Command used: `--candidate-keys 64:3:64 64:2:512 --epochs 300 --n-repeats 3
--study-name store_sales_arch_robustness_confirm`. MLflow: experiment
`StoreSales_ArchRobustness`, parent run `store_sales_arch_robustness_confirm`.

| Candidate      | Mean MSLE (300ep) | Std   | Best  | Mean MSLE (150ep) | Phase-1 (50ep, 1 run) |
|----------------|--------------------|-------|-------|--------------------|-----------------------|
| `d64_l2_ff512` | **1.113**          | 0.060 | 1.028 | 1.208              | 1.151 ("winner")      |
| `d64_l3_ff64`  | 1.131              | 0.066 | 1.043 | 1.166 ("winner")   | 1.378                 |

**The ranking flipped a third time.** `d64_l2_ff512` was the phase-1 winner (50ep),
lost to `d64_l3_ff64` at 150ep, and reclaimed the lead at 300ep — though the two are
now close (1.113 vs 1.131, within ~1 std of each other) rather than clearly separated.
Both candidates also improved substantially in absolute terms from 150→300 epochs
(`d64_l2_ff512`: 1.208→1.113; `d64_l3_ff64`: 1.166→1.131), confirming 150 epochs was
too short for either.

**Still not converged at 300 epochs either.** `best_epoch` tag per trial:
`d64_l3_ff64` → 287, 295, 295; `d64_l2_ff512` → 271, 251, 289. Mean = 281.3/300
(94% of budget), essentially the same fraction as the 150-epoch run (134.5/150 = 90%).
This is a **recurring pattern, not a fluke**: whatever epoch budget we use, these
architectures are still improving in the final ~10% of training. Bumping the epoch
count alone doesn't converge the comparison — it just re-triggers the same issue
one order of magnitude later. Likely needs either a much larger budget (600+?), a
different stopping criterion (train to a plateau, e.g. early-stopping on a patience
window, rather than a fixed epoch count), or a learning-rate schedule (cosine decay)
to force convergence within a reasonable budget — current runs use a fixed lr for
the entire run.

**Given the closeness of the two candidates and the noise budget (std ~0.06),
`d64_l2_ff512` and `d64_l3_ff64` should be treated as roughly tied** rather than
picking one as a confident winner. Either is a reasonable choice for the feature
ablation study; `d64_l2_ff512` has a slight edge in the most-converged data we have.

### Next steps
1. ~~Re-run robustness check with a larger epoch budget~~ — done; ranking flipped
   again and convergence still not reached. Not worth a fourth blind epoch-count
   bump — fix the underlying training setup instead (see below).
2. Add an LR schedule (cosine decay) and/or early stopping with a patience window,
   so "best_epoch near the budget ceiling" stops being the default outcome. This is
   the real blocker to a trustworthy architecture comparison, more so than picking
   the "right" epoch count.
3. Try narrower `dim_feedforward` values (32? 16?) given `ff64` outperformed `ff512`
   at `d_model=64` at 150 epochs (though the 300-epoch result narrows that gap).
4. Rerun the feature ablation study (`tune_store_sales_features.py`, now fixed) with
   `d64_l2_ff512` (slight edge) or `d64_l3_ff64` (near-tied) and real repeats — but
   consider doing this after #2, since an LR schedule would likely change the
   architecture ranking yet again.

### Cosine annealing added; confirmation rerun — convergence signal cleaned up (2026-07-01/02)

Added `CosineAnnealingLR(self.optim, T_max=epochs)` to `Trainer.train()` in
`store_sales.py`, stepped once per epoch after that epoch's `optim.step()` calls, lr
decaying from the configured `lr` down to ~0 over the run. This lives in the shared
`Trainer`/`train_and_eval` path, so it automatically applies to the main script and
all three tuning scripts (`tune_store_sales_encoder_only.py`,
`tune_store_sales_features.py`, `tune_store_sales_robustness.py`) with no changes to
those files. Per-epoch `lr` is now also logged to MLflow for visibility. All 125
existing tests pass (one test that fully mocks `train_loop`/`val_loop` triggers
PyTorch's harmless "scheduler.step() before optimizer.step()" warning, since with
those loops mocked out `optim.step()` never actually runs — not a real issue).

Reran the same confirmation study with cosine annealing
(`--candidate-keys 64:3:64 64:2:512 --epochs 300 --n-repeats 3
--study-name store_sales_arch_robustness_confirm_cosine`):

| Candidate      | Mean MSLE | Std   | Best  | Prior (300ep, no schedule) |
|----------------|-----------|-------|-------|------------------------------|
| `d64_l2_ff512` | **1.101** | 0.100 | 1.016 | 1.113                        |
| `d64_l3_ff64`  | 1.299     | 0.085 | 1.181 | 1.131                        |

`d64_l2_ff512` now clearly wins (previously near-tied). `best_epoch` per trial:
`d64_l3_ff64` → 172, 249, 190; `d64_l2_ff512` → 150, 233, 214. **Mean = 201.3/300
(67% of budget)**, down from 94% in both prior fixed-lr runs — and the spread (150–249)
is much wider than the fixed-lr runs' tight clustering near the ceiling (251–295).
This is exactly the signature predicted: fixed lr was sampling a noise ball whose
"best" reading kept landing near the tail by chance; annealing lr lets the runs
actually settle, so `best_epoch` now varies meaningfully by trial instead of being
pinned to the budget ceiling. The earlier "still improving, may be undertrained"
diagnosis was the noise-ball artifact, not genuine undertraining — mystery resolved.

**Conclusion: `d64_l2_ff512` is the confirmed architecture** — winner at every stage
except the noisy 150-epoch checkpoint. Recommended config: `lr` from its phase-1
trial, `d_model=64`, `nhead=2`, `num_layers=2`, `dim_feedforward=512`,
`batch_size=64`, `pooling_mode=PoolingMode.ALL`, cosine-annealed over the training run.

### Next steps (updated)
1. ~~Confirm architecture ranking with a proper LR schedule~~ — done, see above.
   `d64_l2_ff512` confirmed.
2. ~~Rerun the feature ablation study with `d64_l2_ff512` and real repeats~~ — done,
   see below.
3. Optional/lower priority: try narrower `dim_feedforward` values now that the
   comparison methodology is trustworthy — the earlier "ff64 beats ff512" signal
   didn't hold up once trained properly, so this is exploratory rather than
   following up on a real lead.

## Feature Ablation Study — final results (2026-07-02)

`_ARCH_CONFIG` in `tune_store_sales_features.py` updated to the confirmed architecture
(`d64_l2_ff512`, `lr=0.0011852762898126566` from its matching phase-1 trial). Ran with
`--epochs 300 --n-repeats 3` (18 trials: 6 feature configs × 3 real repeats, GridSampler
repeat fix + cosine annealing both in effect). MLflow: experiment
`StoreSales_FeatureAblation`, parent run `store_sales_feature_ablation`.

The script's own summary table only reports best-of-3 (min), which is exactly the kind
of single-number-without-variance we've learned not to trust — pulled mean/std directly
from the 18 child runs instead:

| Config              | Mean MSLE | Std   | Best  | best_epoch samples (of 300) |
|---------------------|-----------|-------|-------|-------------------------------|
| **all_features**    | **0.996** | 0.120 | 0.887 | 243, 189, 206                 |
| onpromotion         | 1.107     | 0.091 | 0.979 | 220, 270, 263                 |
| oil                 | 1.171     | 0.023 | 1.138 | 255, 204, 207                 |
| holiday_features    | 1.184     | 0.138 | 1.010 | 270, 183, 298                 |
| store_features      | 1.198     | 0.029 | 1.177 | 262, 260, 298                 |
| baseline            | 1.203     | 0.096 | 1.077 | 278, 297, 272                 |

**All features combined is a clear, robust win**: 0.996 vs baseline's 1.203 — a gap
(~0.21) much larger than any individual config's std, so this holds up. `onpromotion`
is the strongest single feature by a clear margin over the rest of the individual
groups. The middle group (`oil`, `holiday_features`, `store_features`) mostly overlaps
within noise of each other and of `baseline` — **not** confident in a ranking among
those three individually; only the top (all_features) and the onpromotion signal are
trustworthy conclusions from this data.

**Convergence caveat**: mean `best_epoch` across all 18 trials is 248.6/300 (83%).
Some configs converged well before the ceiling (`oil`: 204, 207; `all_features`: 189,
206) while others are still sitting near it (`baseline`: 297, 298; `store_features`:
298; `holiday_features`: 298). Given the architecture-search experience, configs still
pinned near the ceiling may have somewhat unreliable "best" readings — this affects
`baseline` and `store_features` most, and could modestly understate `holiday_features`
too (one of its three repeats hit 298). It's less likely to change the headline
conclusion (all_features wins clearly) but could shift the ordering of the
already-noisy middle group.

### Recommendation
Use all features (oil, onpromotion, store metadata, holidays) — the combined-feature
config is a clean, well-separated win. If further precision on individual feature
contributions is wanted, a follow-up run with a larger epoch budget for just
`baseline`/`store_features`/`holiday_features` (the configs still near the ceiling)
would be needed; not done here since the primary question (do the features help at
all?) is already answered decisively.

## Factorized Encoder Attempt — Architecture Post-Mortem (2026-07-11)

### What it tried to do

`StoreSalesHierarchicalEncoder` in `models.py` attempted a two-stage design:

1. **Time encoder** — process each (store, family) time series independently with a
   `TransformerEncoder` operating over the `T` time steps, then mean-pool to get one
   embedding per (store, family).
2. **SF encoder** — run a second `TransformerEncoder` across all 54×33 = 1782
   (store, family) tokens to model cross-series dependencies, then project to the
   output horizon.

The motivation: separate "what does this store-family's history look like" from "how
do stores and families relate to each other," rather than conflating both in a single
encoder.

### Why it failed

**Implementation bug: batch explosion.**  To run the time encoder over all
(store, family) series in parallel, the implementation flattened the S×F spatial
dimensions into the batch axis. With default `batch_size=64`:

$$\text{effective batch} = B \times n_{\text{stores}} \times n_{\text{families}} = 64 \times 54 \times 33 = 114{,}048$$

CUDA's flash attention and memory-efficient attention kernels map the batch
dimension to `grid.y`, which has a hardware maximum of **65535** on all NVIDIA GPUs.
At 114k, the kernel launch is unconditionally rejected with
`cudaErrorInvalidConfiguration` — this happens regardless of dtype, head count, or
sequence length. The fallback (math attention) works because standard `torch.matmul`
uses `grid.x` (max ~$2^{31}$), not `grid.y`. The workaround (`sdp_kernel` context
manager forcing math-only for the time encoder) was committed, but the fix only
papers over the symptom.

The SF encoder (batch=64, seq=1782) works fine — flash attention handles long
sequences well, it's the large batch that breaks things.

**Inductive bias vs. cost.** Even if the CUDA issue were fixed (e.g. by reducing
batch size so B×S×F ≤ 65535, which means `batch_size ≤ 36`), the architecture is
expensive: `O(T^2)` attention per (store, family) series at time encoding, then
`O((S \cdot F)^2)` at the spatial stage. The spatial stage with seq=1782 is the
dominant cost; math attention would materialize a 1782×1782 matrix per batch
(~1.5 GB at float32, batch=64). Flash handles this but it is slow.

More importantly: **the encoder-only model already captures cross-store patterns via
the feature channels** (store metadata, promotions, holidays are all included as
feature channels). The temporal attention over a shared feature vector isn't obviously
worse at learning cross-store correlations than explicit spatial attention.

### Where to go from here

**Option A: Better features + same architecture.** The encoder-only model at MSLE
~0.996 (all features, 300 epochs) still has headroom. It converges at ~83% of budget
and the variance across seeds is 0.12 — more epochs, a wider sweep of `lr` and
`dim_feedforward`, or gradient clipping may push it further. This is the path of
least resistance.

**Option B: Per-series models with lag features (likely the highest ceiling).**
Kaggle time series competitions are typically won by gradient-boosted trees (LightGBM,
XGBoost) with hand-crafted lag features (14-day lags, rolling means, seasonal
differences). These models handle the 54×33 independent-series structure naturally,
require no GPU, train in minutes, and routinely outperform deep models on this
competition. Worth implementing as a strong baseline before investing more in
transformer architectures.

**Option C: Purpose-built time series transformers.** Several architectures
specifically address the high-variate, long-horizon forecasting problem:

- **PatchTST** (Nie et al., 2023): tokenizes the input into non-overlapping patches
  (e.g. 16-day windows) → reduces sequence length from T to T/patch_size. Attends
  independently across time within each variate (avoids the S×F batch explosion by
  treating each variate as an independent sequence, not a batch dimension). Strong on
  ETTh/ETTm benchmarks. Available via `neuralforecast`.
- **iTransformer** (Liu et al., 2024): inverts the attention — treats each variate's
  full time series as a single token, attends across variates (S×F tokens) instead of
  across time. Sequence length = S×F = 1782, attention captures cross-series
  correlations directly. Avoids the time-axis attention cost entirely.
- **N-BEATS**: purely MLP-based, no attention, interpretable trend/seasonality
  decomposition. Surprisingly competitive, trains fast, worth trying as a neural
  baseline before heavy transformers.
- **TFT (Temporal Fusion Transformer)**: designed for multi-horizon forecasting with
  static covariates, known/unknown future inputs. Closer to what this dataset
  actually looks like (some covariates — holidays, paydays — are known in advance).

**Option D: Fix the factorized encoder properly.** Rather than flattening S×F into
the batch axis, process the time encoder with a true 3D operation: reshape the input
to `[B, T, S*F*X]`, apply Conv1d over the feature axis to get `[B, T_reduced, S*F]`,
then attend over T within each of the B samples (not B×S×F). This avoids the batch
explosion entirely. The downside: per-(store,family) temporal patterns are no longer
computed independently — they share weights across space in the temporal stage, which
may or may not be an appropriate inductive bias for this dataset.

### Recommendation

Invest in Option B (LightGBM baseline) first — it's a 1-day effort and will give a
meaningful upper-bound signal. Then try Option C (PatchTST or iTransformer via
`neuralforecast`) if the LightGBM ceiling still has room. Don't invest more in the
current encoder-only architecture without first knowing what the non-neural upper
bound looks like.

### Partial training run (killed at epoch 53/300, 2026-07-11)

Ran on a RunPod A4000 (sm86). With the `sdp_kernel` math-only workaround in place for
the time encoder, training started and produced real loss values — no more CUDA errors.
But it was slow:

- **35.8 samples/s** — ~36 sec/epoch → ~3 hours for a full 300-epoch run
- val_loss trajectory: 13.35 → 6.52 over 53 epochs (still clearly descending)
- Run killed at epoch 53; not worth the compute cost given the trajectory

The slow throughput is the math-attention tax. Flash and memory-efficient attention
both fail for batch=114,048 on sm86 (see above); math attention materializes the full
`[114048, 2, 12, 12]` QK^T matrix every step. The encoder-only's `[64, 60, 64]`
attention uses flash and trains far faster.

At epoch 53, val_loss was 6.52. The encoder-only model reached 0.887 after 300 epochs
with all features. Even extrapolating the factorized encoder's descent curve
generously, it's unlikely to beat the encoder-only within a comparable compute budget.

**MLflow run:** experiment `StoreSales_FactorizedEncoder`, run
`43af03e57f45468d863dca6694fa318f` (status left as RUNNING — killed mid-run, treat
metrics as incomplete).
