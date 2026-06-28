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
