"""Root conftest: neutralize the LightGBM/PyTorch OpenMP conflict.

`lightgbm` and `torch` each bundle their own libomp runtime. With both loaded
in one process two distinct failures appear, and each needs its own guard:

  * `lgb.LGBMRegressor.fit()` segfaults if torch initializes its OMP runtime
    first. Importing lightgbm here, before pytest collects (and therefore
    imports) any test module, guarantees it loads first.
  * A plain torch op (e.g. `_to_copy`) otherwise deadlocks at the OpenMP
    fork/join barrier, hanging the whole session. common.openmp_guard sets the
    OMP env vars (single thread + duplicate-runtime tolerance) that avoid that.

Both imports must precede any torch/lightgbm load, so isort must not reorder
them. The rootdir conftest is the earliest hook that runs for every session.
"""

# isort: off
import common.openmp_guard  # noqa: F401  (sets OMP env before torch/lightgbm load)
import lightgbm  # noqa: F401
# isort: on
