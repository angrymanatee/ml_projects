"""Root conftest: neutralize the LightGBM/PyTorch OpenMP conflict.

`lightgbm` and `torch` each bundle their own libomp runtime. With both loaded
in one process two distinct failures appear, and each needs its own guard:

  * `lgb.LGBMRegressor.fit()` segfaults if torch initializes its OMP runtime
    first. Importing lightgbm here, before pytest collects (and therefore
    imports) any test module, guarantees it loads first.
  * A plain torch op (e.g. `_to_copy`) otherwise deadlocks at the OpenMP
    fork/join barrier, hanging the whole session. Forcing a single OMP thread
    and tolerating the duplicate runtime avoids that; both env vars must be set
    before either library initializes its OMP runtime.

The rootdir conftest is the earliest hook that runs for every test session.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import lightgbm  # noqa: E402, F401
