"""Single-source OpenMP guard for the torch/lightgbm libomp conflict.

`torch` and `lightgbm` each bundle their own libomp runtime. With both loaded
in one process a plain torch op (e.g. `_to_copy`) deadlocks at the OpenMP
fork/join barrier, and `lgb.LGBMRegressor.fit()` can segfault. Setting a single
OMP thread and tolerating the duplicate runtime avoids both.

These env vars must be set before either library initializes its OMP runtime,
so import this module *first*, before importing torch or lightgbm — its side
effects run at import time. Both the test session (rootdir conftest.py) and the
runtime entry point import it, so the guard lives in exactly one place.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
