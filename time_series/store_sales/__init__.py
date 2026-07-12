from .backtest import BacktestConfig, BacktestResult, Forecaster, backtest
from .data import (
    HOLIDAY_FEATURE_COLS,
    STORE_FEATURE_COLS,
    EarthquakeEncoding,
    StoreData,
)
from .models import (
    HoldLastValue,
    PoolingMode,
    StoreSalesEncoderOnly,
    StoreSalesHierarchicalEncoder,
    StoreSalesTransformer,
)
from .runners import Trainer, get_device, make_loaders, run
from .tabular import FeatureConfig

# LightGBMForecaster/LGBMParams are intentionally NOT re-exported here: importing
# them pulls in lightgbm (and a second libomp runtime), which deadlocks with
# torch's OpenMP unless guarded. Keeping them in the .lgbm submodule avoids
# forcing that coupling onto every `import time_series.store_sales` (e.g. the
# torch-only transformer path). Import from time_series.store_sales.lgbm directly.

__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "EarthquakeEncoding",
    "FeatureConfig",
    "Forecaster",
    "HOLIDAY_FEATURE_COLS",
    "HoldLastValue",
    "PoolingMode",
    "STORE_FEATURE_COLS",
    "StoreSalesEncoderOnly",
    "StoreSalesHierarchicalEncoder",
    "StoreSalesTransformer",
    "StoreData",
    "Trainer",
    "backtest",
    "get_device",
    "make_loaders",
    "run",
]
