"""Tests for tune_store_sales_features constants and FeatureConfig."""

from time_series.store_sales import HOLIDAY_FEATURE_COLS, STORE_FEATURE_COLS
from time_series.tune_store_sales_features import (
    _ARCH_CONFIG,
    FEATURE_CONFIGS,
    FeatureConfig,
)


def test_feature_configs_includes_baseline() -> None:
    names = [cfg.name for cfg in FEATURE_CONFIGS]
    assert "baseline" in names


def test_feature_configs_includes_all_features() -> None:
    names = [cfg.name for cfg in FEATURE_CONFIGS]
    assert "all_features" in names


def test_feature_configs_includes_each_new_feature_group() -> None:
    names = [cfg.name for cfg in FEATURE_CONFIGS]
    assert "oil" in names
    assert "onpromotion" in names
    assert "store_features" in names
    assert "holiday_features" in names


def test_baseline_has_no_features() -> None:
    baseline = next(cfg for cfg in FEATURE_CONFIGS if cfg.name == "baseline")
    assert not baseline.include_oil
    assert not baseline.include_onpromotion
    assert not baseline.store_feature_cols
    assert not baseline.holiday_features


def test_all_features_config_has_everything() -> None:
    all_cfg = next(cfg for cfg in FEATURE_CONFIGS if cfg.name == "all_features")
    assert all_cfg.include_oil
    assert all_cfg.include_onpromotion
    assert set(all_cfg.store_feature_cols or []) == set(STORE_FEATURE_COLS)
    assert set(all_cfg.holiday_features or []) == set(HOLIDAY_FEATURE_COLS)


def test_store_features_config_uses_all_store_cols() -> None:
    cfg = next(c for c in FEATURE_CONFIGS if c.name == "store_features")
    assert set(cfg.store_feature_cols or []) == set(STORE_FEATURE_COLS)


def test_holiday_features_config_uses_all_holiday_cols() -> None:
    cfg = next(c for c in FEATURE_CONFIGS if c.name == "holiday_features")
    assert set(cfg.holiday_features or []) == set(HOLIDAY_FEATURE_COLS)


def test_arch_config_has_required_keys() -> None:
    required = {
        "lr",
        "d_model",
        "nhead",
        "num_layers",
        "batch_size",
        "pooling_mode",
        "dim_feedforward",
    }
    assert required <= _ARCH_CONFIG.keys()


def test_arch_config_d_model_divisible_by_nhead() -> None:
    assert _ARCH_CONFIG["d_model"] % _ARCH_CONFIG["nhead"] == 0


def test_feature_config_as_mlflow_params_baseline() -> None:
    cfg = FeatureConfig(name="baseline")
    params = cfg.as_mlflow_params()
    assert params["feature_config"] == "baseline"
    assert params["include_oil"] is False
    assert params["store_feature_cols"] == "none"
    assert params["holiday_features"] == "none"


def test_feature_config_as_mlflow_params_lists_joined() -> None:
    cfg = FeatureConfig(name="test", store_feature_cols=["city", "type"])
    params = cfg.as_mlflow_params()
    assert params["store_feature_cols"] == "city,type"
