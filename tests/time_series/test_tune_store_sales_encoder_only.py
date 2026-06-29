"""Tests for tune_store_sales_encoder_only.build_config."""

import unittest.mock

from time_series.main_store_sales_encoder_only import PoolingMode
from time_series.tune_store_sales_encoder_only import build_config


def _make_mock_trial() -> unittest.mock.MagicMock:
    mock_trial = unittest.mock.MagicMock()

    def categorical_side_effect(name: str, choices: list) -> object:
        return {
            "nhead": 4,
            "d_model_per_head": 32,
            "batch_size": 128,
            "pooling_mode": "last",
        }[name]

    mock_trial.suggest_categorical.side_effect = categorical_side_effect
    mock_trial.suggest_float.return_value = 1e-3
    mock_trial.suggest_int.return_value = 3
    return mock_trial


def test_build_config_d_model_divisible_by_nhead() -> None:
    config = build_config(_make_mock_trial())
    assert config["d_model"] % config["nhead"] == 0


def test_build_config_has_required_keys() -> None:
    config = build_config(_make_mock_trial())
    assert {
        "lr",
        "d_model",
        "nhead",
        "num_layers",
        "batch_size",
        "pooling_mode",
    } <= config.keys()


def test_build_config_pooling_mode_is_enum() -> None:
    config = build_config(_make_mock_trial())
    assert isinstance(config["pooling_mode"], PoolingMode)
