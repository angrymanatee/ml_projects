from pathlib import Path

import pytest

from remote.config import load_config


def test_load_config_reads_api_key_from_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RUNPOD_API_KEY", "test_key_abc")
    config_file = tmp_path / "runpod_config.yaml"
    config_file.write_text("gpu_type: 'NVIDIA A100'\ngpu_count: 2\n")
    config = load_config(config_file)
    assert config.api_key == "test_key_abc"
    assert config.gpu_type == "NVIDIA A100"
    assert config.gpu_count == 2


def test_load_config_uses_defaults_when_no_yaml(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RUNPOD_API_KEY", "test_key")
    config = load_config(tmp_path / "nonexistent.yaml")
    assert config.on_complete == "terminate"
    assert config.mlflow_port == 5000
    assert config.default_datasets == []
    assert config.gpu_count == 1


def test_load_config_fails_without_api_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="RUNPOD_API_KEY"):
        load_config(tmp_path / "config.yaml")


def test_load_config_overrides_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RUNPOD_API_KEY", "key")
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "on_complete: stop\nmlflow_port: 5001\ndefault_datasets:\n  - store-sales\n"
    )
    config = load_config(config_file)
    assert config.on_complete == "stop"
    assert config.mlflow_port == 5001
    assert config.default_datasets == ["store-sales"]


def test_load_config_raises_on_unknown_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RUNPOD_API_KEY", "key")
    config_file = tmp_path / "config.yaml"
    config_file.write_text("gpu_typ: 'NVIDIA A100'\ninvalid_field: true\n")
    with pytest.raises(ValueError, match="Unknown keys"):
        load_config(config_file)
