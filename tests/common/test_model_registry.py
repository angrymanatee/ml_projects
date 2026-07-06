import importlib
import os

import pytest


def test_tracking_uri_default() -> None:
    import common.model_registry as registry

    importlib.reload(registry)
    assert (
        os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
        == registry.TRACKING_URI
    )


def test_tracking_uri_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://remote-host:5000")
    import common.model_registry as registry

    importlib.reload(registry)
    assert registry.TRACKING_URI == "http://remote-host:5000"
