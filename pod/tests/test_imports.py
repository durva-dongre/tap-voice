import importlib

import pytest

MODULES = [
    "app.audio",
    "app.codec",
    "app.config",
    "app.control",
    "app.engine",
    "app.logits",
    "app.manifest",
    "app.pipeline",
    "app.prompt",
    "app.telemetry",
    "app.voices",
]


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name):
    importlib.import_module(name)


def test_sampling_restrictions_shapes():
    from app.logits import sampling_restrictions

    assert "allowed_token_ids" in sampling_restrictions("allowed")
    assert sampling_restrictions("none") == {}
