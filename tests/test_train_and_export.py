import importlib
import sys
from types import ModuleType, SimpleNamespace
import numpy as np
from unittest.mock import mock_open

import pytest


@pytest.mark.parametrize("dummy", [None])
def test_train_and_export_runs_without_side_effects(monkeypatch, dummy):
    # Avoid filesystem writes
    monkeypatch.setattr("pathlib.Path.mkdir", lambda self, parents=True, exist_ok=True: None)
    monkeypatch.setattr("pathlib.Path.exists", lambda self: True)
    monkeypatch.setattr("pathlib.Path.stat", lambda self, *args, **kwargs: SimpleNamespace(st_size=1, st_mode=0))
    monkeypatch.setattr("builtins.open", mock_open())

    # Stub persistence and copying
    monkeypatch.setattr("numpy.save", lambda *args, **kwargs: None)
    monkeypatch.setattr("numpy.load", lambda *args, **kwargs: np.zeros(1))
    monkeypatch.setattr("joblib.dump", lambda *args, **kwargs: None)
    monkeypatch.setattr("shutil.copytree", lambda *args, **kwargs: None)

    # Stub gameplay to avoid real training and NotFitted errors
    monkeypatch.setattr("crib_ai_trainer.Arena.Arena.playHands", lambda self, n: ([], [], [0] * max(1, n)))

    # Stub matplotlib so Myrmidon import does not touch real backend
    mock_matplotlib = ModuleType("matplotlib")
    mock_pyplot = ModuleType("matplotlib.pyplot")
    sys.modules["matplotlib"] = mock_matplotlib
    sys.modules["matplotlib.pyplot"] = mock_pyplot

    # Ensure a clean import triggers the script once under these stubs
    sys.modules.pop("scripts.train_and_export", None)
    module = importlib.import_module("scripts.train_and_export")

    # Basic sanity: module was imported and configured a training round count
    assert getattr(module, "TRAINING_ROUNDS", None) is not None
