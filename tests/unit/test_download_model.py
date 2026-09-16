import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from scripts.download_model import _env


def test_env_defaults(monkeypatch):
    monkeypatch.delenv("MODEL_HF_REPO", raising=False)
    assert _env("MODEL_HF_REPO", "ibm-granite/granite-3.0-2b-instruct") == (
        "ibm-granite/granite-3.0-2b-instruct"
    )


def test_env_reads_override(monkeypatch):
    monkeypatch.setenv("MODEL_HF_REPO", "custom/model")
    assert _env("MODEL_HF_REPO", "default/model") == "custom/model"
