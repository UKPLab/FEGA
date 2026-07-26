import sys
from pathlib import Path

import pytest

EXTERNAL_ROOT = Path(__file__).resolve().parents[1] / "external"
sys.path.insert(0, str(EXTERNAL_ROOT))


def pytest_addoption(parser):
    parser.addoption(
        "--run-local-gpt2-gpu",
        action="store_true",
        default=False,
        help="Run opt-in local GPT-2 GPU smoke tests.",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-local-gpt2-gpu"):
        return
    skip = pytest.mark.skip(reason="requires --run-local-gpt2-gpu")
    for item in items:
        if "local_gpt2_gpu" in item.keywords:
            item.add_marker(skip)
