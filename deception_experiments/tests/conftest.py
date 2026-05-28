"""
Pytest fixtures for comparison tests between our implementation and Apollo's deception-detection.

Uses the Llama 3.3/70B tokenizer to compare tokenization and detection masks.
"""

import sys
from pathlib import Path

import pytest
from transformers import AutoTokenizer, PreTrainedTokenizerBase

# Add both implementations to the path
PROJECT_ROOT = Path(__file__).parent.parent
APOLLO_ROOT = PROJECT_ROOT / "deception-detection"

# Add paths for imports
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(APOLLO_ROOT))


@pytest.fixture(scope="session")
def llama_tokenizer() -> PreTrainedTokenizerBase:
    """Load Llama 3.3/70B tokenizer (or 3.1 70B which has the same tokenizer).

    Uses left padding as required for both implementations.
    """
    # Use 3.1 70B tokenizer - same as 3.3 70B
    tokenizer_name = "meta-llama/Llama-3.1-70B-Instruct"

    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, padding_side="left")
    except Exception:
        # Fallback to 8B which has the same tokenizer
        tokenizer_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, padding_side="left")

    tokenizer.pad_token_id = tokenizer.bos_token_id
    return tokenizer


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory) -> Path:
    """Create a temporary data directory for tests."""
    return tmp_path_factory.mktemp("data")


@pytest.fixture(scope="session")
def instructed_pairs_data_dir(data_dir) -> Path:
    """Create data directory for instructed_pairs tests."""
    d = data_dir / "instructed_pairs"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture(scope="session")
def roleplaying_data_dir(data_dir) -> Path:
    """Create data directory for roleplaying tests."""
    d = data_dir / "roleplaying-offpolicy_train"
    d.mkdir(parents=True, exist_ok=True)
    return d


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption("--runslow", action="store_true", default=False, help="run slow tests")


def pytest_collection_modifyitems(config, items):
    """Skip slow tests unless --runslow is provided."""
    if not config.getoption("--runslow"):
        skip_slow = pytest.mark.skip(reason="need --runslow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
