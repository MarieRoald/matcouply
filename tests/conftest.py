import pytest
import numpy as np


@pytest.fixture
def seed(pytestconfig):
    try:
        return pytestconfig.getoption("randomly_seed")
    except ValueError:
        return 1


@pytest.fixture
def rng(seed):
    try:
        return np.random.default_rng(seed=seed)
    except AttributeError:
        return np.random.RandomState(seed=seed)
