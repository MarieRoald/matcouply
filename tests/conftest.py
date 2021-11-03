import pytest
import numpy as np
import tensorly as tl


@pytest.fixture
def seed(pytestconfig):
    try:
        return pytestconfig.getoption("randomly_seed")
    except ValueError:
        return 1


@pytest.fixture
def rng(seed):
    return tl.check_random_state(seed)
