import pytest
import numpy as np
import tensorly as tl
from cm_aoadmm.random import random_coupled_matrices


@pytest.fixture
def seed(pytestconfig):
    try:
        return pytestconfig.getoption("randomly_seed")
    except ValueError:
        return 1


@pytest.fixture
def rng(seed):
    return tl.check_random_state(seed)

@pytest.fixture
def random_ragged_shapes(rng):
    I = rng.randint(1, 20)
    K = rng.randint(1, 20)
    shapes = tuple((rng.randint(1, 20), K) for i in range(I))
    return shapes

@pytest.fixture
def random_regular_shapes(rng):
    I = rng.randint(1, 20)
    J = rng.randint(1, 20)
    K = rng.randint(1, 20)
    shapes = tuple((J, K) for i in range(I))
    return shapes
    
@pytest.fixture
def random_rank(rng):
    rank = rng.randint(1, 6)
    return rank

@pytest.fixture
def random_ragged_cmf(rng, random_ragged_shapes, random_rank):
    cmf = random_coupled_matrices(random_ragged_shapes, random_rank, random_state=rng)
    return cmf, random_ragged_shapes, random_rank


@pytest.fixture
def random_regular_cmf(rng, random_regular_shapes, random_rank):
    cmf = random_coupled_matrices(random_regular_shapes, random_rank, random_state=rng)
    return cmf, random_regular_shapes, random_rank
