import pytest
import tensorly as tl

from matcouply.random import random_coupled_matrices


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
def random_ragged_cmf(
    rng, random_ragged_shapes,
):
    smallest_J = min(shape[0] for shape in random_ragged_shapes)
    rank = rng.randint(1, smallest_J + 1)
    cmf = random_coupled_matrices(random_ragged_shapes, rank, random_state=rng)
    return cmf, random_ragged_shapes, rank


@pytest.fixture
def random_rank5_ragged_cmf(rng):
    I = rng.randint(1, 20)
    K = rng.randint(1, 20)
    random_ragged_shapes = tuple((rng.randint(5, 20), K) for i in range(I))
    rank = 5
    cmf = random_coupled_matrices(random_ragged_shapes, rank, random_state=rng)
    return cmf, random_ragged_shapes, rank


@pytest.fixture
def random_regular_cmf(
    rng, random_regular_shapes,
):
    rank = rng.randint(1, random_regular_shapes[0][0])
    cmf = random_coupled_matrices(random_regular_shapes, rank, random_state=rng)
    return cmf, random_regular_shapes, rank
