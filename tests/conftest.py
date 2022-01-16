import pytest
import tensorly as tl

# We import matcouply only inside the fixtures to avoid any import-time
# side-effects before pytest_configure is ran. This is necessary to disable
# jitting of the tests.


def pytest_configure(config):
    import os

    # Disable JIT for unit tests
    os.environ["NUMBA_DISABLE_JIT"] = "1"

    # Anaconda on Windows can have problems with multiple linked OpenMP dlls. This (unsafe) workaround makes it possible to run code with multiple linked OpenMP dlls.
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


@pytest.fixture
def seed(pytestconfig):
    try:
        return pytestconfig.getoption("randomly_seed")
    except ValueError:
        return 1


@pytest.fixture
def rng(seed):
    return tl.check_random_state(seed)


def random_length(rng, min=2, mean=5):
    """Generate a random dimension length.
    
    Use Poisson distribution since it is discrete and centered around the mean.
    """
    if min >= mean:
        raise ValueError("Min must be less than mean.")
    return min + round(rng.poisson(mean - min))


@pytest.fixture
def random_ragged_shapes(rng):
    I = random_length(rng)
    K = random_length(rng)

    shapes = tuple((random_length(rng), K) for i in range(I))
    return shapes


@pytest.fixture
def random_regular_shapes(rng):
    I = random_length(rng)
    J = random_length(rng)
    K = random_length(rng)
    shapes = tuple((J, K) for i in range(I))
    return shapes


@pytest.fixture
def random_ragged_cmf(
    rng, random_ragged_shapes,
):
    from matcouply.random import random_coupled_matrices

    smallest_J = min(shape[0] for shape in random_ragged_shapes)
    rank = rng.randint(1, smallest_J + 1)
    cmf = random_coupled_matrices(random_ragged_shapes, rank, random_state=rng)
    return cmf, random_ragged_shapes, rank


@pytest.fixture
def random_rank5_ragged_cmf(rng):
    from matcouply.random import random_coupled_matrices

    I = random_length(rng)
    K = random_length(rng)
    random_ragged_shapes = tuple((random_length(rng, min=5, mean=7), K) for i in range(I))
    rank = 5
    cmf = random_coupled_matrices(random_ragged_shapes, rank, random_state=rng)
    return cmf, random_ragged_shapes, rank


@pytest.fixture
def random_regular_cmf(
    rng, random_regular_shapes,
):
    from matcouply.random import random_coupled_matrices

    rank = rng.randint(1, random_regular_shapes[0][0] + 1)
    cmf = random_coupled_matrices(random_regular_shapes, rank, random_state=rng)
    return cmf, random_regular_shapes, rank
