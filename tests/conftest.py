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


@pytest.fixture
def random_ragged_shapes(rng):
    I = rng.randint(1, 20)
    K = rng.randint(2, 20)
    shapes = tuple((rng.randint(1, 20), K) for i in range(I))
    return shapes


@pytest.fixture
def random_regular_shapes(rng):
    I = rng.randint(1, 20)
    J = rng.randint(1, 20)
    K = rng.randint(2, 20)
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

    I = rng.randint(1, 20)
    K = rng.randint(2, 20)
    random_ragged_shapes = tuple((rng.randint(5, 20), K) for i in range(I))
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
