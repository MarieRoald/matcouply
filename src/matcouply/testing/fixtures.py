# MIT License: Copyright (c) 2022, Marie Roald.
# See the LICENSE file in the root directory for full license text.

import pytest
from .utils import random_length


@pytest.fixture
def seed(pytestconfig):
    try:
        return pytestconfig.getoption("randomly_seed")
    except ValueError:
        return 1


@pytest.fixture
def rng(seed):
    import tensorly as tl

    return tl.check_random_state(seed)


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


@pytest.fixture
def random_matrix(rng):
    import tensorly as tl

    return tl.tensor(rng.standard_normal((10, 3)))


@pytest.fixture
def random_matrices(rng):
    import tensorly as tl

    return [tl.tensor(rng.standard_normal((10, 3))) for i in range(5)]
