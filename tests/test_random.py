import pytest
import tensorly as tl
from numpy.linalg import matrix_rank

from matcouply.random import random_coupled_matrices
from matcouply.testing import assert_allclose


def test_random_coupled_matrices():
    """test for random.random_coupled_matrices"""
    shapes = [(10, 11), (12, 11), (9, 11), (10, 11), (15, 11)]
    rank = 4

    # Check that assembled matrices have correct shapes and ranks
    coupled_matrices = random_coupled_matrices(shapes, rank, full=True)

    assert len(coupled_matrices) == len(shapes)
    for matrix, shape in zip(coupled_matrices, shapes):
        assert tl.shape(matrix) == shape
        array = tl.to_numpy(matrix)
        assert matrix_rank(array) == rank

    # Check that factor matrices have correct shape
    weights, (A, B_is, C) = random_coupled_matrices(shapes, rank, full=False)
    assert tl.shape(A) == (len(shapes), rank)
    assert all(tl.shape(B_i) == (J_i, rank) for B_i, (J_i, K) in zip(B_is, shapes))
    assert tl.shape(C) == (shapes[0][1], rank)

    # Check that normalising B_is gives B_is with unit normed columns
    weights, (A, B_is, C) = random_coupled_matrices(shapes, rank, full=False, normalise_B=True, normalise_factors=False)
    for B_i in B_is:
        assert_allclose(tl.norm(B_i, axis=0), tl.ones(rank), rtol=1e-6)  # rtol=1e-6 due to PyTorch single precision
    assert_allclose(weights, tl.ones(rank), rtol=1e-6)  # rtol=1e-6 due to PyTorch single precision

    # Check that normalising all factors gives factor matrices with unit normed columns
    weights, (A, B_is, C) = random_coupled_matrices(shapes, rank, full=False, normalise_factors=True)
    assert_allclose(tl.norm(A, axis=0), tl.ones(rank), rtol=1e-6)  # rtol=1e-6 due to PyTorch single precision
    for B_i in B_is:
        assert_allclose(tl.norm(B_i, axis=0), tl.ones(rank), rtol=1e-6)  # rtol=1e-6 due to PyTorch single precision
    assert_allclose(tl.norm(C, axis=0), tl.ones(rank), rtol=1e-6)  # rtol=1e-6 due to PyTorch single precision

    # Should fail when shapes have different value for the number of columns
    shapes = [(10, 10), (12, 11), (9, 11), (10, 11), (15, 11)]
    with pytest.raises(ValueError):
        random_coupled_matrices(shapes, rank)
