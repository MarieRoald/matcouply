import numpy as np
import pytest
import scipy
import tensorly as tl
from numpy.core.numeric import allclose
from tensorly.testing import assert_array_equal

from matcouply import _utils as utils


def test_is_iterable():
    # Test with some objects that are iterable
    class TestIterable:
        def __iter__(self):
            return self

        def __next__(self):
            return 1

    test_iterables = [
        [1, 2, 3, 4],
        (1, 2, 3, 4),
        "iterablestring",
        {1: "value1", 2: "value2"},
        range(5),
        TestIterable(),
    ]
    for test_iterable in test_iterables:
        assert utils.is_iterable(test_iterable)

    # Test with some objects that arent't iterable
    def test_function(x):
        return x

    class TestNotIterable:
        pass

    test_not_iterables = [1, 3.14, test_function, TestNotIterable()]
    for test_not_iterable in test_not_iterables:
        assert not utils.is_iterable(test_not_iterable)


@pytest.mark.parametrize("svd", ["numpy_svd", "truncated_svd"])
def test_get_svd(rng, svd):
    X = rng.standard_normal(size=(10, 20))
    U1, s1, Vh1 = scipy.linalg.svd(X)
    svd_fun = utils.get_svd(svd)
    U2, s2, Vh2 = svd_fun(tl.tensor(X))
    U2, s2, Vh2 = tl.to_numpy(U2), tl.to_numpy(s2), tl.to_numpy(Vh2)

    # Check singular values are the same
    assert allclose(s1, s2)

    # Check that first 10 (rank) singular vectors are equal or flipped
    U1TU2 = U1.T @ U2
    Vh1Vh2T = Vh1 @ Vh2.T
    Vh1Vh2T = Vh1Vh2T[:10, :10]
    assert allclose(U1TU2, Vh1Vh2T, atol=1e-6)  # low tolerance due to roundoff errors
    assert allclose(U1TU2 * Vh1Vh2T, np.eye(U1TU2.shape[0]))


def test_get_svd_fails_with_invalid_svd_name():
    with pytest.raises(ValueError):
        utils.get_svd("THIS_IS_NOT_A_VALID_SVD")


def test_get_shapes(rng, random_ragged_cmf):
    # Small manual test
    matrices = [tl.zeros((1, 2)), tl.zeros((3, 4)), tl.zeros((5, 6))]
    matrix_shapes = utils.get_shapes(matrices)
    assert matrix_shapes[0] == (1, 2)
    assert matrix_shapes[1] == (3, 4)
    assert matrix_shapes[2] == (5, 6)
    assert len(matrix_shapes) == 3

    # Test on random ragged cmf
    cmf, shapes, rank = random_ragged_cmf
    matrix_shapes = utils.get_shapes(cmf.to_matrices())
    for matrix_shape, shape in zip(matrix_shapes, shapes):
        assert matrix_shape == shape


def test_get_padded_tensor_shape(rng, random_ragged_cmf):
    cmf, shapes, rank = random_ragged_cmf

    I = len(shapes)
    J = max([shape[0] for shape in shapes])
    K = shapes[0][1]

    assert (I, J, K) == utils.get_padded_tensor_shape(cmf.to_matrices())

    matrices_different_columns = [
        tl.tensor(rng.standard_normal(size=(3, 4))),
        tl.tensor(rng.standard_normal(size=(5, 6))),
        tl.tensor(rng.standard_normal(size=(5, 6))),
    ]
    with pytest.raises(ValueError):
        utils.get_padded_tensor_shape(matrices_different_columns)


def test_create_padded_tensor(rng, random_ragged_cmf):
    cmf, shapes, rank = random_ragged_cmf
    matrices = cmf.to_matrices()
    padded_tensor = utils.create_padded_tensor(matrices)

    I = len(shapes)
    J = max([shape[0] for shape in shapes])
    K = shapes[0][1]

    assert (I, J, K) == tl.shape(padded_tensor)

    for i, (matrix, shape) in enumerate(zip(matrices, shapes)):
        assert_array_equal(padded_tensor[i, : shape[0], :], matrix)
        assert_array_equal(padded_tensor[i, shape[0] :, :], 0)


@pytest.mark.parametrize("shape", [(10, 3), (10, 10)])
def test_scipy_svd(rng, shape):
    svd = utils.get_svd("scipy")
    X = rng.standard_normal(shape)
    U, s, Vh = svd(X)
    assert U.shape == (shape[0], shape[0])
    assert s.shape == (min(shape),)
    assert Vh.shape == (shape[1], shape[1])

    U, s, Vh = svd(X, n_eigenvecs=3)
    assert U.shape == (shape[0], 3)
    assert s.shape == (3,)
    assert Vh.shape == (3, shape[1])

    U, s, Vh = svd(X, n_eigenvecs=2)
    assert U.shape == (shape[0], 2)
    assert s.shape == (2,)
    assert Vh.shape == (2, shape[1])
