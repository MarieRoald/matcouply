import numpy as np
import pytest
import scipy
from numpy.core.numeric import allclose

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
    U2, s2, Vh2 = svd_fun(X)

    # Check singular values are the same
    assert allclose(s1, s2)

    # Check that first 10 (rank) singular vectors are equal or flipped
    U1TU2 = U1.T @ U2
    Vh1Vh2T = Vh1 @ Vh2.T
    Vh1Vh2T = Vh1Vh2T[:10, :10]
    assert allclose(U1TU2, Vh1Vh2T)
    assert allclose(U1TU2 * Vh1Vh2T, np.eye(U1TU2.shape[0]))


def test_get_svd_fails_with_invalid_svd_name():
    with pytest.raises(ValueError):
        utils.get_svd("THIS_IS_NOT_A_VALID_SVD")
