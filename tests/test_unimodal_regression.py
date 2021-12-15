from math import ceil
from unittest.mock import patch

import numpy as np
import pytest
from pytest import approx
from sklearn.isotonic import IsotonicRegression
from tensorly.testing import assert_array_equal

from matcouply._unimodal_regression import (
    _compute_isotonic_from_index,
    _unimodal_regression,
    prefix_isotonic_regression,
    unimodal_regression,
)


@pytest.mark.parametrize("shape", [5, 10, 50, 100, 500])
@pytest.mark.parametrize("std", [1, 2, 3])
@pytest.mark.parametrize("non_negativity", [False, True])
def test_isotonic_regression(shape, std, non_negativity):
    np.random.seed(0)
    x = np.arange(shape).astype(float)
    y = x + np.random.standard_normal(shape) * std
    prefix_regressor, errors = prefix_isotonic_regression(y, non_negativity=non_negativity)
    for i in range(1, shape + 1):
        prefix_yhat = _compute_isotonic_from_index(i, *prefix_regressor)
        sklearn_yhat = IsotonicRegression(y_min=[None, 0][non_negativity]).fit_transform(x[:i], y[:i])

        np.testing.assert_allclose(prefix_yhat, sklearn_yhat)

        error = np.sum((prefix_yhat - y[:i]) ** 2)
        assert error == approx(errors[i])

        if std == 0:
            np.testing.assert_allclose(prefix_yhat, y[:i])
        if non_negativity:
            assert all(prefix_yhat >= 0)


@pytest.mark.parametrize("shape", [5, 10, 50, 100, 500])
@pytest.mark.parametrize("std", [1, 2, 3])
@pytest.mark.parametrize("non_negativity", [False, True])
def test_unimodal_regression_error(shape, std, non_negativity):
    np.random.seed(0)

    # Test increasing x
    x = np.arange(shape).astype(float)
    y = x + np.random.standard_normal(shape) * std
    yhat, error = _unimodal_regression(y, non_negativity=non_negativity)

    if std == 0:
        np.testing.assert_allclose(yhat, y)

    assert np.sum((yhat - y) ** 2) == approx(error)
    if non_negativity:
        assert all(yhat >= 0)

    # Test decreasing x
    x = np.arange(shape)[::-1].astype(float)
    y = x + np.random.standard_normal(shape) * std
    yhat, error = _unimodal_regression(y, non_negativity=non_negativity)

    if std == 0:
        np.testing.assert_allclose(yhat, y)

    assert np.sum((yhat - y) ** 2) == approx(error)
    if non_negativity:
        assert all(yhat >= 0)

    # Test unimodal x
    x = np.arange(shape).astype(float)
    y = np.zeros_like(x)
    y[: shape // 2] = x[: shape // 2] + np.random.standard_normal(shape // 2) * std
    y[shape // 2 :] = x[: ceil(shape / 2)][::-1] + np.random.standard_normal(ceil(shape / 2)) * std
    yhat, error = _unimodal_regression(y, non_negativity=non_negativity)

    if std == 0:
        np.testing.assert_allclose(yhat, y)
    if non_negativity:
        assert all(yhat >= 0)

    assert np.sum((yhat - y) ** 2) == approx(error)


@pytest.mark.parametrize("non_negativity", [True, False])
def test_unimodal_regression_with_ndim_arrays(non_negativity):
    y = np.arange(10)
    with patch("matcouply._unimodal_regression._unimodal_regression") as mock:
        unimodal_regression(y, non_negativity=non_negativity)
        mock.assert_called_once()
        mock.assert_called_with(y, non_negativity=non_negativity)

    Y = np.arange(15).reshape(5, 3)
    with patch("matcouply._unimodal_regression._unimodal_regression", return_value=(np.zeros(5),)) as mock:
        out = unimodal_regression(Y, non_negativity=non_negativity)
        assert out.shape == Y.shape
        mock.assert_called()
        for call, y in zip(mock.call_args_list, Y.T):
            args, kwargs = call
            assert_array_equal(args[0], y)
            assert kwargs["non_negativity"] == non_negativity

    T = np.arange(30).reshape(5, 3, 2)
    with patch("matcouply._unimodal_regression._unimodal_regression", return_value=(np.zeros(5),)) as mock:
        out = unimodal_regression(T, non_negativity=non_negativity)
        assert out.shape == T.shape
        mock.assert_called()
        for call, y in zip(mock.call_args_list, T.reshape(5, -1).T):
            args, kwargs = call
            assert_array_equal(args[0], y)
            assert kwargs["non_negativity"] == non_negativity
