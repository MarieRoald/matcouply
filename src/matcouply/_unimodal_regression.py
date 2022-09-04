# MIT License: Copyright (c) 2022, Marie Roald.
# See the LICENSE file in the root directory for full license text.

import numpy as np

try:
    from numba import jit
except ImportError:  # pragma: no cover

    def jit(*args, **kwargs):
        return lambda x: x


@jit(nopython=True, cache=True, parallel=True)
def _merge_intervals_inplace(merge_target, merger, sum_weighted_y, sum_weighted_y_sq, sum_weights, level_set):
    sum_weighted_y[merge_target] += sum_weighted_y[merger]
    sum_weighted_y_sq[merge_target] += sum_weighted_y_sq[merger]
    sum_weights[merge_target] += sum_weights[merger]

    # Update the level set
    level_set[merge_target] = sum_weighted_y[merge_target] / sum_weights[merge_target]


@jit(
    nopython=True, cache=True,
)
def prefix_isotonic_regression(y, weights=None, non_negativity=False):
    if weights is None:
        weights = np.ones_like(y)

    sumwy = weights * y
    sumwy2 = weights * y * y
    sumw = weights.copy()

    level_set = np.zeros_like(y)
    index_range = np.zeros_like(y, dtype=np.int32)
    error = np.zeros(y.shape[0] + 1)  # +1 since error[0] is error of empty set

    level_set[0] = y[0]
    index_range[0] = 0
    num_samples = y.shape[0]

    if non_negativity:
        cumsumwy2 = np.cumsum(sumwy2)
        threshold = np.zeros(level_set.shape)
        if level_set[0] < 0:
            threshold[0] = True
            error[1] = cumsumwy2[0]

    for i in range(1, num_samples):
        level_set[i] = y[i]
        index_range[i] = i
        while level_set[i] <= level_set[index_range[i] - 1] and index_range[i] != 0:
            _merge_intervals_inplace(i, index_range[i] - 1, sumwy, sumwy2, sumw, level_set)
            index_range[i] = index_range[index_range[i] - 1]

        levelerror = sumwy2[i] - (sumwy[i] ** 2 / sumw[i])
        if non_negativity and level_set[i] < 0:
            threshold[i] = True
            error[i + 1] = cumsumwy2[i]
        else:
            error[i + 1] = levelerror + error[index_range[i]]

    if non_negativity:
        for i in range(len(level_set)):
            if threshold[i]:
                level_set[i] = 0

    return (level_set, index_range), error


@jit(nopython=True, cache=True)
def _compute_isotonic_from_index(end_index, level_set, index_range):
    idx = end_index - 1
    y_iso = np.empty_like(level_set[: idx + 1]) * np.nan

    while idx >= 0:
        y_iso[index_range[idx] : idx + 1] = level_set[idx]
        idx = index_range[idx] - 1

    return y_iso


def _get_best_unimodality_index(error_left, error_right):
    best_error = error_right[-1]
    best_idx = 0
    for i in range(error_left.shape[0]):
        error = error_left[i] + error_right[len(error_left) - i - 1]
        if error < best_error:
            best_error = error
            best_idx = i
    return best_idx, best_error


def _unimodal_regression(y, non_negativity):
    iso_left, error_left = prefix_isotonic_regression(y, non_negativity=non_negativity)
    iso_right, error_right = prefix_isotonic_regression(y[::-1], non_negativity=non_negativity)

    num_samples = y.shape[0]
    best_idx, error = _get_best_unimodality_index(error_left, error_right)
    y_iso_left = _compute_isotonic_from_index(best_idx, iso_left[0], iso_left[1])
    y_iso_right = _compute_isotonic_from_index(num_samples - best_idx, iso_right[0], iso_right[1])

    return np.concatenate([y_iso_left, y_iso_right[::-1]]), error


def unimodal_regression(y, non_negativity=False):
    r"""Compute the unimodal vector, :math:`\mathbf{u}` that minimizes :math:`\|\mathbf{y} - \mathbf{u}\|`.

    The unimodal regression problem is a problem on the form

    .. math:: \min_{\mathbf{u}} \|\mathbf{y} - \mathbf{u}\|\\
              \text{s.t.} u_1 \leq u_2 \leq ... \leq u_{t-1} \leq u_t \geq u_{t+1} \geq ... \geq u_{n-1} \geq u_n,

    for some index :math:`1 \leq t \leq n`. That is, it projects the input vector :math:`\mathbf{y}` onto the
    set of unimodal vectors. The *unimodal regression via prefix isotonic regression*  algorithm :cite:p:`stout2008unimodal`
    is used to efficiently solve the unimodal regression problem.

    Parameters
    ----------
    y : ndarray
        Vector to project. If it is an N-dimensional array, then it projects the first-mode fibers
        (e.g. columns in the case of matrices).
    non_negativity : bool
        If True, then non-negativity is imposed

    Returns
    -------
    ndarray
        The unimodal vector or array of unimodal vectors
    """
    y = np.asarray(y)
    if y.ndim == 1:
        return _unimodal_regression(y, non_negativity=non_negativity)[0]
    else:
        y2 = np.ascontiguousarray(y)
        y2 = y2.reshape(y2.shape[0], -1)
        unfolded_output = np.stack(
            [_unimodal_regression(y2[:, r], non_negativity=non_negativity)[0] for r in range(y2.shape[1])], axis=1,
        )
        return unfolded_output.reshape(y.shape)
