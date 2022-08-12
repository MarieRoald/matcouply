# MIT License: Copyright (c) 2022, Marie Roald.
# See the LICENSE file in the root directory for full license text.

import tensorly as tl

from .coupled_matrices import CoupledMatrixFactorization


def random_coupled_matrices(
    shapes, rank, full=False, random_state=None, normalise_factors=True, normalise_B=False, **context
):
    """Generate a random coupled matrix decomposition (with non-negative entries)

    Parameters
    ----------
    shapes : tuple
        A tuple where each element represents the shape of a matrix
        represented by the coupled matrix decomposition model. The
        second element in each shape-tuple must be constant.
    rank : int or int list
        rank of the coupled matrix decomposition
    full : bool, optional, default is False
        if True, a list of dense matrices is returned otherwise,
        the decomposition is returned
    random_state : `np.random.RandomState`

    Examples
    --------
    Here is an example of how to generate a random coupled matrix factorization

    >>> from matcouply.random import random_coupled_matrices
    >>> shapes = ((5, 10), (6, 10), (7, 10))
    >>> cmf = random_coupled_matrices(shapes, rank=4)
    >>> print(cmf)
    (weights, factors) : rank-4 CoupledMatrixFactorization of shape ((5, 10), (6, 10), (7, 10))
    """
    rns = tl.check_random_state(random_state)
    if not all(shape[1] == shapes[0][1] for shape in shapes):
        raise ValueError("All matrices must have equal number of columns.")

    A = tl.tensor(rns.random_sample((len(shapes), rank), **context))
    B_is = [tl.tensor(rns.random_sample((j_i, rank), **context)) for j_i, k in shapes]
    K = shapes[0][1]
    C = tl.tensor(rns.random_sample((K, rank), **context))

    weights = tl.ones(rank, **context)

    if normalise_factors or normalise_B:
        B_i_norms = [tl.norm(B_i, axis=0) for B_i in B_is]
        B_is = [B_i / B_i_norm for B_i, B_i_norm in zip(B_is, B_i_norms)]
        B_i_norms = tl.stack(B_i_norms)

        A = A * B_i_norms
    if normalise_factors:
        A_norm = tl.norm(A, axis=0)
        A = A / A_norm

        C_norm = tl.norm(C, axis=0)
        C = C / C_norm
        weights = A_norm * C_norm

    cmf = CoupledMatrixFactorization((weights, (A, B_is, C)))
    if full:
        return cmf.to_matrices()
    else:
        return cmf
