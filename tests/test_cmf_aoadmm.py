import itertools

import numpy as np
import pytest
import tensorly as tl
from numpy.core.fromnumeric import shape

from cm_aoadmm import cmf_aoadmm, coupled_matrices, random
from cm_aoadmm._utils import get_svd
from cm_aoadmm.penalties import NonNegativity


def all_combinations(*args):
    """All combinations of the input iterables.

    Each argument must be an iterable.

    Examples:
    ---------
    >>> all_combinations([1, 2], ["ab", "cd"])
    [(1, 'ab'), (1, 'cd'), (2, 'ab'), (2, 'cd')]
    """
    return list(itertools.product(*args))


@pytest.mark.parametrize("rank,init", all_combinations([1, 2, 5], ["random", "svd", "threshold_svd"]))
def test_initialize_cmf(rng, rank, init):
    shapes = ((5, 10), (10, 10), (15, 10))
    matrices = [rng.random_sample(shape) for shape in shapes]

    svd_fun = get_svd("truncated_svd")
    cmf = cmf_aoadmm.initialize_cmf(matrices, rank, init, svd_fun, random_state=None, init_params=None)
    init_matrices = coupled_matrices.cmf_to_matrices(cmf)
    for matrix, init_matrix in zip(matrices, init_matrices):
        assert matrix.shape == init_matrix.shape


@pytest.mark.parametrize(
    "rank", [1, 2, 5],
)
def test_initialize_aux(rng, rank):
    shapes = ((5, 10), (10, 10), (15, 10))  # TODO CHECK
    matrices = [rng.random(shape) for shape in shapes]

    reg = [[NonNegativity(), NonNegativity(), NonNegativity()], [NonNegativity()], []]
    A_aux_list, B_aux_list, C_aux_list = cmf_aoadmm.initialize_aux(matrices, rank, reg, rng)
    assert len(A_aux_list) == 3
    assert len(B_aux_list) == 1
    assert len(C_aux_list) == 0

    for A_aux in A_aux_list:
        assert tl.shape(A_aux) == (len(shapes), rank)
    for B_is_aux in B_aux_list:
        for B_i_aux, shape in zip(B_is_aux, shapes):
            assert tl.shape(B_i_aux) == (shape[0], rank)
        assert len(B_is_aux) == len(shapes)
    for C_aux in C_aux_list:
        assert tl.shape(C_aux) == (shapes[0][1], rank)


@pytest.mark.parametrize(
    "rank", [1, 2, 5],
)
def test_initialize_dual(rng, rank):
    shapes = ((5, 10), (10, 10), (15, 10))  # TODO CHECK
    matrices = [rng.random(shape) for shape in shapes]

    reg = [[NonNegativity(), NonNegativity(), NonNegativity()], [NonNegativity()], []]
    A_dual_list, B_dual_list, C_dual_list = cmf_aoadmm.initialize_dual(matrices, rank, reg, rng)
    assert len(A_dual_list) == 3
    assert len(B_dual_list) == 1
    assert len(C_dual_list) == 0

    for A_dual in A_dual_list:
        assert tl.shape(A_dual) == (len(shapes), rank)
    for B_is_dual in B_dual_list:
        for B_i_dual, shape in zip(B_is_dual, shapes):
            assert tl.shape(B_i_dual) == (shape[0], rank)
        assert len(B_is_dual) == len(shapes)
    for C_dual in C_dual_list:
        assert tl.shape(C_dual) == (shapes[0][1], rank)


def test_cmf_reconstruction_error(rng):
    shapes = ((11, 10), (11, 10), (11, 10), (11, 10))
    rank = 3
    cmf = random.random_coupled_matrices(shapes, rank, random_state=rng)
    matrices = cmf.to_matrices()
    noise = [rng.standard_normal(size=shape) for shape in shapes]
    noisy_matrices = [matrix + n for matrix, n in zip(matrices, noise)]

    error = cmf_aoadmm._cmf_reconstruction_error(noisy_matrices, cmf)
    assert error == pytest.approx(np.linalg.norm(noise))


def test_compute_feasibility_gaps(rng):
    # TODO: Make this test.
    # TESTPLAN:
    # compute_feasibility_gaps(cmf, reg, A_aux_list, B_aux_list, C_aux_list) -> A_gaps, B_gaps, C_gaps

    # Create A_noise_list, B_noise_list and C_noise_list
    # Create A_aux_list = [A + A_noise for A_noise in A_noise_list] and equivalent for B_is (B_is may require nested list comprehension or an explicit for loop) and C
    # Use ADMMPenalty() as all the regularisers (used for subtract_from_auxes)
    # Check that the norm of each element in A_noise_list and C_noise_list is equal to the corresponding feasibility gaps in A_gaps and C_gaps
    # Create B_stacked_noise_list that contains [tl.stack(B_is_noise) for B_is_noise in B_noise_list]
    # Check that the norm of each element in B_stacked_noise_list is equal to the feasibility gaps in B_gaps
    pass


def test_admm_update_A(rng, feasibility_penalty_scale, constant_feasibility_penalty):
    # TODO: Make this test.
    # TESTPLAN:
    # Construct NN-CMF
    # Create copy with same B & C but where A is some other NN matrix
    # Set matrices=constructed from original NN-CMF
    # Set cmf=modified NN-CMF
    # Set reg=[]
    # Set A_aux_list=[]
    # Set A_dual_list=[]
    # Set l2_penalty=None
    # Set inner_n_iter_max=1000
    # Set inner_tol=-1
    # Set feasibility_penalty_scale=[1, 10]
    # Set constant_feasibility_penalty=[True, False]
    # Set svd_fun=tl.GET_SVD["truncated_svd"]
    # Assert that output A is correct
    # Assert that output feasibility gap is low

    # Create a NonNegativity-instance
    # Use NonNegativity instance to init aux and duals
    # Set matrices=constructed from original NN-CMF
    # Set cmf=modified NN-CMF
    # Set reg=[NonNegativity()]
    # Set A_aux_list=[aux]
    # Set A_dual_list=[dual]
    # Set l2_penalty=None
    # Set inner_n_iter_max=1000
    # Set inner_tol=-1
    # Set feasibility_penalty_scale=[1, 10]
    # Set constant_feasibility_penalty=[True, False]
    # Set svd_fun=tl.GET_SVD["truncated_svd"]

    # TODO: Test for l2_penalty > 0
    ## For l2_penalty, compute linear system and solve using SVD to obtain regularised components. This will work with NN constraints too
    pass


def test_admm_update_B(rng, feasibility_penalty_scale, constant_feasibility_penalty):
    # TODO: Make this test.
    # TESTPLAN:
    # Construct NN-CMF
    # Create copy with same A & C but where B_is is some other list of NN matrices
    # Set matrices=constructed from original NN-CMF
    # Set cmf=modified NN-CMF
    # Set reg=[]
    # Set B_aux_list=[]
    # Set B_dual_list=[]
    # Set l2_penalty=None
    # Set inner_n_iter_max=1000
    # Set inner_tol=-1
    # Set feasibility_penalty_scale=[1, 10]
    # Set constant_feasibility_penalty=[True, False]
    # Set svd_fun=tl.GET_SVD["truncated_svd"]
    # Assert that output A is correct
    # Assert that output feasibility gap is low

    # Create a NonNegativity-instance
    # Use NonNegativity instance to init auxes and duals
    # Set matrices=constructed from original NN-CMF
    # Set cmf=modified NN-CMF
    # Set reg=[NonNegativity()]
    # Set B_aux_list=[aux]
    # Set B_dual_list=[aux]
    # Set l2_penalty=None
    # Set inner_n_iter_max=1000
    # Set inner_tol=-1
    # Set feasibility_penalty_scale=[1, 10]
    # Set constant_feasibility_penalty=[True, False]
    # Set svd_fun=tl.GET_SVD["truncated_svd"]

    # TODO: Test for l2_penalty > 0
    ## For l2_penalty, compute linear system and solve using SVD to obtain regularised components. This will work with NN constraints too
    pass


def test_admm_update_C(rng, feasibility_penalty_scale):
    # TODO: Make this test.
    # TESTPLAN:
    # Construct NN-CMF
    # Create copy with same A & B but where C is some other NN matrix
    # Set matrices=constructed from original NN-CMF
    # Set cmf=modified NN-CMF
    # Set reg=[]
    # Set C_aux_list=[]
    # Set C_dual_list=[]
    # Set l2_penalty=None
    # Set inner_n_iter_max=1000
    # Set inner_tol=-1
    # Set feasibility_penalty_scale=[1, 10]
    # Set svd_fun=tl.GET_SVD["truncated_svd"]
    # Assert that output A is correct
    # Assert that output feasibility gap is low

    # Create a NonNegativity-instance
    # Use NonNegativity instance to init aux and duals
    # Set matrices=constructed from original NN-CMF
    # Set cmf=modified NN-CMF
    # Set reg=[NonNegativity()]
    # Set C_aux_list=[aux]
    # Set C_dual_list=[dual]
    # Set l2_penalty=None
    # Set inner_n_iter_max=1000
    # Set inner_tol=-1
    # Set feasibility_penalty_scale=[1, 10]
    # Set svd_fun=tl.GET_SVD["truncated_svd"]

    # TODO: Test for l2_penalty > 0
    ## For l2_penalty, compute linear system and solve using SVD to obtain regularised components. This will work with NN constraints too
    pass


def test_cmf_aoadmm(rng):
    # TODO: Make this test.
    # TESTPLAN:
    # Create random NN CMF
    # Construct matrices
    # Decompose matrices with cmf_aoadmm with no constraints
    # Check that reconstruction error is low
    # Add non-negativity constraints on all modes
    # Check that reconstruction error is low
    # Change B_is to follow the PARAFAC2 constraint
    # Decompose with nonnegative PARAFAC2
    # Check that reconstruction error is low
    # Check that factors are good

    # Check that we get errors out when we ask for errors. Even if convergence checking is disabled and verbose=False
    # Check that final reconstruction error is the same as when we compute it with the returned decomposition and auxes
    # Check that feasibility gaps are the same as when we compute it with the returned decomposition and auxes
    pass
