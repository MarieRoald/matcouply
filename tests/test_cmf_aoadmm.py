import itertools
import pytest


def all_combinations(*args):
    """All combinations of the input iterables.

    Each argument must be an iterable.

    Examples:
    ---------
    >>> all_combinations([1, 2], ["ab", "cd"])
    [(1, 'ab'), (1, 'cd'), (2, 'ab'), (2, 'cd')]
    """
    return list(itertools.product(*args))


@pytest.mark.parametrize(
    "rank,init",
    all_combinations([1, 2, 5], ["random", "svd", "threshold_svd"])
)
def test_initialize_cmf(rng, rank, init):
    # initialize_cmf(matrices, rank, init, svd_fun, random_state=None, init_params=None) -> cmf
    # Set shapes equal to some list of matrix shapes
    # Set matrices equal to [rng.random_sample(shape) for shape in shapes]
    # Set rank=1, 2 and 5
    # 
    # Set init="random", "svd" and "threshold_svd"
    # Construct matrices from CMF, check that shape of each constructed matrix is equal to the input matrices
    pass


def test_initialize_aux():
    # initialize_aux(matrices, rank, reg, random_state) -> A_aux_list, B_aux_list, C_aux_list
    # Set reg=[[NonNegativity(), NonNegativity(), NonNegativity()], [NonNegativity()], []]
    # assert len(A_aux_list) == 3. osv.
    pass


def test_initialize_dual():
    # initialize_dual(matrices, rank, reg, random_state) -> A_dual_list, B_dual_list, C_dual_list
    # Set reg=[[NonNegativity(), NonNegativity(), NonNegativity()], [NonNegativity()], []]
    # assert len(A_dual_list) == 3. osv.
    pass


def test_cmf_reconstruction_error(rng):
    # Construct random cmf
    # Construct tensor from random cmf
    # Add noise to that tensor
    # Compute reconstruction error
    # Ravel noise tensor and compute norm
    # Reconstruction error should be equal to noise norm
    pass


def test_compute_feasibility_gaps(rng):
    # compute_feasibility_gaps(cmf, reg, A_aux_list, B_aux_list, C_aux_list) -> A_gaps, B_gaps, C_gaps

    # Create A_noise_list, B_noise_list and C_noise_list
    # Create A_aux_list = [A + A_noise for A_noise in A_noise_list] and equivalent for B_is (B_is may require nested list comprehension or an explicit for loop) and C
    # Use ADMMPenalty() as all the regularisers (used for subtract_from_auxes)
    # Check that the norm of each element in A_noise_list and C_noise_list is equal to the corresponding feasibility gaps in A_gaps and C_gaps
    # Create B_stacked_noise_list that contains [tl.stack(B_is_noise) for B_is_noise in B_noise_list]
    # Check that the norm of each element in B_stacked_noise_list is equal to the feasibility gaps in B_gaps
    pass


def test_admm_update_A(rng, feasibility_penalty_scale, constant_feasibility_penalty):
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