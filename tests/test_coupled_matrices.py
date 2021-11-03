
import numpy as np
from numpy.core.fromnumeric import shape
from pytest import approx
from cm_aoadmm import coupled_matrices, random
import pytest
from copy import copy


@pytest.mark.parametrize("rank", [1, 2, 5])
def test_validate_cmf(rng, rank):
    shapes = ((5, 10), (10, 10), (15, 10), (10, 10))
    cmf = random.random_coupled_matrices(shapes, rank, random_state=rng)
    val_shapes, val_rank = coupled_matrices._validate_cmf(cmf)
    assert val_rank == rank
    assert shapes == val_shapes

    weights, (A, B_is, C) = cmf

    #####
    # Check that non-tensor inputs result in TypeErrors
    # The weights is a scalar
    with pytest.raises(TypeError):
        coupled_matrices._validate_cmf((3, (A, B_is, C)))
    with pytest.raises(TypeError):
        coupled_matrices._validate_cmf((weights, (1, B_is, C)))
    with pytest.raises(TypeError):
        coupled_matrices._validate_cmf((weights, (None, B_is, C)))
    with pytest.raises(TypeError):
        coupled_matrices._validate_cmf((weights, (A, 1, C)))
    with pytest.raises(TypeError):
        coupled_matrices._validate_cmf((weights, (A, None, C)))
    with pytest.raises(TypeError):
        B_is_copy = copy(B_is)
        B_is_copy[1] = 1
        coupled_matrices._validate_cmf((weights, (A, B_is_copy, C)))
    with pytest.raises(TypeError):
        B_is_copy = copy(B_is)
        B_is_copy[1] = None
        coupled_matrices._validate_cmf((weights, (A, B_is_copy, C)))
    with pytest.raises(TypeError):
        coupled_matrices._validate_cmf((weights, (A, B_is, 1)))
    with pytest.raises(TypeError):
        coupled_matrices._validate_cmf((weights, (A, B_is, None)))
    
    #####
    # Check that None-valued weights do not raise any errors
    coupled_matrices._validate_cmf((None, (A, B_is, C)))
    
    #####
    # Check that wrongly shaped inputs result in ValueErrors

    ### Weights
    # The weights is a matrix
    with pytest.raises(ValueError):
        coupled_matrices._validate_cmf((np.ones(shape=(rank, rank)), (A, B_is, C)))
    # Wrong number of weights
    with pytest.raises(ValueError):
        coupled_matrices._validate_cmf((np.ones(shape=(rank+1, )), (A, B_is, C)))
    
    ### Factor matrices
    # One of the matrices is a third order tensor
    with pytest.raises(ValueError):
        coupled_matrices._validate_cmf((weights, (rng.random(size=(4, rank, rank)), B_is, C)))
    with pytest.raises(ValueError):
        coupled_matrices._validate_cmf((weights, (A, B_is, rng.random(size=(4, rank, rank)))))
    with pytest.raises(ValueError):
        B_is_copy = copy(B_is)
        B_is_copy[1] = rng.random(size=(4, rank, rank))
        coupled_matrices._validate_cmf((weights, (A, B_is_copy, C)))


    # One of the matrices is a vector
    with pytest.raises(ValueError):
        coupled_matrices._validate_cmf((weights, (rng.random(size=(rank,)), B_is, C)))
    with pytest.raises(ValueError):
        coupled_matrices._validate_cmf((weights, (A, B_is, rng.random(size=(rank,)))))
    with pytest.raises(ValueError):
        B_is_copy = copy(B_is)
        B_is_copy[1] = rng.random(size=(rank,))
        coupled_matrices._validate_cmf((weights, (A, B_is_copy, C)))

    ### Check wrong rank
    # Check with incorrect rank for one of the factors
    invalid_A = rng.random((len(shapes), rank+1))
    invalid_C = rng.random((shapes[0][1], rank+1))
    invalid_B_is_2 = [rng.random((j_i, rank)) for j_i, k in shapes]
    invalid_B_is_2[0] = rng.random((shapes[0][0], rank+1))

    # Both A and C have the wrong rank:
    with pytest.raises(ValueError): 
        coupled_matrices._validate_cmf((weights, (invalid_A, B_is, invalid_C)))

    # One of the matrices (A, C or any of B_is) have wrong rank
    with pytest.raises(ValueError):
        coupled_matrices._validate_cmf((weights, (invalid_A, B_is, C)))
    with pytest.raises(ValueError):
        coupled_matrices._validate_cmf((weights, (A, B_is, invalid_C)))
    with pytest.raises(ValueError):
        coupled_matrices._validate_cmf((weights, (A, invalid_B_is_2, C)))


def test_cmf_to_matrix(rng):
    # TODO: Make this test. 
    # TESTPLAN:
    # Generate random cmf
    # Construct single matrix manually # TODO: normalise?
    # Check that matrix is computed correctly
    # Set the wrong number of columns for one of the matrices B_is[0] = rng.random((tl.shape(B_is[0])[0], tl.shape(B_is[0])[1]+1))
    # Check that it fails when validate=True and not when validate=False
    # Check that we get same result as with cmf_to_slice
    shapes = ((5, 10), (10, 10), (15, 10), (10, 10))
    rank = 5
    cmf = random.random_coupled_matrices(shapes, rank, random_state=rng)
    weights, (A, B_is, C) = cmf



def test_cmf_to_matrices(rng):
    # TODO: Make this test. 
    # TESTPLAN:
    # Generate random cmf
    # Construct each matrix manually TODO: CHECK
    # Check that each matrix is computed correctly
    # Set the wrong number of columns for one of the matrices B_is[0] = rng.random((tl.shape(B_is[0])[0], tl.shape(B_is[0])[1]+1))
    # Check that it fails when validate=True and not when validate=False
    # Check that we get same result as with cmf_to_slices
    pass


def test_cmf_to_tensor(rng):
    # TODO: Make this test. 
    # TESTPLAN:
    # Generate random tensor represented by a cmf
    # Construct the tensor manually
    # Construct tensor and check that it is the same
    # Generate random cmf with different number of rows (J_is)
    # Construct matrices manually
    # Construct tensor and check that it is the same as the matrices, but padded with zeros
    #   -> Iterate over matrices and tensor slabs.
    #   -> Slice out relevant part of tensor slab and compare with matrix
    #   -> Slice out irrelevant part of tensor slab and check that it is equal to zero

    # TODO: How to check validate? Mocking maybe?
    pass


def test_cmf_to_unfolded(rng):
    # TODO: Make this test. 
    # TESTPLAN:
    # Generate random tensor represented by a cmf
    # Construct the tensor manually and unfold it in the correct mode
    # Construct unfolded tensor and check that it is the same
    
    # TODO: How to check validate? Mocking maybe?
    pass


def test_cmf_to_unfolded(rng):
    # TODO: Make this test. 
    # TESTPLAN:
    # Generate random tensor represented by a cmf
    # Construct the tensor manually and ravel it to get a vector
    # Construct vector and check that it is the same
    
    # TODO: How to check validate? Mocking maybe?
    pass
