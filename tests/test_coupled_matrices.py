
import numpy as np
from pytest import approx
from cm_aoadmm import coupled_matrices, random
import pytest


@pytest.mark.parametrize("rank", [1, 2, 5])
def test_validate_cmf(rng, rank):
    # TODO: Make this test. 
    # TESTPLAN:
    # Create list of shapes
    # Create rank
    # Generate random valid CMF
    # Check that validate_cmf returns correct shapes and rank

    shapes = [(5, 10), (10, 10), (15, 10), (10, 10)]
    
    cmf = random.random_coupled_matrices(shapes, rank, random_state=rng)
    val_shapes, val_rank = coupled_matrices._validate_cmf(cmf)
    assert val_rank == rank
    assert shapes == val_shapes
    

    # Check different fail cases (with pytest.raises(<ExceptionType>))
    #   * One of the matrices (A, C or any of B_is) have wrong rank (e.g. rank+1)
    #   * Both A and C have wrong rank (e.g. rank+1)
    #   * One of the B_is have wrong number of columns (second element in shape)
    #   * One of the matrices is a third order tensor
    #   * One of the matrices is a vector
    #   * The weights is a matrix
    #   * The weights is a scalar
    #   * The wrong number of weights
    pass


def test_cmf_to_matrix(rng):
    # TODO: Make this test. 
    # TESTPLAN:
    # Generate random cmf
    # Construct single matrix manually
    # Check that matrix is computed correctly
    # Set the wrong number of columns for one of the matrices B_is[0] = rng.random_sample((tl.shape(B_is[0])[0], tl.shape(B_is[0])[1]+1))
    # Check that it fails when validate=True and not when validate=False
    # Check that we get same result as with cmf_to_slice
    pass


def test_cmf_to_matrices(rng):
    # TODO: Make this test. 
    # TESTPLAN:
    # Generate random cmf
    # Construct each matrix manually
    # Check that each matrix is computed correctly
    # Set the wrong number of columns for one of the matrices B_is[0] = rng.random_sample((tl.shape(B_is[0])[0], tl.shape(B_is[0])[1]+1))
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
