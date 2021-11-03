# Test ideas:
# Stationary points: 
#   * TV: Constant
#   * Graph Laplacian: Piecewise constant on connected components
#   * L1: Zero
#   * Unimodal: Unimodal components
#   * PARAFAC2: A PARAFAC2 component (May not be fulfilled since prox is only approximate)
#   * Non-negative: Non-negative components
# Test that nonstationary points move
import pytest
from cm_aoadmm import penalties
import numpy as np
import scipy.stats as stats
from pytest import approx, fixture
from tensorly.testing import assert_array_almost_equal


def test_row_vector_penalty_forwards_updates_correctly(rng):
    # TODO: Make this test
    # To test if a RowVectorPenalty can be implemented by only overloading one method
    # Create a new RowVectorPenalty subclass for box-constraints with values between 0 and 1 where only the factor_matrix_row_update method is implemented
    # Test that factor_matrix_update and factor_matrices_update clip values correctly
    # REMEMBER: Generate input for update-functions by calling standard_normal
    
    
    pass


def test_matrix_penalty_forwards_updates_correctly(rng):
    # TODO: Make this test
    # To test if a MatrixPenalty can be implemented by only overloading one method
    # Create a new MatrixPenalty subclass for box-constraints with values between 0 and 1 where only the factor_matrix_update method is implemented
    # Test that factor_matrices_update clip values correctly
    # REMEMBER: Generate input for update-functions by calling standard_normal
    pass


## Interfaces only, not code to be run or inherited from:
class BaseTestADMMPenalty:
    @fixture
    def random_row(self, rng):
        return rng.standard_normal(3)
    
    @fixture
    def random_matrix(self, rng):
        return rng.standard_normal((10, 3))
    
    @fixture
    def random_matrices(self, rng):
        return [rng.standard_normal((10, 3)) for i in range(5)]

    def test_init_aux(self, rng):
        # TODO: Make this test
        # Generate random cmf
        # Generate matrices from random cmf
        # Initialize with uniform mode 0. Check shape of aux matrix is correct, check all >= 0
        # Initialize with uniform mode 1. Check shape of each aux matrix is correct, check all >= 0
        # Initialize with uniform mode 2. Check shape of aux matrix is correct, check all >= 0
        
        # Initialize with standard normal mode 0. Check shape of aux matrix is correct, check any < 0
        # Initialize with standard normal mode 1. Check shape of each aux matrix is correct, check any < 0
        # Initialize with standard normal mode 2. Check shape of aux matrix is correct, check any < 0

        # Check that we get type error if mode is not int
        # Check that we get value error when we use mode != 0, 1 or 2
        # Check that we get type error if aux_init is not str-type or tensor (for mode 0 and 2) or list for mode 1
        # Check that we get value error if aux_init is tensor of wrong size (mode 0 or 2) and if any of the tensors have wrong size (mode 1)
        # Check that we get value error if aux init is str but not "random_uniform" or "random_standard_normal"
        pass

    def test_init_dual(self, rng):
        # TODO: Make this test
        # Generate random cmf
        # Generate matrices from random cmf
        # Initialize with uniform mode 0. Check shape of dual matrix is correct, check all >= 0
        # Initialize with uniform mode 1. Check shape of each dual matrix is correct, check all >= 0
        # Initialize with uniform mode 2. Check shape of dual matrix is correct, check all >= 0
        
        # Initialize with standard normal mode 0. Check shape of dual matrix is correct, check any < 0
        # Initialize with standard normal mode 1. Check shape of each dual matrix is correct, check any < 0
        # Initialize with standard normal mode 2. Check shape of dual matrix is correct, check any < 0

        # Check that we get type error if mode is not int
        # Check that we get value error when we use mode != 0, 1 or 2
        # Check that we get type error if dual_init is not str-type or tensor (for mode 0 and 2) or list for mode 1
        # Check that we get value error if dual_init is tensor of wrong size (mode 0 or 2) and if any of the tensors have wrong size (mode 1)
        # Check that we get value error if dual_init is str but not "random_uniform" or "random_standard_normal"
        pass

    def test_penalty(self, rng):
        # TODO: Make this test
        # TODO: Implement for all subclasses, where we check penalty with some known values
        raise NotImplementedError

    def test_subtract_from_aux(self, rng):
        # TODO: Make this test
        # Use two matrices of equal shape
        # Check that subtract_from_aux(aux, dual) computes aux - dual
        pass

    def test_subtract_from_auxes(self, rng):
        # TODO: Make this test
        # Use two lists of matrices of equal shape
        # Check that subtract_from_auxes(auxes, duals) computes auxes - duals
        pass

    def test_aux_as_matrix(self, rng):
        # TODO: Make this test
        # Check that this is an identity operator. (Use random matrix and check that it is returned exactly the same)
        pass

    def test_auxes_as_matrices(self, rng):
        # TODO: Make this test
        # Check that this is an identity operator. (Use list of random matrices and check that it is returned exactly the same)
        pass


class BaseTestRowVectorPenalty(BaseTestADMMPenalty):  # e.g. non-negativity
    def test_row_update_stationary_point(self):
        raise NotImplementedError

    def test_factor_matrix_update_stationary_point(self):
        raise NotImplementedError
    
    def test_factor_matrices_update_stationary_point(self):
        raise NotImplementedError

    def test_row_update_reduces_penalty(self, random_row):
        raise NotImplementedError

    def test_factor_matrix_update_reduces_penalty(self, random_matrix):
        raise NotImplementedError
    
    def test_factor_matrices_update_reduces_penalty(self, random_matrices):
        raise NotImplementedError

    def test_row_update_changes_input(self, random_row):
        raise NotImplementedError

    def test_factor_matrix_update_changes_input(self, random_matrix):
        raise NotImplementedError

    def test_factor_matrices_update_changes_input(self, random_matrices):
        raise NotImplementedError


class BaseTestFactorMatrixPenalty(BaseTestADMMPenalty):  # e.g. unimodality
    def test_factor_matrix_update_stationary_point(self, rng):
        raise NotImplementedError
    
    def test_factor_matrices_update_stationary_point(self, rng):
        raise NotImplementedError

    def test_factor_matrix_update_reduces_penalty(self, random_matrix):
        raise NotImplementedError
    
    def test_factor_matrices_update_reduces_penalty(self, random_matrices):
        raise NotImplementedError

    def test_factor_matrix_update_changes_input(self, random_matrix):
        raise NotImplementedError

    def test_factor_matrices_update_changes_input(self, random_matrices):
        raise NotImplementedError


class BaseTestFactorMatricesPenalty(BaseTestADMMPenalty):  # e.g. PARAFAC2  
    def test_factor_matrices_update_stationary_point(self,):
        raise NotImplementedError
    
    def test_factor_matrices_update_reduces_penalty(self, random_matrices):
        raise NotImplementedError

    def test_factor_matrices_update_changes_input(self, random_matrices):
        raise NotImplementedError


# TODO: For all these, add parameter for non-negativity
class TestL1Penalty(BaseTestRowVectorPenalty):
    @pytest.mark.parametrize("non_negativity", [True, False])
    def test_row_update_stationary_point(self, non_negativity):
        stationary_matrix_row = np.zeros((1,4))
        l1_penalty = penalties.L1Penalty(0.1, non_negativity=non_negativity)

        out = l1_penalty.factor_matrix_row_update(stationary_matrix_row, 10, None)
        assert_array_almost_equal(stationary_matrix_row, out)

    def test_factor_matrix_update_stationary_point(self):
        stationary_matrix = np.zeros((10, 3))
        l1_penalty = penalties.L1Penalty(0.1)
        
        out = l1_penalty.factor_matrix_update(stationary_matrix, 10, None)        
        assert_array_almost_equal(stationary_matrix, out)

    def test_factor_matrices_update_stationary_point(self):        
        stationary_matrices = [np.zeros((10, 3)) for i in range(5)]
        feasibility_penalties = [10]*len(stationary_matrices)
        auxes = [None]*len(stationary_matrices)
        l1_penalty = penalties.L1Penalty(0.1)

        out = l1_penalty.factor_matrices_update(stationary_matrices, feasibility_penalties, auxes)
        for stationary_matrix, out_matrix in zip(stationary_matrices, out):
            assert_array_almost_equal(stationary_matrix, out_matrix)

    def test_row_update_reduces_penalty(self, random_row):
        l1_penalty = penalties.L1Penalty(0.1)

        initial_penalty = l1_penalty.penalty(random_row)
        out = l1_penalty.factor_matrix_row_update(random_row, 10, None)
        assert l1_penalty.penalty(out) <= initial_penalty

    def test_factor_matrix_update_reduces_penalty(self, random_matrix):
        l1_penalty = penalties.L1Penalty(0.1)

        initial_penalty = l1_penalty.penalty(random_matrix)
        out = l1_penalty.factor_matrix_update(random_matrix, 10, None)
        assert l1_penalty.penalty(out) <= initial_penalty

    def test_factor_matrices_update_reduces_penalty(self, random_matrices):
        l1_penalty = penalties.L1Penalty(0.1)
        feasibility_penalties = [10]*len(random_matrices)
        auxes = [None]*len(random_matrices)
        initial_penalty = l1_penalty.penalty(random_matrices)
        out = l1_penalty.factor_matrices_update(random_matrices, feasibility_penalties, auxes)
        assert l1_penalty.penalty(out) <= initial_penalty

    def test_row_update_changes_input(self, random_row):
        l1_penalty = penalties.L1Penalty(0.1)

        out = l1_penalty.factor_matrix_row_update(random_row, 10, None)
        assert not np.allclose(out, random_row)

    def test_factor_matrix_update_changes_input(self, random_matrix):
        l1_penalty = penalties.L1Penalty(0.1)

        out = l1_penalty.factor_matrix_update(random_matrix, 10, None)
        assert not np.allclose(out, random_matrix)

    def test_factor_matrices_update_changes_input(self, random_matrices):
        feasibility_penalties = [10]*len(random_matrices)
        auxes = [None]*len(random_matrices)
        l1_penalty = penalties.L1Penalty(0.1)

        out = l1_penalty.factor_matrices_update(random_matrices, feasibility_penalties, auxes)
        for random_matrix, out_matrix in zip(random_matrices, out):
            assert not np.allclose(random_matrix, out_matrix)

    def test_factor_matrix_update_sets_small_weights_to_zero(self, random_matrix):
        random_matrix /= np.abs(random_matrix).max()
        feasibility_penalty = 1
        aux = None
        l1_penalty = penalties.L1Penalty(1)

        out = l1_penalty.factor_matrix_update(random_matrix, feasibility_penalty, aux)
        assert_array_almost_equal(out, 0)


class TestBoxConstraint(BaseTestRowVectorPenalty):
    def test_row_update_stationary_point(self, rng):
        stationary_matrix_row = rng.uniform(size=(1, 3), low=-1, high=0) 
        box_penalty = penalties.BoxConstraint(min_val=-1, max_val=0)
        
        out = box_penalty.factor_matrix_row_update(stationary_matrix_row, 10, None)
        assert_array_almost_equal(stationary_matrix_row, out)

    
    def test_factor_matrix_update_stationary_point(self, rng):
        stationary_matrix = rng.uniform(size=(10, 3), low=-1, high=0)   
        box_penalty = penalties.BoxConstraint(min_val=-1, max_val=0)
        
        out = box_penalty.factor_matrix_update(stationary_matrix, 10, None)
        assert_array_almost_equal(stationary_matrix, out)
    
    def test_factor_matrices_update_stationary_point(self, rng):        
        stationary_matrices = [rng.uniform(size=(10, 3), low=-1, high=0)  for i in range(5)]
        feasibility_penalties = [10]*len(stationary_matrices)
        auxes = [None]*len(stationary_matrices)  
        box_penalty = penalties.BoxConstraint(min_val=-1, max_val=0)

        out = box_penalty.factor_matrices_update(stationary_matrices, feasibility_penalties, auxes)
        for stationary_matrix, out_matrix in zip(stationary_matrices, out):
            assert_array_almost_equal(stationary_matrix, out_matrix)

    def test_row_update_reduces_penalty(self, random_row):
        box_penalty = penalties.BoxConstraint(min_val=-1, max_val=0)

        initial_penalty = box_penalty.penalty(random_row)
        out = box_penalty.factor_matrix_row_update(random_row, 10, None)
        assert box_penalty.penalty(out) <= initial_penalty
    
    def test_factor_matrix_update_reduces_penalty(self, random_matrix):
        box_penalty = penalties.BoxConstraint(min_val=-1, max_val=0)

        initial_penalty = box_penalty.penalty(random_matrix)
        out = box_penalty.factor_matrix_update(random_matrix, 10, None)
        assert box_penalty.penalty(out) <= initial_penalty

    
    def test_factor_matrices_update_reduces_penalty(self, random_matrices):
        box_penalty = penalties.BoxConstraint(min_val=-1, max_val=0)
        feasibility_penalties = [10]*len(random_matrices)
        auxes = [None]*len(random_matrices)
        initial_penalty = box_penalty.penalty(random_matrices)
        out = box_penalty.factor_matrices_update(random_matrices, feasibility_penalties, auxes)
        assert box_penalty.penalty(out) <= initial_penalty
    
    
    def test_row_update_changes_input(self, random_row):
        random_row[0] += 100
        box_penalty = penalties.BoxConstraint(min_val=-1, max_val=0)

        out = box_penalty.factor_matrix_row_update(random_row, 10, None)
        assert not np.allclose(out, random_row)
    
    
    def test_factor_matrix_update_changes_input(self, random_matrix):
        random_matrix[:, 0] += 100
        box_penalty = penalties.BoxConstraint(min_val=-1, max_val=0)

        out = box_penalty.factor_matrix_update(random_matrix, 10, None)
        assert not np.allclose(out, random_matrix)
    
    def test_factor_matrices_update_changes_input(self, random_matrices):
        for random_matrix in random_matrices:
            random_matrix[:, 0] += 100

        feasibility_penalties = [10]*len(random_matrices)
        auxes = [None]*len(random_matrices)
        box_penalty = penalties.BoxConstraint(min_val=-1, max_val=0)

        out = box_penalty.factor_matrices_update(random_matrices, feasibility_penalties, auxes)
        for random_matrix, out_matrix in zip(random_matrices, out):
            assert not np.allclose(random_matrix, out_matrix)


class TestNonNegativity(BaseTestRowVectorPenalty):
    def test_row_update_stationary_point(self, rng):
        stationary_matrix_row = rng.uniform(size=(1, 3)) 
        nn_penalty = penalties.NonNegativity()
        
        out = nn_penalty.factor_matrix_row_update(stationary_matrix_row, 10, None)
        assert_array_almost_equal(stationary_matrix_row, out)

    
    def test_factor_matrix_update_stationary_point(self, rng):
        stationary_matrix = rng.uniform(size=(10, 3))      
        nn_penalty = penalties.NonNegativity()
        
        out = nn_penalty.factor_matrix_update(stationary_matrix, 10, None)
        assert_array_almost_equal(stationary_matrix, out)
    
    def test_factor_matrices_update_stationary_point(self, rng):        
        stationary_matrices = [rng.uniform(size=(10, 3)) for i in range(5)]
        feasibility_penalties = [10]*len(stationary_matrices)
        auxes = [None]*len(stationary_matrices)  
        nn_penalty = penalties.NonNegativity()

        out = nn_penalty.factor_matrices_update(stationary_matrices, feasibility_penalties, auxes)
        for stationary_matrix, out_matrix in zip(stationary_matrices, out):
            assert_array_almost_equal(stationary_matrix, out_matrix)

    
    def test_row_update_reduces_penalty(self, random_row):
        nn_penalty = penalties.NonNegativity()

        initial_penalty = nn_penalty.penalty(random_row)
        out = nn_penalty.factor_matrix_row_update(random_row, 10, None)
        assert nn_penalty.penalty(out) <= initial_penalty
    
    def test_factor_matrix_update_reduces_penalty(self, random_matrix):
        nn_penalty = penalties.NonNegativity()

        initial_penalty = nn_penalty.penalty(random_matrix)
        out = nn_penalty.factor_matrix_update(random_matrix, 10, None)
        assert nn_penalty.penalty(out) <= initial_penalty
    
    def test_factor_matrices_update_reduces_penalty(self, random_matrices):
        nn_penalty = penalties.NonNegativity()
        feasibility_penalties = [10]*len(random_matrices)
        auxes = [None]*len(random_matrices)
        initial_penalty = nn_penalty.penalty(random_matrices)
        out = nn_penalty.factor_matrices_update(random_matrices, feasibility_penalties, auxes)
        assert nn_penalty.penalty(out) <= initial_penalty

    def test_row_update_changes_input(self, random_row):
        nn_penalty = penalties.NonNegativity()

        out = nn_penalty.factor_matrix_row_update(random_row, 10, None)
        random_row[0] = -1
        assert not np.allclose(out, random_row)
    
    def test_factor_matrix_update_changes_input(self, random_matrix):
        nn_penalty = penalties.NonNegativity()
        random_matrix[:, 0] = -1

        out = nn_penalty.factor_matrix_update(random_matrix, 10, None)
        assert not np.allclose(out, random_matrix)
    
    def test_factor_matrices_update_changes_input(self, random_matrices):
        feasibility_penalties = [10]*len(random_matrices)
        auxes = [None]*len(random_matrices)
        nn_penalty = penalties.NonNegativity()
        for random_matrix in random_matrices:
            random_matrix[:, 0] = -1

        out = nn_penalty.factor_matrices_update(random_matrices, feasibility_penalties, auxes)
        for random_matrix, out_matrix in zip(random_matrices, out):
            assert not np.allclose(random_matrix, out_matrix)


class TestParafac2(BaseTestFactorMatricesPenalty):  
    def test_factor_matrices_update_stationary_point(self, rng):
        deltaB = rng.standard_normal((3, 3))
        Pks = [np.linalg.qr(rng.standard_normal(size=(10, 3)))[0] for _ in range(5)]
        auxes = Pks, deltaB
        stationary_matrices = [Pk @deltaB for Pk in Pks]

        feasibility_penalties = [10]*len(stationary_matrices)
        pf2_penalty = penalties.Parafac2()

        out = pf2_penalty.factor_matrices_update(stationary_matrices, feasibility_penalties, auxes)
        assert_array_almost_equal(deltaB, out[1])
        for Pk, out_matrix in zip(Pks, out[0]):
            assert_array_almost_equal(Pk, out_matrix)
    
    def test_factor_matrices_update_reduces_penalty(self, rng, random_matrices):
        deltaB = rng.standard_normal((3, 3))
        Pks = [np.linalg.qr(rng.standard_normal(size=(10, 3)))[0] for _ in range(5)]
        auxes = Pks, deltaB
        
        feasibility_penalties = [10]*len(random_matrices)
        pf2_penalty = penalties.Parafac2()

        initial_penalty = pf2_penalty.penalty(auxes)
        out = pf2_penalty.factor_matrices_update(random_matrices, feasibility_penalties, auxes)
        assert pf2_penalty.penalty(out) <= initial_penalty

    def test_factor_matrices_update_changes_input(self, random_matrices, rng):
        deltaB = rng.standard_normal((3, 3))
        Pks = [np.linalg.qr(rng.standard_normal(size=(10, 3)))[0] for _ in range(5)]
        auxes = Pks, deltaB
        
        feasibility_penalties = [10]*len(random_matrices)
        pf2_penalty = penalties.Parafac2()

        out = pf2_penalty.factor_matrices_update(random_matrices, feasibility_penalties, auxes)
        constructed_out = [Pk@out[1] for Pk in out[0]]
        for random_matrix, out_matrix in zip(random_matrices, constructed_out):
            assert not np.allclose(random_matrix, out_matrix)

    def test_init_aux(self, rng):
        # TODO: Make this test
        # Generate random cmf
        # Generate matrices from random cmf
        # Check that initialize fails with mode != 1

        # Initialize with both uniform and standard normal. For both:
        # Check that shape of the coordinate matrix is rank x rank
        # Check that the shape of the orthogonal basis-matrices is J_i x rank
        # For uniform, check all >= 0
        # For standard normal, check any < 0

        # Check that we get type error if mode is not int
        # Check that we get type error if aux_init is not str-type or list
        # Check that we get value error if aux_init is list and any of the factor matrices have wrong size
        # Check that we get value error if aux init is str but not "random_uniform" or "random_standard_normal"
        pass

    def test_subtract_from_aux(self, rng):
        # TODO: Make this test
        # Check that subtract_from_aux(None, None) raises TypeError
        pass

    def test_subtract_from_auxes(self, rng):
        # TODO: Make this test
        # Get aux variables and dual variables by using their init functions with random_uniform init
        # Compute aux_matrices by calling auxes_as_matrices
        # Check that subtract_from_auxes(auxes, duals) computes aux_matrices - duals
        pass

    def test_aux_as_matrix(self, rng):
        # TODO: Make this test
        # Check that this raises TypeError
        pass

    def test_auxes_as_matrices(self, rng):
        # TODO: Make this test
        # Construct auxes manually with a list comprehension 
        # Test that auxes_as_matrices gives the correct output.
        pass
