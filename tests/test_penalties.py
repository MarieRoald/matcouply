# Test ideas:
# Stationary points: 
#   * TV: Constant
#   * Graph Laplacian: Piecewise constant on connected components
#   * L1: Zero
#   * Unimodal: Unimodal components
#   * PARAFAC2: A PARAFAC2 component (May not be fulfilled since prox is only approximate)
#   * Non-negative: Non-negative components
# Test that nonstationary points move
from cm_aoadmm import penalties
import numpy as np
import scipy.stats as stats
from pytest import approx, fixture

## Interfaces only, not code to be run or inherited from:
class BaseTestRowVectorPenalty:  # e.g. non-negativity
    @fixture
    def random_row(self, rng):
        return rng.standard_normal(3)
    
    @fixture
    def random_matrix(self, rng):
        return rng.standard_normal((10, 3))
    
    @fixture
    def random_matrices(self, rng):
        return [rng.standard_normal((10, 3)) for i in range(5)]

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


class BaseTestFactorMatrixPenalty:  # e.g. unimodality
    @fixture
    def random_matrix(self, rng):
        return rng.standard_normal((10, 3))

    @fixture
    def random_matrices(self, rng):
        return [rng.standard_normal((10, 3)) for i in range(5)]

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


class BaseTestFactorMatricesPenalty:  # e.g. PARAFAC2  
    @fixture 
    def random_matrices(self, rng):
        return [rng.standard_normal((10, 3)) for i in range(5)] 

    def test_factor_matrices_update_stationary_point(self,):
        raise NotImplementedError
    
    def test_factor_matrices_update_reduces_penalty(self, random_matrices):
        raise NotImplementedError

    def test_factor_matrices_update_changes_input(self, random_matrices):
        raise NotImplementedError


class TestL1Penalty(BaseTestRowVectorPenalty):
    def test_row_update_stationary_point(self):
        stationary_matrix_row = np.zeros((1,4))
        l1_penalty = penalties.L1Penalty(0.1)

        out = l1_penalty.factor_matrix_row_update(stationary_matrix_row, 10, None)
        np.testing.assert_allclose(stationary_matrix_row, out)

    def test_factor_matrix_update_stationary_point(self):
        stationary_matrix = np.zeros((10, 3))
        l1_penalty = penalties.L1Penalty(0.1)
        
        out = l1_penalty.factor_matrix_update(stationary_matrix, 10, None)        
        np.testing.assert_allclose(stationary_matrix, out)

    def test_factor_matrices_update_stationary_point(self):        
        stationary_matrices = [np.zeros((10, 3)) for i in range(5)]
        feasibility_penalties = [10]*len(stationary_matrices)
        auxes = [None]*len(stationary_matrices)
        l1_penalty = penalties.L1Penalty(0.1)

        out = l1_penalty.factor_matrices_update(stationary_matrices, feasibility_penalties, auxes)
        for stationary_matrix, out_matrix in zip(stationary_matrices, out):
            np.testing.assert_allclose(stationary_matrix, out_matrix)

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
        np.testing.assert_allclose(out, 0)


class TestBoxConstraint(BaseTestRowVectorPenalty):
    def test_row_update_stationary_point(self, rng):
        stationary_matrix_row = rng.uniform(size=(1, 3), low=-1, high=0) 
        box_penalty = penalties.BoxConstraint(min_val=-1, max_val=0)
        
        out = box_penalty.factor_matrix_row_update(stationary_matrix_row, 10, None)
        np.testing.assert_allclose(stationary_matrix_row, out)

    
    def test_factor_matrix_update_stationary_point(self, rng):
        stationary_matrix = rng.uniform(size=(10, 3), low=-1, high=0)   
        box_penalty = penalties.BoxConstraint(min_val=-1, max_val=0)
        
        out = box_penalty.factor_matrix_update(stationary_matrix, 10, None)
        np.testing.assert_allclose(stationary_matrix, out)
    
    def test_factor_matrices_update_stationary_point(self, rng):        
        stationary_matrices = [rng.uniform(size=(10, 3), low=-1, high=0)  for i in range(5)]
        feasibility_penalties = [10]*len(stationary_matrices)
        auxes = [None]*len(stationary_matrices)  
        box_penalty = penalties.BoxConstraint(min_val=-1, max_val=0)

        out = box_penalty.factor_matrices_update(stationary_matrices, feasibility_penalties, auxes)
        for stationary_matrix, out_matrix in zip(stationary_matrices, out):
            np.testing.assert_allclose(stationary_matrix, out_matrix)

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
        np.testing.assert_allclose(stationary_matrix_row, out)

    
    def test_factor_matrix_update_stationary_point(self, rng):
        stationary_matrix = rng.uniform(size=(10, 3))      
        nn_penalty = penalties.NonNegativity()
        
        out = nn_penalty.factor_matrix_update(stationary_matrix, 10, None)
        np.testing.assert_allclose(stationary_matrix, out)
    
    def test_factor_matrices_update_stationary_point(self, rng):        
        stationary_matrices = [rng.uniform(size=(10, 3)) for i in range(5)]
        feasibility_penalties = [10]*len(stationary_matrices)
        auxes = [None]*len(stationary_matrices)  
        nn_penalty = penalties.NonNegativity()

        out = nn_penalty.factor_matrices_update(stationary_matrices, feasibility_penalties, auxes)
        for stationary_matrix, out_matrix in zip(stationary_matrices, out):
            np.testing.assert_allclose(stationary_matrix, out_matrix)

    
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
        pf2_penalty = penalties.Parafac2(svd_fun=np.linalg.svd ) #FIXME: svd fun

        out = pf2_penalty.factor_matrices_update(stationary_matrices, feasibility_penalties, auxes)
        np.testing.assert_allclose(deltaB, out[1])
        for Pk, out_matrix in zip(Pks, out[0]):
            np.testing.assert_allclose(Pk, out_matrix)
    
    def test_factor_matrices_update_reduces_penalty(self, rng, random_matrices):
        deltaB = rng.standard_normal((3, 3))
        Pks = [np.linalg.qr(rng.standard_normal(size=(10, 3)))[0] for _ in range(5)]
        auxes = Pks, deltaB
        
        feasibility_penalties = [10]*len(random_matrices)
        pf2_penalty = penalties.Parafac2(svd_fun=np.linalg.svd) #FIXME: svd fun

        initial_penalty = pf2_penalty.penalty(auxes)
        out = pf2_penalty.factor_matrices_update(random_matrices, feasibility_penalties, auxes)
        assert pf2_penalty.penalty(out) <= initial_penalty

    def test_factor_matrices_update_changes_input(self, random_matrices, rng):
        deltaB = rng.standard_normal((3, 3))
        Pks = [np.linalg.qr(rng.standard_normal(size=(10, 3)))[0] for _ in range(5)]
        auxes = Pks, deltaB
        
        feasibility_penalties = [10]*len(random_matrices)
        pf2_penalty = penalties.Parafac2(svd_fun=np.linalg.svd) #FIXME: svd fun

        out = pf2_penalty.factor_matrices_update(random_matrices, feasibility_penalties, auxes)
        constructed_out = [Pk@out[1] for Pk in out[0]]
        for random_matrix, out_matrix in zip(random_matrices, constructed_out):
            assert not np.allclose(random_matrix, out_matrix)
