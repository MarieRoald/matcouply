# Test ideas:
# Stationary points:
#   * TV: Constant
#   * Graph Laplacian: Piecewise constant on connected components
#   * L1: Zero
#   * Unimodal: Unimodal components
#   * PARAFAC2: A PARAFAC2 component (May not be fulfilled since prox is only approximate)
#   * Non-negative: Non-negative components
# Test that nonstationary points move
from copy import copy

import numpy as np
import pytest
import tensorly as tl
from pytest import fixture
from tensorly.testing import assert_array_almost_equal, assert_array_equal

from matcouply import penalties


@fixture
def random_row(rng):
    return rng.standard_normal(3)


@fixture
def random_matrix(rng):
    return rng.standard_normal((10, 3))


@fixture
def random_matrices(rng):
    return [rng.standard_normal((10, 3)) for i in range(5)]


def test_row_vector_penalty_forwards_updates_correctly(rng, random_matrix, random_matrices):
    class BetweenZeroAndOneConstraint(penalties.RowVectorPenalty):
        def factor_matrix_row_update(self, factor_matrix_row, feasibility_penalty, aux_row):
            return tl.clip(factor_matrix_row, 0, 1)

        def penalty(self, x):
            return 0

    penalty = BetweenZeroAndOneConstraint()
    updated_factor_matrix = penalty.factor_matrix_update(random_matrix, 1, random_matrix)  # last input is ignored

    assert tl.all(updated_factor_matrix >= 0)
    assert tl.all(updated_factor_matrix <= 1)

    updated_factor_matrices = penalty.factor_matrices_update(
        random_matrices, np.ones(len(random_matrices)), random_matrices
    )  # last input is ignored
    for factor_matrix in updated_factor_matrices:
        assert tl.all(factor_matrix >= 0)
        assert tl.all(factor_matrix <= 1)


def test_matrix_penalty_forwards_updates_correctly(rng, random_matrices):
    class BetweenZeroAndOneConstraint(penalties.MatrixPenalty):
        def factor_matrix_update(self, factor_matrix, feasibility_penalty, aux_row):
            return tl.clip(factor_matrix, 0, 1)

        def penalty(self, x):
            return 0

    penalty = BetweenZeroAndOneConstraint()
    updated_factor_matrices = penalty.factor_matrices_update(
        random_matrices, np.ones(len(random_matrices)), random_matrices
    )  # last input is ignored
    for factor_matrix in updated_factor_matrices:
        assert tl.all(factor_matrix >= 0)
        assert tl.all(factor_matrix <= 1)


# # Interfaces only, not code to be run or inherited from:
class BaseTestADMMPenalty:
    PenaltyType = penalties.ADMMPenalty
    penalty_default_kwargs = {}

    @pytest.mark.parametrize("dual_init", ["random_uniform", "random_standard_normal", "zeros"])
    def test_init_aux(self, rng, random_ragged_cmf, dual_init):
        cmf, shapes, rank = random_ragged_cmf
        matrices = cmf.to_matrices()

        # TODO: Separate this into multiple tests
        # Test that init works with random uniform init
        penalty = self.PenaltyType(aux_init="random_uniform", dual_init=dual_init, **self.penalty_default_kwargs)

        init_matrix = penalty.init_aux(matrices, rank, mode=0, random_state=rng)
        assert init_matrix.shape[0] == len(shapes)
        assert init_matrix.shape[1] == rank
        assert tl.all(init_matrix >= 0)
        assert tl.all(init_matrix < 1)

        init_matrices = penalty.init_aux(matrices, rank, mode=1, random_state=rng)
        for init_matrix, shape in zip(init_matrices, shapes):
            assert init_matrix.shape[0] == shape[0]
            assert init_matrix.shape[1] == rank
            assert tl.all(init_matrix >= 0)
            assert tl.all(init_matrix < 1)

        init_matrix = penalty.init_aux(matrices, rank, mode=2, random_state=rng)
        assert init_matrix.shape[0] == shapes[0][1]
        assert init_matrix.shape[1] == rank
        assert tl.all(init_matrix >= 0)
        assert tl.all(init_matrix < 1)

        # Test that init works with random standard normal init
        penalty = self.PenaltyType(
            aux_init="random_standard_normal", dual_init=dual_init, **self.penalty_default_kwargs
        )

        init_matrix = penalty.init_aux(matrices, rank, mode=0, random_state=rng)
        assert init_matrix.shape[0] == len(shapes)
        assert init_matrix.shape[1] == rank
        # TODO: Add normality test

        init_matrices = penalty.init_aux(matrices, rank, mode=1, random_state=rng)
        for init_matrix, shape in zip(init_matrices, shapes):
            assert init_matrix.shape[0] == shape[0]
            assert init_matrix.shape[1] == rank
            # TODO: Add normality test

        init_matrix = penalty.init_aux(matrices, rank, mode=2, random_state=rng)
        assert init_matrix.shape[0] == shapes[0][1]
        assert init_matrix.shape[1] == rank
        # TODO: Add normality test

        # Test that init works with zeros init
        penalty = self.PenaltyType(aux_init="zeros", dual_init=dual_init, **self.penalty_default_kwargs)

        init_matrix = penalty.init_aux(matrices, rank, mode=0, random_state=rng)
        assert init_matrix.shape[0] == len(shapes)
        assert init_matrix.shape[1] == rank
        assert_array_equal(init_matrix, 0)

        init_matrices = penalty.init_aux(matrices, rank, mode=1, random_state=rng)
        for init_matrix, shape in zip(init_matrices, shapes):
            assert init_matrix.shape[0] == shape[0]
            assert init_matrix.shape[1] == rank
            assert_array_equal(init_matrix, 0)

        init_matrix = penalty.init_aux(matrices, rank, mode=2, random_state=rng)
        assert init_matrix.shape[0] == shapes[0][1]
        assert init_matrix.shape[1] == rank
        assert_array_equal(init_matrix, 0)

        # Test that mode and rank needs int input
        with pytest.raises(TypeError):
            penalty.init_aux(matrices, rank, mode=None)
        with pytest.raises(TypeError):
            penalty.init_aux(matrices, rank=None, mode=0)

        # Test that mode needs to be between 0 and 2
        with pytest.raises(ValueError):
            penalty.init_aux(matrices, rank, mode=-1)
        with pytest.raises(ValueError):
            penalty.init_aux(matrices, rank, mode=3)

        # Test that the init method must be a valid type
        invalid_inits = [None, 1, 1.1]
        for invalid_init in invalid_inits:
            penalty = self.PenaltyType(aux_init=invalid_init, dual_init=dual_init, **self.penalty_default_kwargs)
            for mode in range(2):
                with pytest.raises(TypeError):
                    penalty.init_aux(matrices, rank, mode=mode, random_state=rng)

        # Check that aux_init can be tensor (for mode 0 and 2) or list for mode 1
        weights, (A, B_is, C) = cmf
        penalty = self.PenaltyType(aux_init=A, dual_init=dual_init, **self.penalty_default_kwargs)
        assert_array_equal(A, penalty.init_aux(matrices, rank, 0, random_state=rng))

        penalty = self.PenaltyType(aux_init=C, dual_init=dual_init, **self.penalty_default_kwargs)
        assert_array_equal(C, penalty.init_aux(matrices, rank, 2, random_state=rng))

        penalty = self.PenaltyType(aux_init=B_is, dual_init=dual_init, **self.penalty_default_kwargs)
        dual_B_is = penalty.init_aux(matrices, rank, 1, random_state=rng)
        for B_i, dual_B_i in zip(B_is, dual_B_is):
            assert_array_equal(B_i, dual_B_i)

        # Check that we get value error if aux_init is tensor of wrong size (mode 0 or 2)
        # and if any of the tensors have wrong size (mode 1) or the list has the wrong length (mode 1)
        I = tl.shape(A)[0]
        J_is = [tl.shape(B_i)[0] for B_i in B_is]
        K = tl.shape(C)[0]

        invalid_A = rng.random((I + 1, rank))
        invalid_C = rng.random((K + 1, rank))
        invalid_B_is = [rng.random((J_i, rank)) for J_i in J_is]
        invalid_B_is[0] = rng.random((J_is[0] + 1, rank))

        penalty = self.PenaltyType(aux_init=invalid_A, dual_init=dual_init, **self.penalty_default_kwargs)
        with pytest.raises(ValueError):
            penalty.init_aux(matrices, rank, 0, random_state=rng)
        penalty = self.PenaltyType(aux_init=invalid_C, dual_init=dual_init, **self.penalty_default_kwargs)
        with pytest.raises(ValueError):
            penalty.init_aux(matrices, rank, 2, random_state=rng)
        penalty = self.PenaltyType(aux_init=invalid_B_is, dual_init=dual_init, **self.penalty_default_kwargs)
        with pytest.raises(ValueError):
            penalty.init_aux(matrices, rank, 1, random_state=rng)
        penalty = self.PenaltyType(aux_init=B_is + B_is, dual_init=dual_init, **self.penalty_default_kwargs)
        with pytest.raises(ValueError):
            penalty.init_aux(matrices, rank, 1, random_state=rng)

        # Check that mode 0 and 2 cannot accept list of matrices
        penalty = self.PenaltyType(aux_init=B_is, dual_init=dual_init, **self.penalty_default_kwargs)
        with pytest.raises(TypeError):
            penalty.init_aux(matrices, rank, 0, random_state=rng)
        with pytest.raises(TypeError):
            penalty.init_aux(matrices, rank, 2, random_state=rng)

        # Check that mode 1 cannot accept single matrix
        penalty = self.PenaltyType(aux_init=A, dual_init=dual_init, **self.penalty_default_kwargs)
        with pytest.raises(TypeError):
            penalty.init_aux(matrices, rank, 1, random_state=rng)

        # Check that we get value error if aux init is str but not "random_uniform" or "random_standard_normal"
        penalty = self.PenaltyType(aux_init="invalid init name", dual_init=dual_init, **self.penalty_default_kwargs)
        for mode in range(2):
            with pytest.raises(ValueError):
                penalty.init_aux(matrices, rank, mode=mode, random_state=None)

    @pytest.mark.parametrize("aux_init", ["random_uniform", "random_standard_normal", "zeros"])
    def test_init_dual(self, rng, random_ragged_cmf, aux_init):
        cmf, shapes, rank = random_ragged_cmf
        matrices = cmf.to_matrices()

        # Test that init works with random uniform init
        penalty = self.PenaltyType(aux_init=aux_init, dual_init="random_uniform", **self.penalty_default_kwargs)

        init_matrix = penalty.init_dual(matrices, rank, mode=0, random_state=rng)
        assert init_matrix.shape[0] == len(shapes)
        assert init_matrix.shape[1] == rank
        assert tl.all(init_matrix >= 0)
        assert tl.all(init_matrix < 1)

        init_matrices = penalty.init_dual(matrices, rank, mode=1, random_state=rng)
        for init_matrix, shape in zip(init_matrices, shapes):
            assert init_matrix.shape[0] == shape[0]
            assert init_matrix.shape[1] == rank
            assert tl.all(init_matrix >= 0)
            assert tl.all(init_matrix < 1)

        init_matrix = penalty.init_dual(matrices, rank, mode=2, random_state=rng)
        assert init_matrix.shape[0] == shapes[0][1]
        assert init_matrix.shape[1] == rank
        assert tl.all(init_matrix >= 0)
        assert tl.all(init_matrix < 1)

        # Test that init works with random standard normal init
        penalty = self.PenaltyType(aux_init=aux_init, dual_init="random_standard_normal", **self.penalty_default_kwargs)

        init_matrix = penalty.init_dual(matrices, rank, mode=0, random_state=rng)
        assert init_matrix.shape[0] == len(shapes)
        assert init_matrix.shape[1] == rank
        # TODO: Add normality test

        init_matrices = penalty.init_dual(matrices, rank, mode=1, random_state=rng)
        for init_matrix, shape in zip(init_matrices, shapes):
            assert init_matrix.shape[0] == shape[0]
            assert init_matrix.shape[1] == rank
            # TODO: Add normality test

        init_matrix = penalty.init_dual(matrices, rank, mode=2, random_state=rng)
        assert init_matrix.shape[0] == shapes[0][1]
        assert init_matrix.shape[1] == rank
        # TODO: Add normality test

        # Test that init works with zeros init
        penalty = self.PenaltyType(aux_init=aux_init, dual_init="zeros", **self.penalty_default_kwargs)

        init_matrix = penalty.init_dual(matrices, rank, mode=0, random_state=rng)
        assert init_matrix.shape[0] == len(shapes)
        assert init_matrix.shape[1] == rank
        assert_array_equal(init_matrix, 0)

        init_matrices = penalty.init_dual(matrices, rank, mode=1, random_state=rng)
        for init_matrix, shape in zip(init_matrices, shapes):
            assert init_matrix.shape[0] == shape[0]
            assert init_matrix.shape[1] == rank
            assert_array_equal(init_matrix, 0)

        init_matrix = penalty.init_dual(matrices, rank, mode=2, random_state=rng)
        assert init_matrix.shape[0] == shapes[0][1]
        assert init_matrix.shape[1] == rank
        assert_array_equal(init_matrix, 0)

        # Test that mode and rank needs int input
        with pytest.raises(TypeError):
            penalty.init_dual(matrices, rank, mode=None)
        with pytest.raises(TypeError):
            penalty.init_dual(matrices, rank=None, mode=0)

        # Test that mode needs to be between 0 and 2
        with pytest.raises(ValueError):
            penalty.init_dual(matrices, rank, mode=-1)
        with pytest.raises(ValueError):
            penalty.init_dual(matrices, rank, mode=3)

        # Test that the init method must be a valid type
        invalid_inits = [None, 1, 1.1]
        for invalid_init in invalid_inits:
            penalty = self.PenaltyType(aux_init=aux_init, dual_init=invalid_init, **self.penalty_default_kwargs)
            for mode in range(2):
                with pytest.raises(TypeError):
                    penalty.init_dual(matrices, rank, mode=mode, random_state=rng)

        # Check that aux_init can be tensor (for mode 0 and 2) or list for mode 1
        weights, (A, B_is, C) = cmf
        penalty = self.PenaltyType(aux_init=aux_init, dual_init=A, **self.penalty_default_kwargs)
        assert_array_equal(A, penalty.init_dual(matrices, rank, 0, random_state=rng))

        penalty = self.PenaltyType(aux_init=aux_init, dual_init=C, **self.penalty_default_kwargs)
        assert_array_equal(C, penalty.init_dual(matrices, rank, 2, random_state=rng))

        penalty = self.PenaltyType(aux_init=aux_init, dual_init=B_is, **self.penalty_default_kwargs)
        dual_B_is = penalty.init_dual(matrices, rank, 1, random_state=rng)
        for B_i, dual_B_i in zip(B_is, dual_B_is):
            assert_array_equal(B_i, dual_B_i)

        # Check that we get value error if aux_init is tensor of wrong size (mode 0 or 2)
        # and if any of the tensors have wrong size (mode 1) or the list has the wrong length (mode 1)
        I = tl.shape(A)[0]
        J_is = [tl.shape(B_i)[0] for B_i in B_is]
        K = tl.shape(C)[0]

        invalid_A = rng.random((I + 1, rank))
        invalid_C = rng.random((K + 1, rank))
        invalid_B_is = [rng.random((J_i, rank)) for J_i in J_is]
        invalid_B_is[0] = rng.random((J_is[0] + 1, rank))

        penalty = self.PenaltyType(aux_init=aux_init, dual_init=invalid_A, **self.penalty_default_kwargs)
        with pytest.raises(ValueError):
            penalty.init_dual(matrices, rank, 0, random_state=rng)
        penalty = self.PenaltyType(aux_init=aux_init, dual_init=invalid_C, **self.penalty_default_kwargs)
        with pytest.raises(ValueError):
            penalty.init_dual(matrices, rank, 2, random_state=rng)
        penalty = self.PenaltyType(aux_init=aux_init, dual_init=invalid_B_is, **self.penalty_default_kwargs)
        with pytest.raises(ValueError):
            penalty.init_dual(matrices, rank, 1, random_state=rng)
        penalty = self.PenaltyType(aux_init=aux_init, dual_init=B_is + B_is, **self.penalty_default_kwargs)
        with pytest.raises(ValueError):
            penalty.init_dual(matrices, rank, 1, random_state=rng)

        # Check that mode 0 and 2 cannot accept list of matrices
        penalty = self.PenaltyType(aux_init=aux_init, dual_init=B_is, **self.penalty_default_kwargs)
        with pytest.raises(TypeError):
            penalty.init_dual(matrices, rank, 0, random_state=rng)
        with pytest.raises(TypeError):
            penalty.init_dual(matrices, rank, 2, random_state=rng)

        # Check that mode 1 cannot accept single matrix
        penalty = self.PenaltyType(aux_init=aux_init, dual_init=A, **self.penalty_default_kwargs)
        with pytest.raises(TypeError):
            penalty.init_dual(matrices, rank, 1, random_state=rng)

        # Check that we get value error if aux init is str but not "random_uniform" or "random_standard_normal"
        penalty = self.PenaltyType(aux_init=aux_init, dual_init="invalid init name", **self.penalty_default_kwargs)
        for mode in range(2):
            with pytest.raises(ValueError):
                penalty.init_dual(matrices, rank, mode=mode, random_state=rng)

    def test_penalty(self, rng):
        # TODO: Implement for all subclasses, where we check penalty with some known values
        raise NotImplementedError

    def test_subtract_from_aux(self, random_matrices):
        penalty = self.PenaltyType(**self.penalty_default_kwargs)
        for matrix in random_matrices:
            assert_array_equal(penalty.subtract_from_aux(matrix, matrix), 0)

    def test_subtract_from_auxes(self, random_matrices):
        penalty = self.PenaltyType(**self.penalty_default_kwargs)
        zero_matrices = penalty.subtract_from_auxes(random_matrices, random_matrices)
        for zeros in zero_matrices:
            assert_array_equal(zeros, 0)

    def test_aux_as_matrix(self, random_matrix):
        # Check that this is an identity operator.
        penalty = self.PenaltyType(**self.penalty_default_kwargs)
        random_matrix2 = penalty.aux_as_matrix(random_matrix)
        assert_array_equal(random_matrix, random_matrix2)

    def test_auxes_as_matrices(self, random_matrices):
        # Check that this is an identity operator.
        penalty = self.PenaltyType(**self.penalty_default_kwargs)
        random_matrices2 = penalty.auxes_as_matrices(random_matrices)
        assert len(random_matrices) == len(random_matrices2)
        for random_matrix, random_matrix2 in zip(random_matrices, random_matrices2):
            assert_array_equal(random_matrix, random_matrix2)


class BaseTestFactorMatricesPenalty(BaseTestADMMPenalty):  # e.g. PARAFAC2
    def get_stationary_matrix(self, rng, shape):
        raise NotImplementedError

    def get_non_stationary_matrix(self, rng, shape):
        raise NotImplementedError

    @pytest.fixture
    def stationary_matrices(self, rng):
        n_rows = rng.randint(1, 10)
        n_matrices = rng.randint(1, 10)
        shapes = tuple((n_rows, rng.randint(1, 10)) for k in range(n_matrices))
        return self.get_stationary_matrices(rng, shapes)

    @pytest.fixture
    def non_stationary_matrices(self, rng):
        n_rows = rng.randint(1, 10)
        n_matrices = rng.randint(1, 10)
        shapes = tuple((n_rows, rng.randint(1, 10)) for k in range(n_matrices))
        return self.get_non_stationary_matrices(rng, shapes)

    def test_factor_matrices_update_stationary_point(self, stationary_matrices):
        feasibility_penalties = [10] * len(stationary_matrices)
        auxes = [None] * len(stationary_matrices)
        penalty = self.PenaltyType(**self.penalty_default_kwargs)

        out = penalty.factor_matrices_update(stationary_matrices, feasibility_penalties, auxes)
        for stationary_matrix, out_matrix in zip(stationary_matrices, out):
            assert_array_almost_equal(stationary_matrix, out_matrix)

    def test_factor_matrices_update_changes_input(self, non_stationary_matrices):
        feasibility_penalties = [10] * len(non_stationary_matrices)
        auxes = [None] * len(non_stationary_matrices)
        penalty = self.PenaltyType(**self.penalty_default_kwargs)

        out = penalty.factor_matrices_update(non_stationary_matrices, feasibility_penalties, auxes)
        for non_stationary_matrix, out_matrix in zip(non_stationary_matrices, out):
            assert not np.allclose(out_matrix, non_stationary_matrix)

    def test_factor_matrices_update_reduces_penalty(self, random_matrices):
        penalty = self.PenaltyType(**self.penalty_default_kwargs)
        feasibility_penalties = [10] * len(random_matrices)
        auxes = [None] * len(random_matrices)
        initial_penalty = penalty.penalty(random_matrices)
        out = penalty.factor_matrices_update(random_matrices, feasibility_penalties, auxes)
        assert penalty.penalty(out) <= initial_penalty


class BaseTestFactorMatrixPenalty(BaseTestFactorMatricesPenalty):  # e.g. unimodality
    def get_stationary_matrix(self, rng, shape):
        raise NotImplementedError

    def get_stationary_matrices(self, rng, shapes):
        return [self.get_stationary_matrix(rng, shape) for shape in shapes]

    def get_non_stationary_matrix(self, rng, shape):
        raise NotImplementedError

    def get_non_stationary_matrices(self, rng, shapes):
        return [self.get_non_stationary_matrix(rng, shape) for shape in shapes]

    @pytest.fixture
    def stationary_matrix(self, rng):
        n_columns = rng.randint(1, 10)
        n_rows = rng.randint(1, 10)
        shape = (n_rows, n_columns)
        return self.get_stationary_matrix(rng, shape)

    @pytest.fixture
    def non_stationary_matrix(self, rng):
        n_columns = rng.randint(1, 10)
        n_rows = rng.randint(1, 10)
        shape = (n_rows, n_columns)
        return self.get_non_stationary_matrix(rng, shape)

    def test_factor_matrix_update_stationary_point(self, stationary_matrix):
        penalty = self.PenaltyType(**self.penalty_default_kwargs)

        out = penalty.factor_matrix_update(stationary_matrix, 10, None)
        assert_array_almost_equal(stationary_matrix, out)

    def test_factor_matrix_update_changes_input(self, non_stationary_matrix):
        penalty = self.PenaltyType(**self.penalty_default_kwargs)

        out = penalty.factor_matrix_update(non_stationary_matrix, 10, None)
        assert not np.allclose(out, non_stationary_matrix)

    def test_factor_matrix_update_reduces_penalty(self, random_matrix):
        penalty = self.PenaltyType(**self.penalty_default_kwargs)

        initial_penalty = penalty.penalty(random_matrix)
        out = penalty.factor_matrix_update(random_matrix, 10, None)
        assert penalty.penalty(out) <= initial_penalty


class BaseTestRowVectorPenalty(BaseTestFactorMatrixPenalty):  # e.g. non-negativity
    def get_stationary_row(self, rng, n_columns):
        raise NotImplementedError

    def get_stationary_matrix(self, rng, shape):
        return tl.stack([self.get_stationary_row(rng, shape[1]) for _ in range(shape[0])], axis=0)

    def get_non_stationary_row(self, rng, n_columns):
        raise NotImplementedError

    def get_non_stationary_matrix(self, rng, shape):
        return tl.stack([self.get_non_stationary_row(rng, shape[1]) for _ in range(shape[0])], axis=0)

    @pytest.fixture
    def stationary_row(self, rng):
        n_columns = rng.randint(1, 10)
        return self.get_stationary_row(rng, n_columns)

    @pytest.fixture
    def non_stationary_row(self, rng):
        rank = rng.randint(1, 10)
        return self.get_non_stationary_row(rng, rank)

    def test_row_update_stationary_point(self, stationary_row):
        penalty = self.PenaltyType(**self.penalty_default_kwargs)

        out = penalty.factor_matrix_row_update(stationary_row, 10, None)
        assert_array_almost_equal(stationary_row, out)

    def test_row_update_changes_input(self, non_stationary_row):
        penalty = self.PenaltyType(**self.penalty_default_kwargs)

        out = penalty.factor_matrix_row_update(non_stationary_row, 10, None)
        assert not np.allclose(out, non_stationary_row)

    def test_row_update_reduces_penalty(self, random_row):
        penalty = self.PenaltyType(**self.penalty_default_kwargs)

        initial_penalty = penalty.penalty(random_row)
        out = penalty.factor_matrix_row_update(random_row, 10, None)
        assert penalty.penalty(out) <= initial_penalty


class MixinTestHardConstraint:
    def test_penalty(self, random_ragged_cmf):
        cmf, shapes, rank = random_ragged_cmf
        weights, (A, B_is, C) = cmf
        penalty = self.PenaltyType(**self.penalty_default_kwargs)
        assert penalty.penalty(A) == 0
        assert penalty.penalty(B_is) == 0
        assert penalty.penalty(C) == 0


class TestL1Penalty(BaseTestRowVectorPenalty):
    PenaltyType = penalties.L1Penalty
    penalty_default_kwargs = {"reg_strength": 1}

    @pytest.mark.parametrize("non_negativity", [True, False])
    def test_row_update_stationary_point(self, non_negativity):
        stationary_matrix_row = np.zeros((1, 4))
        l1_penalty = penalties.L1Penalty(0.1, non_negativity=non_negativity)

        out = l1_penalty.factor_matrix_row_update(stationary_matrix_row, 10, None)
        assert_array_almost_equal(stationary_matrix_row, out)

    @pytest.mark.parametrize("non_negativity", [True, False])
    def test_factor_matrix_update_stationary_point(self, non_negativity):
        stationary_matrix = np.zeros((10, 3))
        l1_penalty = penalties.L1Penalty(0.1, non_negativity=non_negativity)

        out = l1_penalty.factor_matrix_update(stationary_matrix, 10, None)
        assert_array_almost_equal(stationary_matrix, out)

    @pytest.mark.parametrize("non_negativity", [True, False])
    def test_factor_matrices_update_stationary_point(self, non_negativity):
        stationary_matrices = [np.zeros((10, 3)) for i in range(5)]
        feasibility_penalties = [10] * len(stationary_matrices)
        auxes = [None] * len(stationary_matrices)
        l1_penalty = penalties.L1Penalty(0.1, non_negativity=non_negativity)

        out = l1_penalty.factor_matrices_update(stationary_matrices, feasibility_penalties, auxes)
        for stationary_matrix, out_matrix in zip(stationary_matrices, out):
            assert_array_almost_equal(stationary_matrix, out_matrix)

    @pytest.mark.parametrize("non_negativity", [True, False])
    def test_row_update_reduces_penalty(self, random_row, non_negativity):
        l1_penalty = penalties.L1Penalty(0.1, non_negativity=non_negativity)

        initial_penalty = l1_penalty.penalty(random_row)
        out = l1_penalty.factor_matrix_row_update(random_row, 10, None)
        assert l1_penalty.penalty(out) <= initial_penalty

    @pytest.mark.parametrize("non_negativity", [True, False])
    def test_factor_matrix_update_reduces_penalty(self, random_matrix, non_negativity):
        l1_penalty = penalties.L1Penalty(0.1, non_negativity=non_negativity)

        initial_penalty = l1_penalty.penalty(random_matrix)
        out = l1_penalty.factor_matrix_update(random_matrix, 10, None)
        assert l1_penalty.penalty(out) <= initial_penalty

    @pytest.mark.parametrize("non_negativity", [True, False])
    def test_factor_matrices_update_reduces_penalty(self, random_matrices, non_negativity):
        l1_penalty = penalties.L1Penalty(0.1, non_negativity=non_negativity)
        feasibility_penalties = [10] * len(random_matrices)
        auxes = [None] * len(random_matrices)
        initial_penalty = l1_penalty.penalty(random_matrices)
        out = l1_penalty.factor_matrices_update(random_matrices, feasibility_penalties, auxes)
        assert l1_penalty.penalty(out) <= initial_penalty

    @pytest.mark.parametrize("non_negativity", [True, False])
    def test_row_update_changes_input(self, random_row, non_negativity):
        l1_penalty = penalties.L1Penalty(0.1, non_negativity=non_negativity)

        out = l1_penalty.factor_matrix_row_update(random_row, 10, None)
        assert not np.allclose(out, random_row)

    @pytest.mark.parametrize("non_negativity", [True, False])
    def test_factor_matrix_update_changes_input(self, random_matrix, non_negativity):
        l1_penalty = penalties.L1Penalty(0.1, non_negativity=non_negativity)

        out = l1_penalty.factor_matrix_update(random_matrix, 10, None)
        assert not np.allclose(out, random_matrix)

    @pytest.mark.parametrize("non_negativity", [True, False])
    def test_factor_matrices_update_changes_input(self, random_matrices, non_negativity):
        feasibility_penalties = [10] * len(random_matrices)
        auxes = [None] * len(random_matrices)
        l1_penalty = penalties.L1Penalty(0.1, non_negativity=non_negativity)

        out = l1_penalty.factor_matrices_update(random_matrices, feasibility_penalties, auxes)
        for random_matrix, out_matrix in zip(random_matrices, out):
            assert not np.allclose(random_matrix, out_matrix)

    @pytest.mark.parametrize("non_negativity", [True, False])
    def test_factor_matrix_update_sets_small_weights_to_zero(self, random_matrix, non_negativity):
        random_matrix /= tl.abs(random_matrix).max()
        feasibility_penalty = 1
        aux = None
        l1_penalty = penalties.L1Penalty(1, non_negativity=non_negativity)

        out = l1_penalty.factor_matrix_update(random_matrix, feasibility_penalty, aux)
        assert_array_almost_equal(out, 0)

    def test_non_negativity_sets_negative_values_to_zero(self):
        negative_matrix = tl.ones((30, 5)) * (-100)
        feasibility_penalty = 1
        aux = None
        l1_penalty = penalties.L1Penalty(1, non_negativity=True)

        out = l1_penalty.factor_matrix_update(negative_matrix, feasibility_penalty, aux)
        assert_array_equal(out, 0)

    def test_penalty(self, random_ragged_cmf):
        cmf, shapes, rank = random_ragged_cmf
        weights, (A, B_is, C) = cmf
        l1_penalty = self.PenaltyType(reg_strength=1)
        assert l1_penalty.penalty(A) == pytest.approx(tl.sum(tl.abs(A)))
        assert l1_penalty.penalty(B_is) == pytest.approx(sum(tl.sum(tl.abs(B_i)) for B_i in B_is))
        assert l1_penalty.penalty(C) == pytest.approx(tl.sum(tl.abs(C)))

        l1_penalty = self.PenaltyType(reg_strength=2)
        assert l1_penalty.penalty(A) == pytest.approx(2 * tl.sum(tl.abs(A)))
        assert l1_penalty.penalty(B_is) == pytest.approx(2 * sum(tl.sum(tl.abs(B_i)) for B_i in B_is))
        assert l1_penalty.penalty(C) == pytest.approx(2 * tl.sum(tl.abs(C)))

        l1_penalty = self.PenaltyType(reg_strength=0)
        assert l1_penalty.penalty(A) == 0
        assert l1_penalty.penalty(B_is) == 0
        assert l1_penalty.penalty(C) == 0

        with pytest.raises(ValueError):
            l1_penalty = self.PenaltyType(reg_strength=-1)


class TestBoxConstraint(MixinTestHardConstraint, BaseTestRowVectorPenalty):
    PenaltyType = penalties.BoxConstraint
    penalty_default_kwargs = {"min_val": 0, "max_val": 1}

    def get_stationary_row(self, rng, n_columns):
        stationary_row = rng.uniform(size=(1, n_columns), low=0, high=1)
        return stationary_row

    def get_non_stationary_row(self, rng, n_columns):
        random_row = rng.uniform(size=(1, n_columns))
        random_row[0] += 100
        return random_row


class TestL2BallConstraint(MixinTestHardConstraint, BaseTestFactorMatrixPenalty):
    PenaltyType = penalties.L2Ball
    penalty_default_kwargs = {"max_norm": 1}

    def get_stationary_matrix(self, rng, shape):
        random_matrix = rng.random(shape)
        return random_matrix / tl.sqrt(shape[0])

    def get_non_stationary_matrix(self, rng, shape):
        random_matrix = rng.random(shape)
        return 10 + random_matrix / tl.sqrt(shape[0])


class TestUnitSimplex(MixinTestHardConstraint, BaseTestFactorMatrixPenalty):
    PenaltyType = penalties.UnitSimplex

    def get_stationary_matrix(self, rng, shape):
        random_matrix = rng.uniform(size=shape)
        col_sum = tl.sum(random_matrix, axis=0)
        return random_matrix / col_sum

    def get_non_stationary_matrix(self, rng, shape):
        random_matrix = rng.uniform(size=shape)
        col_sum = tl.sum(random_matrix, axis=0)
        return 10 + random_matrix / col_sum


class TestNonNegativity(MixinTestHardConstraint, BaseTestRowVectorPenalty):
    PenaltyType = penalties.NonNegativity

    def get_stationary_row(self, rng, n_columns):
        random_row = rng.uniform(size=(1, n_columns))
        return random_row

    def get_non_stationary_row(self, rng, n_columns):
        random_row = rng.uniform(size=(1, n_columns))
        random_row[0] = -1
        return random_row


class TestParafac2(BaseTestFactorMatricesPenalty):
    PenaltyType = penalties.Parafac2

    def test_factor_matrices_update_stationary_point(self, rng):
        deltaB = rng.standard_normal((3, 3))
        Pks = [np.linalg.qr(rng.standard_normal(size=(10, 3)))[0] for _ in range(5)]
        auxes = Pks, deltaB
        stationary_matrices = [Pk @ deltaB for Pk in Pks]

        feasibility_penalties = [10] * len(stationary_matrices)
        pf2_penalty = penalties.Parafac2()

        out = pf2_penalty.factor_matrices_update(stationary_matrices, feasibility_penalties, auxes)
        assert_array_almost_equal(deltaB, out[1])
        for Pk, out_matrix in zip(Pks, out[0]):
            assert_array_almost_equal(Pk, out_matrix)

    def test_factor_matrices_update_reduces_penalty(self, rng, random_matrices):
        deltaB = rng.standard_normal((3, 3))
        Pks = [np.linalg.qr(rng.standard_normal(size=(10, 3)))[0] for _ in range(5)]
        auxes = Pks, deltaB

        feasibility_penalties = [10] * len(random_matrices)
        pf2_penalty = penalties.Parafac2()

        initial_penalty = pf2_penalty.penalty(pf2_penalty.auxes_as_matrices(auxes))
        out = pf2_penalty.factor_matrices_update(random_matrices, feasibility_penalties, auxes)
        assert pf2_penalty.penalty(pf2_penalty.auxes_as_matrices(out)) <= initial_penalty

    def test_factor_matrices_update_changes_input(self, random_matrices, rng):
        deltaB = rng.standard_normal((3, 3))
        Pks = [np.linalg.qr(rng.standard_normal(size=(10, 3)))[0] for _ in range(5)]
        auxes = Pks, deltaB

        feasibility_penalties = [10] * len(random_matrices)
        pf2_penalty = penalties.Parafac2()

        out = pf2_penalty.factor_matrices_update(random_matrices, feasibility_penalties, auxes)
        constructed_out = [Pk @ out[1] for Pk in out[0]]
        for random_matrix, out_matrix in zip(random_matrices, constructed_out):
            assert not np.allclose(random_matrix, out_matrix)

    @pytest.mark.parametrize("dual_init", ["random_uniform", "random_standard_normal", "zeros"])
    def test_init_aux(self, rng, random_ragged_cmf, dual_init):
        # TODO: split into several tests
        cmf, shapes, rank = random_ragged_cmf
        weights, (A, B_is, C) = cmf
        matrices = cmf.to_matrices()

        penalty = self.PenaltyType(**self.penalty_default_kwargs)
        # Test that mode and rank needs int input
        with pytest.raises(TypeError):
            penalty.init_aux(matrices, rank, mode=None)
        with pytest.raises(TypeError):
            penalty.init_aux(matrices, rank=None, mode=1)

        # Check that initialize fails with mode != 1
        with pytest.raises(ValueError):
            penalty.init_aux(matrices, rank, 0, random_state=rng)
        with pytest.raises(ValueError):
            penalty.init_aux(matrices, rank, 2, random_state=rng)

        # Test that the init method must be a valid type
        invalid_inits = [
            None,
            1,
            1.1,
            (None, None),
            ([None] * len(matrices), tl.zeros((rank, rank))),
            ([tl.eye(J_i, rank) for J_i, k in shapes], None),
        ]
        for invalid_init in invalid_inits:
            penalty = self.PenaltyType(aux_init=invalid_init, dual_init=dual_init, **self.penalty_default_kwargs)
            with pytest.raises(TypeError):
                penalty.init_aux(matrices, rank, mode=1, random_state=rng)

        # Test that init works with random uniform init
        penalty = self.PenaltyType(aux_init="random_uniform", dual_init=dual_init, **self.penalty_default_kwargs)
        init_bases, init_coordinates = penalty.init_aux(matrices, rank, mode=1, random_state=rng)
        assert tl.shape(init_coordinates) == (rank, rank)
        assert tl.all(init_coordinates >= 0)
        assert tl.all(init_coordinates <= 1)

        for init_basis, B_i in zip(init_bases, B_is):
            assert_array_almost_equal(init_basis.T @ init_basis, tl.eye(rank))
            assert tl.shape(init_basis) == tl.shape(B_i)

        # Test that init works with random standard normal init
        penalty = self.PenaltyType(
            aux_init="random_standard_normal", dual_init=dual_init, **self.penalty_default_kwargs
        )
        init_bases, init_coordinates = penalty.init_aux(matrices, rank, mode=1, random_state=rng)
        assert tl.shape(init_coordinates) == (rank, rank)
        for init_basis, B_i in zip(init_bases, B_is):
            assert_array_almost_equal(init_basis.T @ init_basis, tl.eye(rank))
            assert tl.shape(init_basis) == tl.shape(B_i)

        # Test that init works with zeros init
        penalty = self.PenaltyType(aux_init="zeros", dual_init=dual_init, **self.penalty_default_kwargs)
        init_bases, init_coordinates = penalty.init_aux(matrices, rank, mode=1, random_state=rng)
        assert tl.shape(init_coordinates) == (rank, rank)
        assert_array_equal(init_coordinates, 0)
        for init_basis, B_i in zip(init_bases, B_is):
            assert_array_almost_equal(init_basis.T @ init_basis, tl.eye(rank))
            assert tl.shape(init_basis) == tl.shape(B_i)

        # Test that init works with specified init
        penalty = self.PenaltyType(
            aux_init=(init_bases, init_coordinates), dual_init=dual_init, **self.penalty_default_kwargs
        )
        init_bases_2, init_coordinates_2 = penalty.init_aux(matrices, rank, mode=1, random_state=rng)
        assert_array_equal(init_coordinates, init_coordinates_2)
        for init_basis, init_basis_2 in zip(init_bases, init_bases_2):
            assert_array_equal(init_basis, init_basis_2)

        # Test with various invalid basis matrix lists
        all_invalid_bases = []

        invalid_init_bases = copy(init_bases)
        invalid_init_bases[0] = tl.zeros(tl.shape(invalid_init_bases[0]))  # Not orthogonal is invalid
        all_invalid_bases.append(invalid_init_bases)

        invalid_init_bases = copy(init_bases)
        invalid_init_bases[0] = tl.zeros((tl.shape(invalid_init_bases[0])[0] + 1, rank))  # Wrong shape
        all_invalid_bases.append(invalid_init_bases)

        invalid_init_bases = copy(init_bases)
        invalid_init_bases[0] = tl.zeros((*tl.shape(invalid_init_bases[0]), 2))  # Wrong order
        all_invalid_bases.append(invalid_init_bases)

        all_invalid_bases.append(init_bases + init_bases)  # Wrong number of matrices

        for invalid_init_bases in all_invalid_bases:
            aux_init = (invalid_init_bases, init_coordinates)
            penalty = self.PenaltyType(aux_init=aux_init, dual_init=dual_init, **self.penalty_default_kwargs)

            with pytest.raises(ValueError):
                penalty.init_aux(matrices, rank, mode=1, random_state=rng)

        invalid_coordinates = [
            tl.zeros((rank, rank, rank)),
            tl.zeros((rank + 1, rank)),
            tl.zeros((rank, rank + 1)),
            tl.zeros((rank + 1, rank + 1)),
        ]
        for invalid_init_coordinates in invalid_coordinates:
            aux_init = (init_bases, invalid_init_coordinates)
            penalty = self.PenaltyType(aux_init=aux_init, dual_init=dual_init, **self.penalty_default_kwargs)

            with pytest.raises(ValueError):
                penalty.init_aux(matrices, rank, mode=1, random_state=rng)

        # Check that we get value error if aux init is str but not "random_uniform" or "random_standard_normal"
        penalty = self.PenaltyType(aux_init="invalid init name", dual_init=dual_init, **self.penalty_default_kwargs)
        for mode in range(2):
            with pytest.raises(ValueError):
                penalty.init_aux(matrices, rank, mode=mode, random_state=None)

    def test_subtract_from_aux(self,):
        penalty = self.PenaltyType(**self.penalty_default_kwargs)
        with pytest.raises(TypeError):
            penalty.subtract_from_aux(None, None)

    def test_subtract_from_auxes(self, rng, random_ragged_cmf):
        cmf, shapes, rank = random_ragged_cmf
        weights, (A, B_is, C) = cmf
        matrices = cmf.to_matrices()

        penalty = self.PenaltyType(**self.penalty_default_kwargs)
        auxes = penalty.init_aux(matrices, rank, 1, random_state=rng)
        aux_matrices = penalty.auxes_as_matrices(auxes)

        aux_diffs = penalty.subtract_from_auxes(auxes, B_is)
        for i, B_i in enumerate(B_is):
            aux = aux_matrices[i]
            aux_diff = aux_diffs[i]

            assert_array_almost_equal(aux_diff, aux - B_i)

    def test_aux_as_matrix(self, random_matrix):
        penalty = self.PenaltyType(**self.penalty_default_kwargs)
        # Check that this raises TypeError
        with pytest.raises(TypeError):
            penalty.aux_as_matrix(random_matrix)

    def test_auxes_as_matrices(self, rng, random_ragged_cmf):
        cmf, shapes, rank = random_ragged_cmf
        weights, (A, B_is, C) = cmf
        matrices = cmf.to_matrices()

        penalty = self.PenaltyType(**self.penalty_default_kwargs)
        auxes = penalty.init_aux(matrices, rank, 1, random_state=rng)
        aux_matrices = penalty.auxes_as_matrices(auxes)

        for i, aux in enumerate(aux_matrices):
            aux = aux_matrices[i]
            basis_i = auxes[0][i]

            assert_array_almost_equal(aux, basis_i @ auxes[1])

    def test_penalty(self, random_ragged_cmf):
        cmf, shapes, rank = random_ragged_cmf
        weights, (A, B_is, C) = cmf
        penalty = self.PenaltyType(**self.penalty_default_kwargs)
        assert penalty.penalty(B_is) == 0

        with pytest.raises(TypeError):
            penalty.penalty(A)
