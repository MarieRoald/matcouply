# MIT License: Copyright (c) 2022, Marie Roald.
# See the LICENSE file in the root directory for full license text.

import numpy as np
import pytest
import tensorly as tl
from tensorly.testing import assert_array_equal

from matcouply import penalties


def assert_allclose(actual, desired, *args, **kwargs):
    np.testing.assert_allclose(tl.to_numpy(actual), tl.to_numpy(desired), *args, **kwargs)


# # Interfaces only, not code to be run or inherited from:
class BaseTestADMMPenalty:
    PenaltyType = penalties.ADMMPenalty
    penalty_default_kwargs = {}
    min_rows = 1
    max_rows = 10
    min_columns = 1
    max_columns = 10
    min_matrices = 1
    max_matrices = 10

    # TODO: Fixtures for random_matrix random_ragged_cmf, etc as part of this which uses the min_rows, etc...
    atol = 1e-10

    @property
    def rtol(self):
        if tl.get_backend() == "numpy":
            return 1e-6
        return 500 * 1e-6  # Single precision backends need less strict tests

    @pytest.fixture
    def random_row(self, rng):
        n_columns = rng.randint(self.min_columns, self.max_columns + 1)
        return tl.tensor(rng.standard_normal(n_columns))

    @pytest.fixture
    def random_matrix(self, rng):
        n_rows = rng.randint(self.min_rows, self.max_rows + 1)
        n_columns = rng.randint(self.min_columns, self.max_columns + 1)
        return tl.tensor(rng.standard_normal((n_rows, n_columns)))

    @pytest.fixture
    def random_matrices(self, rng):
        n_rows = rng.randint(self.min_rows, self.max_rows + 1)
        n_columns = rng.randint(self.min_columns, self.max_columns + 1)
        n_matrices = rng.randint(self.min_matrices, self.max_matrices + 1)
        return [tl.tensor(rng.standard_normal((n_rows, n_columns))) for i in range(n_matrices)]

    @pytest.mark.parametrize("dual_init", ["random_uniform", "random_standard_normal", "zeros"])
    def test_uniform_init_aux(self, rng, random_ragged_cmf, dual_init):
        cmf, shapes, rank = random_ragged_cmf
        matrices = cmf.to_matrices()
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

    @pytest.mark.parametrize("dual_init", ["random_uniform", "random_standard_normal", "zeros"])
    def test_standard_normal_init_aux(self, rng, random_ragged_cmf, dual_init):
        cmf, shapes, rank = random_ragged_cmf
        matrices = cmf.to_matrices()
        penalty = self.PenaltyType(
            aux_init="random_standard_normal", dual_init=dual_init, **self.penalty_default_kwargs
        )

        init_matrix = penalty.init_aux(matrices, rank, mode=0, random_state=rng)
        assert init_matrix.shape[0] == len(shapes)
        assert init_matrix.shape[1] == rank

        init_matrices = penalty.init_aux(matrices, rank, mode=1, random_state=rng)
        for init_matrix, shape in zip(init_matrices, shapes):
            assert init_matrix.shape[0] == shape[0]
            assert init_matrix.shape[1] == rank

        init_matrix = penalty.init_aux(matrices, rank, mode=2, random_state=rng)
        assert init_matrix.shape[0] == shapes[0][1]
        assert init_matrix.shape[1] == rank

    @pytest.mark.parametrize("dual_init", ["random_uniform", "random_standard_normal", "zeros"])
    def test_zeros_init_aux(self, rng, random_ragged_cmf, dual_init):
        cmf, shapes, rank = random_ragged_cmf
        matrices = cmf.to_matrices()
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

    @pytest.mark.parametrize("dual_init", ["random_uniform", "random_standard_normal", "zeros"])
    def test_given_init_aux(self, rng, random_ragged_cmf, dual_init):
        cmf, shapes, rank = random_ragged_cmf
        matrices = cmf.to_matrices()

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

    @pytest.mark.parametrize("dual_init", ["random_uniform", "random_standard_normal", "zeros"])
    @pytest.mark.parametrize("aux_init", ["random_uniform", "random_standard_normal", "zeros"])
    def test_rank_and_mode_validation_for_init_aux(self, rng, random_ragged_cmf, dual_init, aux_init):
        cmf, shapes, rank = random_ragged_cmf
        matrices = cmf.to_matrices()
        penalty = self.PenaltyType(aux_init="zeros", dual_init=dual_init, **self.penalty_default_kwargs)
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

    @pytest.mark.parametrize("dual_init", ["random_uniform", "random_standard_normal", "zeros"])
    def test_validating_given_init_aux(self, rng, random_ragged_cmf, dual_init):
        cmf, shapes, rank = random_ragged_cmf
        matrices = cmf.to_matrices()

        # Check that we get value error if aux_init is tensor of wrong size (mode 0 or 2)
        # and if any of the tensors have wrong size (mode 1) or the list has the wrong length (mode 1)
        weights, (A, B_is, C) = cmf
        I = tl.shape(A)[0]
        J_is = [tl.shape(B_i)[0] for B_i in B_is]
        K = tl.shape(C)[0]

        invalid_A = tl.tensor(rng.random_sample((I + 1, rank)))
        invalid_C = tl.tensor(rng.random_sample((K + 1, rank)))
        invalid_B_is = [tl.tensor(rng.random_sample((J_i, rank))) for J_i in J_is]
        invalid_B_is[0] = tl.tensor(rng.random_sample((J_is[0] + 1, rank)))

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

    @pytest.mark.parametrize("dual_init", ["random_uniform", "random_standard_normal", "zeros"])
    def test_input_validation_for_init_aux(self, rng, random_ragged_cmf, dual_init):
        cmf, shapes, rank = random_ragged_cmf
        matrices = cmf.to_matrices()
        # Test that the init method must be a valid type
        invalid_inits = [None, 1, 1.1]
        for invalid_init in invalid_inits:
            penalty = self.PenaltyType(aux_init=invalid_init, dual_init=dual_init, **self.penalty_default_kwargs)
            for mode in range(2):
                with pytest.raises(TypeError):
                    penalty.init_aux(matrices, rank, mode=mode, random_state=rng)

        # Check that we get value error if aux init is str but not "random_uniform" or "random_standard_normal"
        penalty = self.PenaltyType(aux_init="invalid init name", dual_init=dual_init, **self.penalty_default_kwargs)
        for mode in range(2):
            with pytest.raises(ValueError):
                penalty.init_aux(matrices, rank, mode=mode, random_state=None)

    @pytest.mark.parametrize("aux_init", ["random_uniform", "random_standard_normal", "zeros"])
    def test_uniform_init_dual(self, rng, random_ragged_cmf, aux_init):
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

    @pytest.mark.parametrize("aux_init", ["random_uniform", "random_standard_normal", "zeros"])
    def test_standard_normal_init_dual(self, rng, random_ragged_cmf, aux_init):
        cmf, shapes, rank = random_ragged_cmf
        matrices = cmf.to_matrices()
        # Test that init works with random standard normal init
        penalty = self.PenaltyType(aux_init=aux_init, dual_init="random_standard_normal", **self.penalty_default_kwargs)

        init_matrix = penalty.init_dual(matrices, rank, mode=0, random_state=rng)
        assert init_matrix.shape[0] == len(shapes)
        assert init_matrix.shape[1] == rank

        init_matrices = penalty.init_dual(matrices, rank, mode=1, random_state=rng)
        for init_matrix, shape in zip(init_matrices, shapes):
            assert init_matrix.shape[0] == shape[0]
            assert init_matrix.shape[1] == rank

        init_matrix = penalty.init_dual(matrices, rank, mode=2, random_state=rng)
        assert init_matrix.shape[0] == shapes[0][1]
        assert init_matrix.shape[1] == rank

    @pytest.mark.parametrize("aux_init", ["random_uniform", "random_standard_normal", "zeros"])
    def test_zeros_init_dual(self, rng, random_ragged_cmf, aux_init):
        cmf, shapes, rank = random_ragged_cmf
        matrices = cmf.to_matrices()
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

    @pytest.mark.parametrize("aux_init", ["random_uniform", "random_standard_normal", "zeros"])
    def test_given_init_dual(self, rng, random_ragged_cmf, aux_init):
        cmf, shapes, rank = random_ragged_cmf
        matrices = cmf.to_matrices()
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

    @pytest.mark.parametrize("aux_init", ["random_uniform", "random_standard_normal", "zeros"])
    def test_validating_given_init_dual(self, rng, random_ragged_cmf, aux_init):
        cmf, shapes, rank = random_ragged_cmf
        matrices = cmf.to_matrices()
        weights, (A, B_is, C) = cmf

        # Check that we get value error if aux_init is tensor of wrong size (mode 0 or 2)
        # and if any of the tensors have wrong size (mode 1) or the list has the wrong length (mode 1)
        I = tl.shape(A)[0]
        J_is = [tl.shape(B_i)[0] for B_i in B_is]
        K = tl.shape(C)[0]

        invalid_A = tl.tensor(rng.random_sample((I + 1, rank)))
        invalid_C = tl.tensor(rng.random_sample((K + 1, rank)))
        invalid_B_is = [tl.tensor(rng.random_sample((J_i, rank))) for J_i in J_is]
        invalid_B_is[0] = tl.tensor(rng.random_sample((J_is[0] + 1, rank)))

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

    @pytest.mark.parametrize(
        "dual_init,", ["random_uniform", "random_standard_normal", "zeros"],
    )
    @pytest.mark.parametrize("aux_init", ["random_uniform", "random_standard_normal", "zeros"])
    def test_rank_and_mode_validation_for_init_dual(self, rng, random_ragged_cmf, dual_init, aux_init):
        cmf, shapes, rank = random_ragged_cmf
        matrices = cmf.to_matrices()
        # Test that init works with zeros init
        penalty = self.PenaltyType(aux_init=aux_init, dual_init=dual_init, **self.penalty_default_kwargs)

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

    @pytest.mark.parametrize("aux_init", ["random_uniform", "random_standard_normal", "zeros"])
    def test_input_validation_init_dual(self, rng, random_ragged_cmf, aux_init):
        cmf, shapes, rank = random_ragged_cmf
        matrices = cmf.to_matrices()
        # Test that the init method must be a valid type
        invalid_inits = [None, 1, 1.1]
        for invalid_init in invalid_inits:
            penalty = self.PenaltyType(aux_init=aux_init, dual_init=invalid_init, **self.penalty_default_kwargs)
            for mode in range(2):
                with pytest.raises(TypeError):
                    penalty.init_dual(matrices, rank, mode=mode, random_state=rng)

        # Check that we get value error if aux init is str but not "random_uniform" or "random_standard_normal"
        penalty = self.PenaltyType(aux_init=aux_init, dual_init="invalid init name", **self.penalty_default_kwargs)
        for mode in range(2):
            with pytest.raises(ValueError):
                penalty.init_dual(matrices, rank, mode=mode, random_state=rng)

    def test_penalty(self, rng):
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
    def get_invariant_matrices(self, rng, shape):
        return NotImplementedError

    def get_non_invariant_matrices(self, rng, shape):
        return NotImplementedError

    @pytest.fixture
    def invariant_matrices(self, rng):
        n_rows = rng.randint(self.min_rows, self.max_rows + 1)
        n_matrices = rng.randint(self.min_matrices, self.max_matrices + 1)
        shapes = tuple((n_rows, rng.randint(self.min_columns, self.max_columns + 1)) for k in range(n_matrices))
        return self.get_invariant_matrices(rng, shapes)

    @pytest.fixture
    def non_invariant_matrices(self, rng):
        n_rows = rng.randint(self.min_rows, self.max_rows + 1)
        n_matrices = rng.randint(self.min_matrices, self.max_matrices + 1)
        shapes = tuple((n_rows, rng.randint(self.min_columns, self.max_columns + 1)) for k in range(n_matrices))
        return self.get_non_invariant_matrices(rng, shapes)

    def test_factor_matrices_update_invariant_point(self, invariant_matrices):
        feasibility_penalties = [10] * len(invariant_matrices)
        auxes = [None] * len(invariant_matrices)
        penalty = self.PenaltyType(**self.penalty_default_kwargs)

        out = penalty.factor_matrices_update(invariant_matrices, feasibility_penalties, auxes)
        for invariant_matrix, out_matrix in zip(invariant_matrices, out):
            assert_allclose(invariant_matrix, out_matrix, rtol=self.rtol, atol=self.atol)

    def test_factor_matrices_update_changes_input(self, non_invariant_matrices):
        feasibility_penalties = [10] * len(non_invariant_matrices)
        auxes = [None] * len(non_invariant_matrices)
        penalty = self.PenaltyType(**self.penalty_default_kwargs)

        out = penalty.factor_matrices_update(non_invariant_matrices, feasibility_penalties, auxes)
        for non_invariant_matrix, out_matrix in zip(non_invariant_matrices, out):
            assert not np.allclose(out_matrix, non_invariant_matrix, rtol=self.rtol, atol=self.atol)

    def test_factor_matrices_update_reduces_penalty(self, random_matrices):
        penalty = self.PenaltyType(**self.penalty_default_kwargs)
        feasibility_penalties = [10] * len(random_matrices)
        auxes = [None] * len(random_matrices)
        initial_penalty = penalty.penalty(random_matrices)
        out = penalty.factor_matrices_update(random_matrices, feasibility_penalties, auxes)
        assert penalty.penalty(out) <= initial_penalty


class BaseTestFactorMatrixPenalty(BaseTestFactorMatricesPenalty):
    def get_invariant_matrix(self, rng, shape):
        raise NotImplementedError

    def get_invariant_matrices(self, rng, shapes):
        return [self.get_invariant_matrix(rng, shape) for shape in shapes]

    def get_non_invariant_matrix(self, rng, shape):
        raise NotImplementedError

    def get_non_invariant_matrices(self, rng, shapes):
        return [self.get_non_invariant_matrix(rng, shape) for shape in shapes]

    @pytest.fixture
    def invariant_matrix(self, rng):
        n_columns = rng.randint(self.min_columns, self.max_columns + 1)
        n_rows = rng.randint(self.min_rows, self.max_rows + 1)
        shape = (n_rows, n_columns)
        return self.get_invariant_matrix(rng, shape)

    @pytest.fixture
    def non_invariant_matrix(self, rng):
        n_columns = rng.randint(self.min_columns, self.max_columns + 1)
        n_rows = rng.randint(self.min_rows, self.max_rows + 1)
        shape = (n_rows, n_columns)
        return self.get_non_invariant_matrix(rng, shape)

    def test_factor_matrix_update_invariant_point(self, invariant_matrix):
        penalty = self.PenaltyType(**self.penalty_default_kwargs)

        out = penalty.factor_matrix_update(invariant_matrix, 10, None)
        assert_allclose(invariant_matrix, out, rtol=self.rtol, atol=self.atol)

    def test_factor_matrix_update_changes_input(self, non_invariant_matrix):
        penalty = self.PenaltyType(**self.penalty_default_kwargs)

        out = penalty.factor_matrix_update(non_invariant_matrix, 10, None)
        assert not np.allclose(out, non_invariant_matrix, rtol=self.rtol, atol=self.atol)

    def test_factor_matrix_update_reduces_penalty(self, random_matrix):
        penalty = self.PenaltyType(**self.penalty_default_kwargs)

        initial_penalty = penalty.penalty(random_matrix)
        out = penalty.factor_matrix_update(random_matrix, 10, None)
        assert penalty.penalty(out) <= initial_penalty


class BaseTestRowVectorPenalty(BaseTestFactorMatrixPenalty):  # e.g. non-negativity
    def get_invariant_row(self, rng, n_columns):
        raise NotImplementedError

    def get_invariant_matrix(self, rng, shape):
        return tl.stack([self.get_invariant_row(rng, shape[1]) for _ in range(shape[0])], axis=0)

    def get_non_invariant_row(self, rng, n_columns):
        raise NotImplementedError

    def get_non_invariant_matrix(self, rng, shape):
        return tl.stack([self.get_non_invariant_row(rng, shape[1]) for _ in range(shape[0])], axis=0)

    @pytest.fixture
    def invariant_row(self, rng):
        n_columns = rng.randint(self.min_columns, self.max_columns + 1)
        return self.get_invariant_row(rng, n_columns)

    @pytest.fixture
    def non_invariant_row(self, rng):
        n_columns = rng.randint(self.min_columns, self.max_columns + 1)
        return self.get_non_invariant_row(rng, n_columns)

    def test_row_update_invariant_point(self, invariant_row):
        penalty = self.PenaltyType(**self.penalty_default_kwargs)

        out = penalty.factor_matrix_row_update(invariant_row, 10, None)
        assert_allclose(invariant_row, out, rtol=self.rtol, atol=self.atol)

    def test_row_update_changes_input(self, non_invariant_row):
        penalty = self.PenaltyType(**self.penalty_default_kwargs)

        out = penalty.factor_matrix_row_update(non_invariant_row, 10, None)
        assert not np.allclose(out, non_invariant_row, rtol=self.rtol, atol=self.atol)

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
