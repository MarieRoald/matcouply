# MIT License: Copyright (c) 2022, Marie Roald.
# See the LICENSE file in the root directory for full license text.

from copy import copy
from unittest.mock import patch

import pytest
import tensorly as tl
from tensorly.testing import assert_array_equal

from matcouply import coupled_matrices, random
from matcouply.coupled_matrices import (
    CoupledMatrixFactorization,
    cmf_to_matrices,
    cmf_to_matrix,
)
from matcouply.testing import assert_allclose


def test_from_cp_tensor(rng):
    # Test that cp_tensor converted to coupled matrix factorization constructs same dense tensor
    cp_tensor = tl.random.random_cp((10, 15, 20), 3)
    cmf = CoupledMatrixFactorization.from_CPTensor(cp_tensor)

    dense_tensor_cp = cp_tensor.to_tensor()
    dense_tensor_cmf = cmf.to_tensor()
    assert_allclose(dense_tensor_cmf, dense_tensor_cp, rtol=1e-6)  # 1e-6 due to PyTorch single precision

    # Test that it fails when given CP tensor of order other than 3
    cp_tensor = tl.random.random_cp((10, 15, 20, 25), 3)
    with pytest.raises(ValueError):
        cmf = CoupledMatrixFactorization.from_CPTensor(cp_tensor)

    cp_tensor = tl.random.random_cp((10, 15), 3)
    with pytest.raises(ValueError):
        cmf = CoupledMatrixFactorization.from_CPTensor(cp_tensor)

    # Test that the B_is created from the cp tensor are copies, not the same view
    weights, (A, B_is, C) = cmf
    B_0 = tl.copy(B_is[0])
    B_is[1][0] += 5
    assert_allclose(B_0, B_is[0])


def test_from_parafac2_tensor(rng, random_ragged_shapes):
    # Test that parafac2 tensor converted to coupled matrix factorization constructs same dense tensor
    rank = 3
    random_ragged_shapes = [(max(rank, J_i), K) for J_i, K in random_ragged_shapes]
    parafac2_tensor = tl.random.random_parafac2(random_ragged_shapes, rank)
    cmf = CoupledMatrixFactorization.from_Parafac2Tensor(parafac2_tensor)

    dense_tensor_pf2 = parafac2_tensor.to_tensor()
    dense_tensor_cmf = cmf.to_tensor()
    assert_allclose(dense_tensor_cmf, dense_tensor_pf2)


def test_coupled_matrix_factorization(rng, random_regular_shapes):
    rank = 4
    cmf = random.random_coupled_matrices(random_regular_shapes, rank, random_state=rng)

    # Check that the length is equal to 2 (weights and factor matrices)
    assert len(cmf) == 2
    with pytest.raises(IndexError):
        cmf[2]

    # Check that the first element is the weight array
    assert_array_equal(cmf[0], cmf.weights)

    # Check that the second element is the factor matrices
    A1, B_is1, C1 = cmf.factors
    A2, B_is2, C2 = cmf[1]

    assert_array_equal(A1, A2)
    assert_array_equal(C1, C2)

    for B_i1, B_i2 in zip(B_is1, B_is2):
        assert_array_equal(B_i1, B_i2)


def test_validate_cmf(rng, random_ragged_cmf):
    cmf, shapes, rank = random_ragged_cmf
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
        B_is_copy[0] = 1
        coupled_matrices._validate_cmf((weights, (A, B_is_copy, C)))
    with pytest.raises(TypeError):
        B_is_copy = copy(B_is)
        B_is_copy[0] = None
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

    # ## Weights
    # The weights is a matrix
    with pytest.raises(ValueError):
        coupled_matrices._validate_cmf((tl.ones((rank, rank)), (A, B_is, C)))
    # Wrong number of weights
    with pytest.raises(ValueError):
        coupled_matrices._validate_cmf((tl.ones((rank + 1,)), (A, B_is, C)))

    # ## Factor matrices
    # One of the matrices is a third order tensor
    with pytest.raises(ValueError):
        coupled_matrices._validate_cmf((weights, (tl.tensor(rng.random_sample(size=(4, rank, rank))), B_is, C)))
    with pytest.raises(ValueError):
        coupled_matrices._validate_cmf((weights, (A, B_is, tl.tensor(rng.random_sample(size=(4, rank, rank))))))
    with pytest.raises(ValueError):
        B_is_copy = copy(B_is)
        B_is_copy[0] = tl.tensor(rng.random_sample(size=(4, rank, rank)))
        coupled_matrices._validate_cmf((weights, (A, B_is_copy, C)))

    # One of the matrices is a vector
    with pytest.raises(ValueError):
        coupled_matrices._validate_cmf((weights, (tl.tensor(rng.random_sample(size=(rank,))), B_is, C)))
    with pytest.raises(ValueError):
        coupled_matrices._validate_cmf((weights, (A, B_is, tl.tensor(rng.random_sample(size=(rank,))))))
    with pytest.raises(ValueError):
        B_is_copy = copy(B_is)
        B_is_copy[0] = tl.tensor(rng.random_sample(size=(rank,)))
        coupled_matrices._validate_cmf((weights, (A, B_is_copy, C)))

    # ## Check wrong rank
    # Check with incorrect rank for one of the factors
    invalid_A = tl.tensor(rng.random_sample((len(shapes), rank + 1)))
    invalid_C = tl.tensor(rng.random_sample((shapes[0][1], rank + 1)))
    invalid_B_is_2 = [tl.tensor(rng.random_sample((j_i, rank))) for j_i, k in shapes]
    invalid_B_is_2[0] = tl.tensor(rng.random_sample((shapes[0][0], rank + 1)))

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

    # Number of rows in A does not match number of B_is
    with pytest.raises(ValueError):
        coupled_matrices._validate_cmf((weights, (A[:-1, :], B_is, C)))
    with pytest.raises(ValueError):
        coupled_matrices._validate_cmf((weights, (A, B_is[:-1], C)))


def test_cmf_to_matrix(rng, random_ragged_cmf):
    cmf, shapes, rank = random_ragged_cmf
    weights, (A, B_is, C) = cmf

    # Compare matcouply implementation with our own implementation
    for i, B_i in enumerate(B_is):
        matrix = cmf.to_matrix(i)
        manually_assembled_matrix = (weights * A[i] * B_i) @ C.T
        assert_allclose(matrix, manually_assembled_matrix)

    # Test that it always fails when a single B_i is invalid and validate=True
    invalid_B_is = copy(B_is)
    invalid_B_is[0] = tl.tensor(rng.random_sample((tl.shape(B_is[0])[0], tl.shape(B_is[0])[1] + 1)))
    invalid_cmf = (weights, (A, invalid_B_is, C))

    for i, _ in enumerate(invalid_B_is):
        with pytest.raises(ValueError):
            cmf_to_matrix(invalid_cmf, i, validate=True)

    # Test that it doesn't fail when a single B_i is invalid and validate=False.
    for i, _ in enumerate(invalid_B_is):
        if i == 0:  # invalid B_i for i = 0
            continue
        cmf_to_matrix(invalid_cmf, i, validate=False)

    # Check that validate is called only when validate=True
    with patch("matcouply.coupled_matrices._validate_cmf", return_value=(shapes, rank)) as mock:
        coupled_matrices.cmf_to_matrix(cmf, 0, validate=False)
        mock.assert_not_called()
        coupled_matrices.cmf_to_matrix(cmf, 0, validate=True)
        mock.assert_called()


def test_cmf_to_matrices(rng, random_ragged_cmf):
    cmf, shapes, rank = random_ragged_cmf
    weights, (A, B_is, C) = cmf

    # Compare matcouply implementation with our own implementation
    matrices = cmf.to_matrices()
    for i, matrix in enumerate(matrices):
        manually_assembled_matrix = (weights * A[i] * B_is[i]) @ C.T
        assert_allclose(matrix, manually_assembled_matrix)

    invalid_B_is = copy(B_is)
    invalid_B_is[0] = tl.tensor(rng.random_sample((tl.shape(B_is[0])[0], tl.shape(B_is[0])[1] + 1)))
    invalid_cmf = (weights, (A, invalid_B_is, C))

    # Test that it always fails when a single B_i is invalid and validate=True
    with pytest.raises(ValueError):
        cmf_to_matrices(invalid_cmf, validate=True)

    # Check that validate is called only when validate=True
    with patch("matcouply.coupled_matrices._validate_cmf", return_value=(shapes, rank)) as mock:
        coupled_matrices.cmf_to_matrices(cmf, validate=False)
        mock.assert_not_called()
        coupled_matrices.cmf_to_matrices(cmf, validate=True)
        mock.assert_called()


def test_cmf_to_slice(rng, random_ragged_cmf):
    cmf, shapes, rank = random_ragged_cmf
    weights, (A, B_is, C) = cmf

    # Compare matcouply implementation with our own implementation
    for i, B_i in enumerate(B_is):
        matrix = cmf.to_matrix(i)
        slice_ = coupled_matrices.cmf_to_slice(cmf, i)
        assert_allclose(matrix, slice_)

    # Check that to_slice is an alias for to_matrix
    with patch("matcouply.coupled_matrices.cmf_to_matrix") as mock:
        coupled_matrices.cmf_to_slice(cmf, 0)
        mock.assert_called()


def test_cmf_to_slices(rng, random_ragged_cmf):
    cmf, shapes, rank = random_ragged_cmf

    # Compare matcouply implementation with our own implementation
    matrices = cmf.to_matrices()
    slices = coupled_matrices.cmf_to_slices(cmf)
    for slice_, matrix in zip(slices, matrices):
        assert_allclose(matrix, slice_)

    # Check that to_slices is an alias for to_matrices
    with patch("matcouply.coupled_matrices.cmf_to_matrices", return_value=matrices) as mock:
        coupled_matrices.cmf_to_slices(cmf)
        mock.assert_called()


def test_cmf_to_tensor(rng, random_regular_cmf):
    cmf, shapes, rank = random_regular_cmf
    weights, (A, B_is, C) = cmf

    # Check that the tensor slices are equal to the manually assembled matrices
    tensor = cmf.to_tensor()
    for i, matrix in enumerate(tensor):
        manually_assembled_matrix = (weights * A[i] * B_is[i]) @ C.T
        assert_allclose(matrix, manually_assembled_matrix)

    # Check that the tensor slices when the matrices have different number
    #   of rows are equal to the manually assembled matrices padded by zeros
    ragged_shapes = ((15, 10), (10, 10), (15, 10), (10, 10))
    max_length = max(length for length, _ in ragged_shapes)
    ragged_cmf = random.random_coupled_matrices(ragged_shapes, rank, random_state=rng)
    weights, (A, B_is, C) = ragged_cmf

    ragged_tensor = ragged_cmf.to_tensor()
    for i, matrix in enumerate(ragged_tensor):
        manually_assembled_matrix = (weights * A[i] * B_is[i]) @ C.T

        shape = ragged_shapes[i]
        assert_allclose(matrix[: shape[0]], manually_assembled_matrix)
        assert_allclose(matrix[shape[0] :], 0)
        assert matrix.shape[0] == max_length

    with patch("matcouply.coupled_matrices._validate_cmf", return_value=(ragged_shapes, rank)) as mock:
        coupled_matrices.cmf_to_tensor(ragged_cmf, validate=False)
        mock.assert_not_called()
        coupled_matrices.cmf_to_tensor(ragged_cmf, validate=True)
        mock.assert_called()


def test_cmf_to_unfolded(rng, random_ragged_cmf):
    cmf, shapes, rank = random_ragged_cmf

    # with padding
    tensor = cmf.to_tensor()
    for mode in range(3):
        unfolded_tensor = cmf.to_unfolded(mode)
        assert_allclose(unfolded_tensor, tl.unfold(tensor, mode))

        with patch("matcouply.coupled_matrices._validate_cmf", return_value=(shapes, rank)) as mock:
            coupled_matrices.cmf_to_unfolded(cmf, mode, validate=False)
            mock.assert_not_called()
            coupled_matrices.cmf_to_unfolded(cmf, mode, validate=True)
            mock.assert_called()

    # without padding
    with pytest.raises(ValueError):
        cmf.to_unfolded(pad=False, mode=0)
    with pytest.raises(ValueError):
        cmf.to_unfolded(pad=False, mode=1)

    matrices = cmf.to_matrices()

    unfolded_matrices = tl.transpose(tl.concatenate(matrices, axis=0))
    assert_allclose(cmf.to_unfolded(pad=False, mode=2), unfolded_matrices)
    with patch("matcouply.coupled_matrices._validate_cmf", return_value=(shapes, rank)) as mock:
        coupled_matrices.cmf_to_unfolded(cmf, 2, pad=False, validate=False)
        mock.assert_not_called()
        coupled_matrices.cmf_to_unfolded(cmf, 2, pad=False, validate=True)
        mock.assert_called()


def test_cmf_to_vec(rng, random_ragged_cmf):
    cmf, shapes, rank = random_ragged_cmf

    # With zero padding
    tensor = cmf.to_tensor()
    vector = tl.reshape(tensor, (-1,))
    assert_allclose(cmf.to_vec(), vector)

    # Without zero padding TODO:CHECK
    matrices = cmf.to_matrices()
    assert_allclose(cmf.to_vec(pad=False), tl.concatenate([tl.reshape(matrix, (-1,)) for matrix in matrices]))

    # Test that validate is called when it should be
    with patch("matcouply.coupled_matrices._validate_cmf", return_value=(shapes, rank)) as mock:
        coupled_matrices.cmf_to_vec(cmf, validate=False)
        mock.assert_not_called()
        coupled_matrices.cmf_to_vec(cmf, validate=True)
        mock.assert_called()


def test_from_CPTensor_with_shapes(rng):
    A = tl.tensor(rng.standard_normal(size=(5, 4)))
    B = tl.tensor(rng.standard_normal(size=(10, 4)))
    C = tl.tensor(rng.standard_normal(size=(15, 4)))

    # Check that we get decomposition with the correct shape
    cp_tensor = tl.cp_tensor.CPTensor((None, (A, B, C)))
    full_shapes = ((10, 15), (10, 15), (10, 15), (10, 15), (10, 15))
    cmf_full = coupled_matrices.CoupledMatrixFactorization.from_CPTensor(cp_tensor, shapes=full_shapes)
    assert full_shapes == cmf_full.shape

    ragged_shapes = ((9, 15), (10, 15), (8, 15), (10, 15), (10, 15))
    cmf_ragged = coupled_matrices.CoupledMatrixFactorization.from_CPTensor(cp_tensor, shapes=ragged_shapes)
    assert ragged_shapes == cmf_ragged.shape

    # Check that invalid shapes yields valueerror
    shapes_invalid_I = ((10, 15), (10, 15), (10, 15), (10, 15), (10, 15), (10, 15))
    with pytest.raises(ValueError):
        coupled_matrices.CoupledMatrixFactorization.from_CPTensor(cp_tensor, shapes=shapes_invalid_I)
    shapes_invalid_J = ((10, 15), (10, 15), (10, 15), (11, 15), (10, 15))
    with pytest.raises(ValueError):
        coupled_matrices.CoupledMatrixFactorization.from_CPTensor(cp_tensor, shapes=shapes_invalid_J)
    shapes_invalid_K = ((10, 16), (10, 16), (10, 16), (10, 16), (10, 16))
    with pytest.raises(ValueError):
        coupled_matrices.CoupledMatrixFactorization.from_CPTensor(cp_tensor, shapes=shapes_invalid_K)
