from tensorly._factorized_tensor import FactorizedTensor
import tensorly as tl


class CoupledMatrixFactorization(FactorizedTensor):
    # TODO: Unit test
    def __init__(self, cmf_matrices):
        super().__init__()

        shape, rank = _validate_cmf(cmf_matrices)
        self.weights, self.factors = cmf_matrices

        self.shape = shape
        self.rank = rank

    @classmethod
    def from_CPTensor(cls, cp_tensor):
        # TODO: implement creating cmf from cp
        pass

    @classmethod
    def from_Parafac2Tensor(cls, parafac2_tensor):
        # TODO: implement creating cmf from PARAFAC2
        pass

    def __getitem__(self, item):
        if item == 0:
            return self.weights
        elif item == 1:
            return self.factors
        else:
            raise IndexError(
                "You tried to access index {} of a coupled matrix factorization.\n"
                "You can only access index 0 and 1 of a coupled matrix factorization"
                "(corresponding respectively to the weights and factors)".format(item)
            )

    def __iter__(self):
        yield self.weights
        yield self.factors

    def __len__(self):
        return 2

    def __repr__(self):
        message = "(weights, factors) : rank-{} CoupledMatrixFactorization of shape {} ".format(self.rank, self.shape)
        return message

    def to_tensor(self):
        return cmf_to_tensor(self)

    def to_vec(self):
        return cmf_to_vec(self)

    def to_unfolded(self, mode):
        return cmf_to_unfolded(self, mode)

    def to_matrices(self):
        return cmf_to_matrices(self)

    def to_matrix(self, matrix_idx):
        return cmf_to_matrix(self, matrix_idx)


def _validate_cmf(cmf):
    # TODO: docstring
    # TODO: Unit test
    weights, (A, B_is, C) = cmf

    rank = int(tl.shape(A)[1])
    if tl.shape(C)[1] != rank:
        raise ValueError(
            "All the factors of a coupled matrix factorization should have the same number of columns."
            "However, A.shape[1]={} but C.shape[1]={}.".format(rank, tl.shape(C)[1])
        )

    shape = []
    for i, B_i in enumerate(B_is):
        if tl.shape(B_i)[1] != rank:
            raise ValueError(
                "All the factors of a coupled matrix factorization should have the same number of columns."
                "However, A.shape[1]={} but B_{}.shape[1]={}.".format(rank, i, tl.shape(B_i)[1])
            )
        shape.append((tl.shape(B_i)[0], tl.shape(C)[0]))

    if weights is not None and tl.shape(weights)[0] != rank:
        raise ValueError(
            "Given factors for a rank-{} coupled matrix factorization but len(weights)={}.".format(
                rank, tl.shape(weights)[0]
            )
        )

    return tuple(shape), rank


def cmf_to_matrix(cmf, matrix_idx, validate=True):
    r"""Generate a single matrix from the coupled matrix factorisation.

    The decomposition is on the form :math:`(A [B_i] C)` such that the i-th matrix,
    :math:`X_i` is given by
    .. math::

        X_i = B_i diag(a_i) C^T,

    where :math: `diag(a_i)` is the diagonal matrix whose nonzero entries are equal to
    the :math:`i`-th row of the :math:`I \times R`factor matrix :math:`A`, :math:`B_i`
    is a :math:`J_i` \times R` factor matrix, and :math:`C` is a :math: `K \times R`factor matrix.

    Parameters
    ----------

    cmf: CoupledMatrixFactorization - (weight, factors)

        * weights : 1D array of shape (rank, )
            weights of the factors
        * factors : List of factors of the coupled matrix decomposition
            Containts the matrices :math:`A`, :math:`B_i` and math:`C` described above

    Returns
    -------
    ndarray
        Full tensor of shape [B[slice_idx].shape[1], C.shape[1]], where
        B #TODO: explanation

    """
    # TODO: Unit test
    if validate:
        cmf = CoupledMatrixFactorization(cmf)
    weights, (A, B_is, C) = cmf
    a = A[matrix_idx]
    if weights is not None:
        a = a * weights

    Ct = tl.transpose(C)
    B_i = B_is[matrix_idx]
    return tl.dot(B_i * a, Ct)


def cmf_to_slice(cmf, slice_idx, validate=True):
    return cmf_to_matrix(cmf, slice_idx, validate=validate)


def cmf_to_matrices(cmf, validate=True):
    # Construct matrices and return list
    # TODO: docstring
    # TODO: B or B_i
    # TODO: Unit test
    if validate:
        cmf = CoupledMatrixFactorization(cmf)

    weights, (A, B_is, C) = cmf
    if weights is not None:
        A = A * weights
        weights = None

    decomposition = weights, (A, B_is, C)
    I, _ = A.shape
    return [cmf_to_matrix(decomposition, i, validate=False) for i in range(I)]


def cmf_to_slices(cmf, validate=True):
    return cmf_to_matrices(cmf, validate=validate)


def cmf_to_tensor(cmf, validate=True):
    # TODO: docstring
    # TODO: Unit test
    _, (A, B_is, C) = cmf
    matrices = cmf_to_matrices(cmf)
    lengths = [B_i.shape[0] for B_i in B_is]

    tensor = tl.zeros((A.shape[0], max(lengths), C.shape[0]), **tl.context(matrices[0]))
    for i, (matrix_, length) in enumerate(zip(matrices, lengths)):
        tensor = tl.index_update(tensor, tl.index[i, :length], matrix_)

    return tensor


def cmf_to_unfolded(cmf, mode, validate=True):
    # TODO: docstring
    # TODO: Unit test
    return tl.unfold(cmf_to_tensor(cmf), mode)


def cmf_to_vec(cmf, validate=True):
    # TODO: docstring
    # TODO: Unit test
    return tl.tensor_to_vec(cmf_to_tensor(cmf))
