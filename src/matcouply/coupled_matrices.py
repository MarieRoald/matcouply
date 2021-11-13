import tensorly as tl
from tensorly._factorized_tensor import FactorizedTensor


class CoupledMatrixFactorization(FactorizedTensor):
    """Class wrapper for coupled matrix factorizations.
    """

    # TODO: Unit test
    def __init__(self, cmf_matrices):
        super().__init__()

        shape, rank = _validate_cmf(cmf_matrices)
        self.weights, self.factors = cmf_matrices

        self.shape = shape
        self.rank = rank

    @classmethod
    def from_CPTensor(cls, cp_tensor):
        """Convert a CP tensor into a coupled matrix factorization.

        Arguments
        ---------
        cp_tensor : tl.cp_tensor.CPTensor
            CP tensor to convert into a coupled matrix factorization

        Returns
        -------
        CoupledMatrixFactorization
            A coupled matrix factorization that represents the same tensor
            as ``cp_tensor``.
        """
        cp_tensor = tl.cp_tensor.CPTensor(cp_tensor)
        weights, factors = cp_tensor
        if len(factors) != 3:
            raise ValueError("Must be a third order CP tensor to convert into a coupled matrix factorization")
        A, B, C = factors
        B_is = [tl.copy(B) for i in range(tl.shape(A)[0])]
        return cls((tl.copy(weights), [tl.copy(A), B_is, tl.copy(C)]))

    @classmethod
    def from_Parafac2Tensor(cls, parafac2_tensor):
        """Convert a PARAFAC2 tensor into a coupled matrix factorization.

        Arguments
        ---------
        parafac2_tensor : tl.parafac2_tensor.Parafac2Tensor
            PARAFAC2 tensor to convert into a coupled matrix factorization

        Returns
        -------
        CoupledMatrixFactorization
            A coupled matrix factorization that represents the same tensor
            as ``parafac2_tensor``.
        """
        parafac2_tensor = tl.parafac2_tensor.Parafac2Tensor(parafac2_tensor)
        weights, factors, projection_matrices = parafac2_tensor
        A, B, C = factors
        B_is = [tl.dot(Pi, B) for Pi in projection_matrices]
        return cls((tl.copy(weights), [tl.copy(A), B_is, tl.copy(C)]))

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

    def __repr__(self):  # pragma: nocover
        message = "(weights, factors) : rank-{} CoupledMatrixFactorization of shape {} ".format(self.rank, self.shape)
        return message

    def to_tensor(self):
        """Convert to a dense tensor (pad uneven slices by zeros). See ``cmf_to_tensor``.
        """
        return cmf_to_tensor(self)

    def to_vec(self):  # TODO: Add flag to not pad by zeros
        """Convert to a vector by first converting to a dense tensor and unraveling. See ``cmf_to_vec``
        """
        return cmf_to_vec(self)

    def to_unfolded(self, mode):
        """Convert to a matrix by first converting to a dense tensor and unfolding. See ``cmf_to_unfolded``
        """
        return cmf_to_unfolded(self, mode)

    def to_matrices(self):
        """Convert to a list of matrices. See ``cmf_to_matrices``
        """
        return cmf_to_matrices(self)

    def to_matrix(self, matrix_idx):
        """Construct a single dense matrix from the decomposition. See ``cmf_to_matrix``
        """
        return cmf_to_matrix(self, matrix_idx)


def _validate_cmf(cmf):
    # TODO: docstring
    weights, (A, B_is, C) = cmf
    if not (tl.is_tensor(weights) or weights is None):
        raise TypeError("Weights should be a first order tensor of length rank, not {}".format(type(weights)))
    elif weights is not None and len(tl.shape(weights)) != 1:
        raise ValueError(
            "Weights should be a first order tensor. However weights has shape {}".format(tl.shape(weights))
        )
    if not tl.is_tensor(A):
        raise TypeError(
            "The first factor matrix, A, should be a second order tensor of size (I, rank)), not {}".format(type(A))
        )
    elif len(tl.shape(A)) != 2:
        raise ValueError(
            "The first factor matrix, A, should be a second order tensor. However A has shape {}".format(tl.shape(A))
        )
    if not tl.is_tensor(C):
        raise TypeError(
            "The last factor matrix, C, should be a second order tensor of size (K, rank)), not {}".format(type(C))
        )
    elif len(tl.shape(C)) != 2:
        raise ValueError(
            "The last factor matrix, C, should be a second order tensor. However C has shape {}".format(tl.shape(C))
        )
    rank = int(tl.shape(A)[1])
    if tl.shape(C)[1] != rank:
        raise ValueError(
            "All the factors of a coupled matrix factorization should have the same number of columns."
            "However, A.shape[1]={} but C.shape[1]={}.".format(rank, tl.shape(C)[1])
        )

    shape = []
    for i, B_i in enumerate(B_is):
        if not tl.is_tensor(B_i):
            raise TypeError(
                "The B_is[{}] factor matrix should be second order tensor of size (J_i, rank)), not {}".format(
                    i, type(B_i)
                )
            )
        elif len(tl.shape(B_i)) != 2:
            raise ValueError(
                "The B_is[{}] factor matrix should be second order tensor. However B_is[{}] has shape {}".format(
                    i, i, tl.shape(B_i)
                )
            )
        if tl.shape(B_i)[1] != rank:
            raise ValueError(
                "All the factors of a coupled matrix factorization should have the same number of columns."
                "However, A.shape[1]={} but B_is[{}].shape[1]={}.".format(rank, i, tl.shape(B_i)[1])
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

    The decomposition is on the form :math:`(\mathbf{A} [\mathbf{B}_i] \mathbf{C})`
    such that the i-th matrix, :math:`X_i` is given by

    .. math::

        X_i = \mathbf{B}_i \text{diag}(\mathbf{a}_i) \mathbf{C}^T,

    where :math:`\text{diag}(a_i)` is the diagonal matrix whose nonzero entries are equal to
    the :math:`i`-th row of the :math:`I \times R`factor matrix :math:`\mathbf{A}`,
    :math:`\mathbf{B}_i` is a :math:`J_i \times R` factor matrix, and :math:`\mathbf{C}`
    is a :math:`K \times R` factor matrix.

    Parameters
    ----------

    cmf : CoupledMatrixFactorization - (weight, factors)

        * weights : 1D array of shape (rank, )
            weights of the factors
        * factors : List of factors of the coupled matrix decomposition
            Containts the matrices :math:`\mathbf{A}`, :math:`\mathbf{B}_i` and
            :math:`\mathbf{C}` described above

    matrix_idx : int
        Index of the matrix we want to construct, :math:`i` in the equations above.

    validate : bool
        If true, then the decomposition is validated before the matrix is constructed
        (see ``CoupledMatrixFactorization``).

    Returns
    -------
    ndarray
        Dense tensor of shape ``[B_is[matrix_idx].shape[0], C.shape[0]]``, where
        ``B`` is a list containing all the :math:`\mathbf{B}_i`-factor matrices.
    """
    # TODO: Example
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
    """Alias for ``cmf_to_matrix``.
    """
    return cmf_to_matrix(cmf, slice_idx, validate=validate)


def cmf_to_matrices(cmf, validate=True):
    r"""Generate a list of all matrices represented by the coupled matrix factorisation.

    The decomposition is on the form :math:`(\mathbf{A} [\mathbf{B}_i] \mathbf{C})`
    such that the i-th matrix, :math:`X_i` is given by

    .. math::

        X_i = \mathbf{B}_i \text{diag}(\mathbf{a}_i) \mathbf{C}^T,

    where :math:`\text{diag}(a_i)` is the diagonal matrix whose nonzero entries are equal to
    the :math:`i`-th row of the :math:`I \times R`factor matrix :math:`\mathbf{A}`,
    :math:`\mathbf{B}_i` is a :math:`J_i \times R` factor matrix, and :math:`\mathbf{C}`
    is a :math:`K \times R` factor matrix.

    Parameters
    ----------

    cmf: CoupledMatrixFactorization - (weight, factors)

        * weights : 1D array of shape (rank, )
            weights of the factors
        * factors : List of factors of the coupled matrix decomposition
            Containts the matrices :math:`\mathbf{A}`, :math:`\mathbf{B}_i` and
            :math:`\mathbf{C}` described above

    validate : bool
        If true, then the decomposition is validated before the matrix is constructed
        (see ``CoupledMatrixFactorization``).

    Returns
    -------
    List of ndarray
        List of all :math:`\mathbf{X}_i`-matrices, where the ``i``-th element of the list
        has shape ``[B_is[i].shape[0], C.shape[0]]``, where ``B_is`` is a list containing all
        the :math:`\mathbf{B}_i`-factor matrices.
    """
    # TODO: Example
    # Construct matrices and return list
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
    """Alias for ``cmf_to_matrices``.
    """
    return cmf_to_matrices(cmf, validate=validate)


def cmf_to_tensor(cmf, validate=True):
    r"""Generate the tensor represented by the coupled matrix factorization.

    If all :math:`\mathbf{B}_i`-factor matrices have the same number of rows, then this
    function returnes a tensorized version of ``cmf_to_matrices``. Otherwise, each
    matrix is padded by zeros to have the same number of rows before forming the tensor.

    The decomposition is on the form :math:`(\mathbf{A} [\mathbf{B}_i] \mathbf{C})`
    such that the i-th matrix, :math:`X_i` is given by

    .. math::

        X_i = \mathbf{B}_i \text{diag}(\mathbf{a}_i) \mathbf{C}^T,

    where :math:`\text{diag}(a_i)` is the diagonal matrix whose nonzero entries are equal to
    the :math:`i`-th row of the :math:`I \times R`factor matrix :math:`\mathbf{A}`,
    :math:`\mathbf{B}_i` is a :math:`J_i \times R` factor matrix, and :math:`\mathbf{C}`
    is a :math:`K \times R` factor matrix.

    Parameters
    ----------

    cmf: CoupledMatrixFactorization - (weight, factors)

        * weights : 1D array of shape (rank, )
            weights of the factors
        * factors : List of factors of the coupled matrix decomposition
            Containts the matrices :math:`\mathbf{A}`, :math:`\mathbf{B}_i` and
            :math:`\mathbf{C}` described above

    validate : bool
        If true, then the decomposition is validated before the matrix is constructed
        (see ``CoupledMatrixFactorization``).

    Returns
    -------
    ndarray
        Full tensor of shape ``[A.shape[0], J, C.shape[0]]``, where ``J`` is the maximum
        number of rows in all the :math:`\mathbf{B}_i`-factor matrices.
    """
    # TODO: Example

    _, (A, B_is, C) = cmf
    matrices = cmf_to_matrices(cmf, validate=validate)
    lengths = [B_i.shape[0] for B_i in B_is]

    tensor = tl.zeros((A.shape[0], max(lengths), C.shape[0]), **tl.context(matrices[0]))
    for i, (matrix_, length) in enumerate(zip(matrices, lengths)):
        tensor = tl.index_update(tensor, tl.index[i, :length], matrix_)

    return tensor


def cmf_to_unfolded(cmf, mode, validate=True):
    # TODO: docstring
    # TODO: Option to use stack of matrices instead of tensor padded with zeros
    return tl.unfold(cmf_to_tensor(cmf, validate=validate), mode)


def cmf_to_vec(cmf, validate=True):
    # TODO: docstring
    # TODO: Option to use stack of matrices instead of tensor padded with zeros
    return tl.tensor_to_vec(cmf_to_tensor(cmf, validate=validate))