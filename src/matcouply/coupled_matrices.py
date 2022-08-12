# MIT License: Copyright (c) 2022, Marie Roald.
# See the LICENSE file in the root directory for full license text.

import tensorly as tl
from tensorly._factorized_tensor import FactorizedTensor


class CoupledMatrixFactorization(FactorizedTensor):
    r"""Class wrapper for coupled matrix factorizations.

    Coupled matrix factorizations decompositions represent stacks of matrices and are on the form
    :math:`(\mathbf{A} [\mathbf{B}^{(0)}, \mathbf{B}^{(1)}, ..., \mathbf{B}^{(I-1)}] \mathbf{C})`,
    such that the i-th matrix, :math:`\mathbf{X}^{(i)}` is given by

    .. math::

        \mathbf{X}^{(i)} = \mathbf{B}^{(i)} \text{diag}(\mathbf{a}_i) \mathbf{C}^T,

    where :math:`\text{diag}(\mathbf{a}_i)` is the diagonal matrix whose nonzero entries are
    equal to the :math:`i`-th row of the :math:`I \times R` factor matrix :math:`\mathbf{A}`,
    :math:`\mathbf{B}^{(i)}` is a :math:`J_i \times R` factor matrix, and :math:`\mathbf{C}`
    is a :math:`K \times R` factor matrix. For more information about coupled matrix decompositions,
    see :doc:`../coupled_matrix_factorization`.

    This class validates the decomposition and provides conversion to dense formats via methods.

    Parameters
    ----------

    cmf: CoupledMatrixFactorization - (weights, factors)
        Coupled matrix factorization represented by weights and factors as described in :doc:`../coupled_matrix_factorization`.

        * weights : 1D array of shape (rank,) or None
            weights of the factors
        * factors : List of factors of the coupled matrix decomposition
            List on the form ``[A, [B_0, B_1, ..., B_i], C]``, where ``A`` represents :math:`\mathbf{A}`,
            ``[B_0, B_1, ..., B_i]`` represents a list of all :math:`\mathbf{B}^{(i)}`-matrices and ``C``
            represents :math:`\mathbf{C}`

    Examples
    --------
    >>> from tensorly.random import random_tensor
    >>> from matcouply.coupled_matrices import CoupledMatrixFactorization
    >>> A = random_tensor((5, 3))
    >>> B_is = [random_tensor((10, 3)) for i in range(5)]
    >>> C = random_tensor((15, 3))
    >>> cmf = CoupledMatrixFactorization((None, (A, B_is, C)))

    We can then convert the factorization to a dense format easily

    >>> matrices = cmf.to_matrices()
    >>> len(matrices)
    5

    >>> tl.shape(matrices[0])
    (10, 15)

    We see that we can get the shape of the decomposition without
    converting it to a dense format first also.

    >>> len(cmf.shape)
    5

    >>> cmf.shape[0]
    (10, 15)

    We can also extract the weights and factor matrices from the decomposition
    object as if it was a tuple.

    >>> weights, (A, B_is, C) = cmf

    And if the decomposition is invalid, then a helpful error message will be
    printed.

    >>> A = random_tensor((5, 3))
    >>> B_is = [random_tensor((10, 3)) for i in range(5)]
    >>> C = random_tensor((15, 15, 3))
    >>> cmf = CoupledMatrixFactorization((None, (A, B_is, C)))
    Traceback (most recent call last):
      ...
    ValueError: The last factor matrix, C, should be a second order tensor. However C has shape (15, 15, 3)

    >>> A = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    >>> B_is = [random_tensor((10, 3)) for i in range(5)]
    >>> C = random_tensor((15, 3))
    >>> cmf = CoupledMatrixFactorization((None, (A, B_is, C)))
    Traceback (most recent call last):
      ...
    TypeError: The first factor matrix, A, should be a second order tensor of size (I, rank)), not <class 'list'>
    """

    def __init__(self, cmf_matrices):
        super().__init__()

        shape, rank = _validate_cmf(cmf_matrices)
        self.weights, self.factors = cmf_matrices

        self.shape = shape
        self.rank = rank

    @classmethod
    def from_CPTensor(cls, cp_tensor, shapes=None):
        """Convert a CP tensor into a coupled matrix factorization.

        Parameters
        ----------
        cp_tensor : tl.cp_tensor.CPTensor
            CP tensor to convert into a coupled matrix factorization

        Returns
        -------
        CoupledMatrixFactorization
            A coupled matrix factorization that represents the same tensor
            as ``cp_tensor``.

        Raises
        ------
        ValueError
            If the CP tensor has more than tree modes.
        """
        cp_tensor = tl.cp_tensor.CPTensor(cp_tensor)
        weights, factors = cp_tensor
        if len(factors) != 3:
            raise ValueError("Must be a third order CP tensor to convert into a coupled matrix factorization")
        A, B, C = factors

        if shapes is not None:
            B_is = []
            if len(shapes) != tl.shape(A)[0]:
                raise ValueError(
                    f"The first mode has length {tl.shape(A)[0]}, which is different "
                    f"than the length indicated by the shapes argument ({len(shapes)})"
                )
            for i, shape in enumerate(shapes):
                J_i, K = shape
                if K != tl.shape(C)[0]:
                    raise ValueError(
                        f"The third mode has length {tl.shape(C)[0]}, which is different "
                        f"than the length indicated by the shapes argument ({K})"
                    )
                if J_i > tl.shape(B)[0]:
                    raise ValueError(
                        f"The second mode of the CP tensor mode has length {tl.shape(B)[0]}, which "
                        f"is smaller than the length indicated by the shape ({J_i}) of matrix"
                    )

                B_is.append(tl.copy(B)[:J_i, :])
        else:
            B_is = [tl.copy(B) for i in range(tl.shape(A)[0])]

        return cls((tl.copy(weights), [tl.copy(A), B_is, tl.copy(C)]))

    @classmethod
    def from_Parafac2Tensor(cls, parafac2_tensor):
        """Convert a PARAFAC2 tensor into a coupled matrix factorization.

        Parameters
        ----------
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
        message = "(weights, factors) : rank-{} CoupledMatrixFactorization of shape {}".format(self.rank, self.shape)
        return message

    def to_tensor(self):
        """Convert to a dense tensor (pad uneven slices by zeros).

        See also
        --------
        cmf_to_tensor
        """
        return cmf_to_tensor(self)

    def to_vec(self, pad=True):
        """Convert to a vector by first converting to a dense tensor and unraveling.

        See also
        --------
        cmf_to_vec
        """
        return cmf_to_vec(self, pad=pad)

    def to_unfolded(self, mode, pad=True):
        """Convert to a matrix by first converting to a dense tensor and unfolding.

        See also
        --------
        cmf_to_unfolded
        """
        return cmf_to_unfolded(self, mode, pad=pad)

    def to_matrices(self):
        """Convert to a list of matrices.

        See also
        --------
        cmf_to_matrices
        """
        return cmf_to_matrices(self)

    def to_matrix(self, matrix_idx):
        """Construct a single dense matrix from the decomposition.

        See also
        --------
        cmf_to_matrix
        """
        return cmf_to_matrix(self, matrix_idx)


def _validate_cmf(cmf):
    r"""Check that the coupled matrix factorization is valid and return the shapes and its rank.

    Parameters
    ----------
    cmf: CoupledMatrixFactorization - (weights, factors)
        Coupled matrix factorization represented by weights and factors as described in :doc:`../coupled_matrix_factorization`.

        * weights : 1D array of shape (rank,) or None
            weights of the factors
        * factors : List of factors of the coupled matrix decomposition
            List on the form ``[A, [B_0, B_1, ..., B_i], C]``, where ``A`` represents :math:`\mathbf{A}`,
            ``[B_0, B_1, ..., B_i]`` represents a list of all :math:`\mathbf{B}^{(i)}`-matrices and ``C``
            represents :math:`\mathbf{C}`

    Returns
    -------
    shapes : tuple of tuple of ints
        A tuple containing the shape of each matrix represented by the
        coupled matrix factorization
    rank : int
        The rank of the factorization

    Examples
    --------
    The ``_validate_cmf`` function returns the shapes and rank of any
    valid coupled matrix factorization

    >>> from tensorly.random import random_tensor
    >>> from matcouply.coupled_matrices import _validate_cmf
    >>> A = random_tensor((5, 3))
    >>> B_is = [random_tensor((10, 3)) for i in range(5)]
    >>> C = random_tensor((15, 3))
    >>> _validate_cmf((None, (A, B_is, C)))
    (((10, 15), (10, 15), (10, 15), (10, 15), (10, 15)), 3)

    And if the decomposition is invalid, then a helpful error message will be
    printed.

    >>> A = random_tensor((5, 3))
    >>> B_is = [random_tensor((10, 3)) for i in range(5)]
    >>> C = random_tensor((15, 15, 3))
    >>> _validate_cmf((None, (A, B_is, C)))
    Traceback (most recent call last):
      ...
    ValueError: The last factor matrix, C, should be a second order tensor. However C has shape (15, 15, 3)

    >>> A = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    >>> B_is = [random_tensor((10, 3)) for i in range(5)]
    >>> C = random_tensor((15, 3))
    >>> _validate_cmf((None, (A, B_is, C)))
    Traceback (most recent call last):
      ...
    TypeError: The first factor matrix, A, should be a second order tensor of size (I, rank)), not <class 'list'>
    """
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

    if tl.shape(A)[0] != len(B_is):
        raise ValueError(
            "The number of rows in A should be the same as the number of B_i matrices"
            "However, tl.shape(A)[0]={}, but len(B_is)={}".format(tl.shape(A)[0], len(B_is))
        )

    return tuple(shape), rank


def cmf_to_matrix(cmf, matrix_idx, validate=True):
    r"""Generate a single matrix from the coupled matrix factorisation.

    The decomposition is on the form :math:`(\mathbf{A} [\mathbf{B}^{(0)}, \mathbf{B}^{(1)}, ..., \mathbf{B}^{(I-1)}] \mathbf{C})`
    such that the i-th matrix, :math:`\mathbf{X}^{(i)}` is given by

    .. math::

        \mathbf{X}^{(i)} = \mathbf{B}^{(i)} \text{diag}(\mathbf{a}_i) \mathbf{C}^T,

    where :math:`\text{diag}(\mathbf{a}_i)` is the diagonal matrix whose nonzero entries are
    equal to the :math:`i`-th row of the :math:`I \times R` factor matrix :math:`\mathbf{A}`,
    :math:`\mathbf{B}^{(i)}` is a :math:`J_i \times R` factor matrix, and :math:`\mathbf{C}`
    is a :math:`K \times R` factor matrix.

    Parameters
    ----------

    cmf: CoupledMatrixFactorization - (weights, factors)
        Coupled matrix factorization represented by weights and factors as described in :doc:`../coupled_matrix_factorization`.

        * weights : 1D array of shape (rank,) or None
            weights of the factors
        * factors : List of factors of the coupled matrix decomposition
            List on the form ``[A, [B_0, B_1, ..., B_i], C]``, where ``A`` represents :math:`\mathbf{A}`,
            ``[B_0, B_1, ..., B_i]`` represents a list of all :math:`\mathbf{B}^{(i)}`-matrices and ``C``
            represents :math:`\mathbf{C}`

    matrix_idx : int
        Index of the matrix we want to construct, :math:`i` in the equations above.

    validate : bool
        If true, then the decomposition is validated before the matrix is constructed
        (see ``CoupledMatrixFactorization``).

    Returns
    -------
    ndarray
        Dense tensor of shape ``[B_is[matrix_idx].shape[0], C.shape[0]]``, where
        ``B`` is a list containing all the :math:`\mathbf{B}^{(i)}`-factor matrices.

    Examples
    --------
    An example where we calculate one of the matrices described by
    a coupled matrix factorization

    >>> from matcouply.random import random_coupled_matrices
    >>> from matcouply.coupled_matrices import cmf_to_matrix
    >>> shapes = ((5, 10), (6, 10), (7, 10))
    >>> cmf = random_coupled_matrices(shapes, rank=3)
    >>> matrix = cmf_to_matrix(cmf, matrix_idx=1)
    >>> tl.shape(matrix)
    (6, 10)
    """
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

    See also
    --------
    cmf_to_matrix
    """
    return cmf_to_matrix(cmf, slice_idx, validate=validate)


def cmf_to_matrices(cmf, validate=True):
    r"""Generate a list of all matrices represented by the coupled matrix factorisation.

    The decomposition is on the form :math:`(\mathbf{A} [\mathbf{B}^{(0)}, \mathbf{B}^{(1)}, ..., \mathbf{B}^{(I-1)}] \mathbf{C})`
    such that the i-th matrix, :math:`\mathbf{X}^{(i)}` is given by

    .. math::

        \mathbf{X}^{(i)} = \mathbf{B}^{(i)} \text{diag}(\mathbf{a}_i) \mathbf{C}^T,

    where :math:`\text{diag}(\mathbf{a}_i)` is the diagonal matrix whose nonzero entries are
    equal to the :math:`i`-th row of the :math:`I \times R` factor matrix :math:`\mathbf{A}`,
    :math:`\mathbf{B}^{(i)}` is a :math:`J_i \times R` factor matrix, and :math:`\mathbf{C}`
    is a :math:`K \times R` factor matrix.

    Parameters
    ----------
    cmf: CoupledMatrixFactorization - (weights, factors)
        Coupled matrix factorization represented by weights and factors as described in :doc:`../coupled_matrix_factorization`.

        * weights : 1D array of shape (rank,) or None
            weights of the factors
        * factors : List of factors of the coupled matrix decomposition
            List on the form ``[A, [B_0, B_1, ..., B_i], C]``, where ``A`` represents :math:`\mathbf{A}`,
            ``[B_0, B_1, ..., B_i]`` represents a list of all :math:`\mathbf{B}^{(i)}`-matrices and ``C``
            represents :math:`\mathbf{C}`

    validate : bool
        If true, then the decomposition is validated before the matrix is constructed
        (see ``CoupledMatrixFactorization``).

    Returns
    -------
    List of ndarray
        List of all :math:`\mathbf{X}^{(i)}`-matrices, where the ``i``-th element of the list
        has shape ``[B_is[i].shape[0], C.shape[0]]``, where ``B_is`` is a list containing all
        the :math:`\mathbf{B}^{(i)}`-factor matrices.

    Examples
    --------
    We can convert a coupled matrix factorization to a list of matrices

    >>> from matcouply.random import random_coupled_matrices
    >>> from matcouply.coupled_matrices import cmf_to_matrix
    >>> shapes = ((5, 10), (6, 10), (7, 10))
    >>> cmf = random_coupled_matrices(shapes, rank=3)
    >>> matrices = cmf_to_matrices(cmf)
    >>> for matrix in matrices:
    ...    print(tl.shape(matrix))
    (5, 10)
    (6, 10)
    (7, 10)
    """
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

    See also
    --------
    cmf_to_matrices
    """
    return cmf_to_matrices(cmf, validate=validate)


def cmf_to_tensor(cmf, validate=True):
    r"""Generate the tensor represented by the coupled matrix factorization.

    If all :math:`\mathbf{B}^{(i)}`-factor matrices have the same number of rows, then this
    function returnes a tensorized version of ``cmf_to_matrices``. Otherwise, each
    matrix is padded by zeros to have the same number of rows before forming the tensor.

    The decomposition is on the form :math:`(\mathbf{A} [\mathbf{B}^{(0)}, \mathbf{B}^{(1)}, ..., \mathbf{B}^{(I-1)}] \mathbf{C})`
    such that the i-th matrix, :math:`\mathbf{X}^{(i)}` is given by

    .. math::

        \mathbf{X}^{(i)} = \mathbf{B}^{(i)} \text{diag}(\mathbf{a}_i) \mathbf{C}^T,

    where :math:`\text{diag}(\mathbf{a}_i)` is the diagonal matrix whose nonzero entries are
    equal to the :math:`i`-th row of the :math:`I \times R` factor matrix :math:`\mathbf{A}`,
    :math:`\mathbf{B}^{(i)}` is a :math:`J_i \times R` factor matrix, and :math:`\mathbf{C}`
    is a :math:`K \times R` factor matrix.

    Parameters
    ----------
    cmf: CoupledMatrixFactorization - (weights, factors)
        Coupled matrix factorization represented by weights and factors as described in :doc:`../coupled_matrix_factorization`.

        * weights : 1D array of shape (rank,) or None
            weights of the factors
        * factors : List of factors of the coupled matrix decomposition
            List on the form ``[A, [B_0, B_1, ..., B_i], C]``, where ``A`` represents :math:`\mathbf{A}`,
            ``[B_0, B_1, ..., B_i]`` represents a list of all :math:`\mathbf{B}^{(i)}`-matrices and ``C``
            represents :math:`\mathbf{C}`

    validate : bool
        If true, then the decomposition is validated before the matrix is constructed
        (see ``CoupledMatrixFactorization``).

    Returns
    -------
    ndarray
        Full tensor of shape ``[A.shape[0], J, C.shape[0]]``, where ``J`` is the maximum
        number of rows in all the :math:`\mathbf{B}^{(i)}`-factor matrices.

    Examples
    --------
    We can convert a coupled matrix factorization to a tensor. This will be equivalent
    to stacking the matrices using axis=0.

    >>> from matcouply.random import random_coupled_matrices
    >>> from matcouply.coupled_matrices import cmf_to_tensor
    >>> shapes = ((5, 10), (5, 10), (5, 10), (5, 10))
    >>> cmf = random_coupled_matrices(shapes, rank=3)
    >>> tensor = cmf_to_tensor(cmf)
    >>> tl.shape(tensor)
    (4, 5, 10)

    We can also convert a coupled matrix factorization that represent matrices with different
    numbers of columns. Then, the smaller matrices will be padded by zeros so all have the same
    shape.

    >>> shapes = ((5, 10), (5, 10), (5, 10), (3, 10))
    >>> cmf = random_coupled_matrices(shapes, rank=3)
    >>> tensor = cmf_to_tensor(cmf)
    >>> tl.shape(tensor)
    (4, 5, 10)

    It is only the last matrix which has a different shape. It has two fewer rows than the rest,
    which means that it has 20 zero-valued elements (:math:`20 = 2 \times 10`).

    >>> num_zeros = (tensor == 0).sum()
    >>> num_zeros
    20
    """
    _, (A, B_is, C) = cmf
    matrices = cmf_to_matrices(cmf, validate=validate)
    lengths = [B_i.shape[0] for B_i in B_is]

    tensor = tl.zeros((A.shape[0], max(lengths), C.shape[0]), **tl.context(matrices[0]))
    for i, (matrix_, length) in enumerate(zip(matrices, lengths)):
        tensor = tl.index_update(tensor, tl.index[i, :length], matrix_)

    return tensor


def cmf_to_unfolded(cmf, mode, pad=True, validate=True):
    r"""Generate the unfolded tensor represented by the coupled matrix factorization.


    By default the function is an alias for first constructing a tensor (``cmf_to_tensor``) and
    then vectorizing that tensor. Note that if the matrices have a different number of
    rows, then they will be padded when the tensor is constructed, and thus, there will
    be zeros in the unfolded tensor too.

    If the zero-padding is unwanted, then setting the ``pad`` parameter to ``False`` (only available for
    ``mode=2``) will instead construct each matrix and concatenate them.

    Parameters
    ----------

    cmf: CoupledMatrixFactorization - (weights, factors)
        Coupled matrix factorization represented by weights and factors as described in :doc:`../coupled_matrix_factorization`.

        * weights : 1D array of shape (rank,) or None
            weights of the factors
        * factors : List of factors of the coupled matrix decomposition
            List on the form ``[A, [B_0, B_1, ..., B_i], C]``, where ``A`` represents :math:`\mathbf{A}`,
            ``[B_0, B_1, ..., B_i]`` represents a list of all :math:`\mathbf{B}^{(i)}`-matrices and ``C``
            represents :math:`\mathbf{C}`

    pad : bool (default=True)
        If true, then the coupled matrix factorization will be converted into a dense tensor,
        padding the matrices with zeros so all have the same size, and then unfolded. Can only
        be ``False`` if ``mode=2``.

    validate : bool (default=True)
        If true, then the decomposition is validated before the matrix is constructed
        (see ``CoupledMatrixFactorization``).

    Returns
    -------
    ndarray
        Matrix of an appropriate shape (see ``cmf_to_tensor`` and ``tensorly.unfold``).

    Raises
    ------
    ValueError
        If ``pad=False`` and ``mode!=2``.

    Examples
    --------
    Here, we show how to unfold a coupled matrix factorization along a given mode.

    >>> from matcouply.random import random_coupled_matrices
    >>> from matcouply.coupled_matrices import cmf_to_tensor
    >>> shapes = ((5, 10), (5, 10), (5, 10), (5, 10))
    >>> cmf = random_coupled_matrices(shapes, rank=3)
    >>> matrix_0 = cmf_to_unfolded(cmf, mode=0)
    >>> tl.shape(matrix_0)
    (4, 50)

    We can also unfold the tensor using ``mode=1``

    >>> matrix_1 = cmf_to_unfolded(cmf, mode=1)
    >>> tl.shape(matrix_1)
    (5, 40)

    And using ``mode=2``

    >>> matrix_2 = cmf_to_unfolded(cmf, mode=2)
    >>> tl.shape(matrix_2)
    (10, 20)

    We can also unfold a coupled matrix factorization where the matrices
    have a varying number of rows

    >>> shapes = ((5, 10), (3, 10), (2, 10), (4, 10))
    >>> cmf = random_coupled_matrices(shapes, rank=3)
    >>> matrix_0 = cmf_to_unfolded(cmf, mode=0)
    >>> tl.shape(matrix_0)
    (4, 50)

    However, as we see, the shape of the unfolded tensor is still (4, 50)
    despite some matrices being shorter. This is because the matrices are
    padded by zeros to construct the tensor, which is subsequently unfolded.

    This padding happens independently of the unfolding mode.

    >>> matrix_1 = cmf_to_unfolded(cmf, mode=1)
    >>> tl.shape(matrix_1)
    (5, 40)

    >>> matrix_2 = cmf_to_unfolded(cmf, mode=2)
    >>> tl.shape(matrix_2)
    (10, 20)

    We can see the padding by counting the number of zeros. The number of
    zeros should be :math:`((5 - 5) + (5 - 3) + (5 - 2) + (5 - 4)) \times 10 = 60`.

    >>> nonzeros_0 = tl.sum(matrix_0 == 0)
    >>> nonzeros_1 = tl.sum(matrix_1 == 0)
    >>> nonzeros_2 = tl.sum(matrix_2 == 0)
    >>> nonzeros_0, nonzeros_1, nonzeros_2
    (60, 60, 60)

    If we want to unfold with ``mode=2`` without padding with zeros, then we can use the ``pad`` argument

    >>> shapes = ((5, 10), (3, 10), (2, 10), (4, 10))
    >>> cmf = random_coupled_matrices(shapes, rank=3)
    >>> matrix_3 = cmf_to_unfolded(cmf, pad=False, mode=2)
    >>> tl.shape(matrix_3)
    (10, 14)

    """
    if pad:
        return tl.unfold(cmf_to_tensor(cmf, validate=validate), mode)
    else:
        if mode == 2:
            return tl.transpose(tl.concatenate(cmf_to_matrices(cmf, validate=validate), axis=0))
        else:
            raise ValueError(f"Cannot unfold along mode {mode} without padding. ")


def cmf_to_vec(cmf, pad=True, validate=True):
    r"""Generate the vectorized tensor represented by the coupled matrix factorization.

    By default the function is an alias for first constructing a tensor (``cmf_to_tensor``) and
    then vectorizing that tensor. Note that if the matrices have a different number of
    rows, then they will be padded when the tensor is constructed, and thus, there will
    be zeros in the vectorized tensor too.

    If the zero-padding is unwanted, then setting the ``pad`` parameter to ``False`` will instead construct and
    vectorize each matrix described by the decomposition and then concatenate these vectors forming
    one vector with no padded zero values.

    Parameters
    ----------

    cmf: CoupledMatrixFactorization - (weights, factors)
        Coupled matrix factorization represented by weights and factors as described in :doc:`../coupled_matrix_factorization`.

        * weights : 1D array of shape (rank,) or None
            weights of the factors
        * factors : List of factors of the coupled matrix decomposition
            List on the form ``[A, [B_0, B_1, ..., B_i], C]``, where ``A`` represents :math:`\mathbf{A}`,
            ``[B_0, B_1, ..., B_i]`` represents a list of all :math:`\mathbf{B}^{(i)}`-matrices and ``C``
            represents :math:`\mathbf{C}`

    pad: bool (default=True)
        If true then if the matrices described by the decomposition have a different number of rows,
        then they will be padded by zeros to construct a tensor which are vectorized, and there will
        be zeros in the vectorized tensor too. If false, the matrices will not be padded.
    validate : bool (default=True)
        If true, then the decomposition is validated before the matrix is constructed
        (see ``CoupledMatrixFactorization``).

    Returns
    -------
    ndarray
        Vector of length ``A.shape[0] * J * C.shape[0]``, where ``J`` is the maximum
        number of rows in all the :math:`\mathbf{B}^{(i)}`-factor matrices.

    Examples
    --------
    Here, we show how to vectorize a coupled matrix factorization

    >>> from matcouply.random import random_coupled_matrices
    >>> from matcouply.coupled_matrices import cmf_to_tensor
    >>> shapes = ((5, 10), (5, 10), (5, 10), (5, 10))
    >>> cmf = random_coupled_matrices(shapes, rank=3)
    >>> vector = cmf_to_vec(cmf)
    >>> tl.shape(vector)
    (200,)

    We can also vectorize a coupled matrix factorization where the matrices
    have a varying number of rows

    >>> shapes = ((5, 10), (3, 10), (2, 10), (4, 10))
    >>> cmf = random_coupled_matrices(shapes, rank=3)
    >>> vector = cmf_to_vec(cmf)
    >>> tl.shape(vector)
    (200,)

    However, as we see, the length of the vectorized coupled matrix factorization
    is still 200, despite some matrices being shorter. This is because the matrices
    are padded by zeros to construct a tensor, which is subsequently unfolded.

    We can see the padding by counting the number of zeros. The number of
    zeros should be :math:`((5 - 5) + (5 - 3) + (5 - 2) + (5 - 4)) \times 10 = 60`.

    >>> tl.sum(vector == 0)
    60

    If we want to vectorize without padding with zeros, we can use the ``pad`` argument

    >>> shapes = ((5, 10), (3, 10), (2, 10), (4, 10))
    >>> cmf = random_coupled_matrices(shapes, rank=3)
    >>> vector = cmf_to_vec(cmf, pad=False)
    >>> tl.shape(vector)
    (140,)
    """
    if pad:
        return tl.tensor_to_vec(cmf_to_tensor(cmf, validate=validate))
    else:
        matrices = cmf_to_matrices(cmf, validate=validate)
        return tl.concatenate([tl.reshape(matrix, (-1,)) for matrix in matrices])
