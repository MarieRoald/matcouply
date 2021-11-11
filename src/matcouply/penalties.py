from abc import ABC, abstractmethod

import tensorly as tl
from scipy.optimize import bisect

from ._utils import get_svd

# TODO: Maybe remove compute_feasibility_gap and only use shift_aux
# TODO: Maybe rename shift_aux to subtract_from_aux
# TODO: Maybe add mixin classes for some of the functionality
# TODO: For all penalties with __init__, make sure they call super().__init__ and have aux_init and dual_init arguments
# TODO: For all penalties, add the parameters of the ADMMPenalty superclass to class docstring.


class ADMMPenalty(ABC):
    """Base class for all regularizers and constraints.

    Parameters
    ----------
    aux_init : {"random_uniform", "random_standard_normal"}
        Initialisation method for the auxiliary variables
    dual_init : {"random_uniform", "random_standard_normal"}
        Initialisation method for the auxiliary variables
    """

    def __init__(self, aux_init="random_uniform", dual_init="random_uniform"):
        self.aux_init = aux_init
        self.dual_init = dual_init

    def init_aux(self, matrices, rank, mode, random_state=None):
        random_state = tl.check_random_state(random_state)

        if not isinstance(rank, int):
            raise TypeError("Rank must be int, not {}".format(type(rank)))
        if not isinstance(mode, int):
            raise TypeError("Mode must be int, not {}".format(type(mode)))
        elif mode not in [0, 1, 2]:
            raise ValueError("Mode must be 0, 1, or 2.")

        if (
            not isinstance(self.aux_init, str)
            and not tl.is_tensor(self.aux_init)
            and not isinstance(self.aux_init, list)
        ):
            raise TypeError(
                "self.aux_init must be a tensor, a list of tensors or a string specifiying init method, not {}".format(
                    type(self.aux_init)
                )
            )
        elif self.aux_init == "random_uniform":
            if mode == 0:
                return random_state.uniform(size=(len(matrices), rank))
            elif mode == 1:
                return [random_state.uniform(size=(matrix.shape[0], rank)) for matrix in matrices]
            elif mode == 2:
                return random_state.uniform(size=(matrices[0].shape[1], rank))
        elif self.aux_init == "random_standard_normal":
            if mode == 0:
                return random_state.standard_normal(size=(len(matrices), rank))
            elif mode == 1:
                return [random_state.standard_normal(size=(matrix.shape[0], rank)) for matrix in matrices]
            elif mode == 2:
                return random_state.standard_normal(size=(matrices[0].shape[1], rank))
        elif self.aux_init == "zeros":
            if mode == 0:
                return tl.zeros(shape=(len(matrices), rank))
            elif mode == 1:
                return [tl.zeros(shape=(matrix.shape[0], rank)) for matrix in matrices]
            elif mode == 2:
                return tl.zeros(shape=(matrices[0].shape[1], rank))
        else:
            if mode in {0, 2} and tl.is_tensor(self.aux_init):
                length_, rank_ = tl.shape(self.aux_init)
                I = len(matrices)
                K = tl.shape(matrices[0])[1]
                if rank != rank_ or (mode == 0 and length_ != I):
                    raise ValueError(
                        "Invalid shape for pre-specified auxiliary variable for mode 0"
                        "\nShould have shape {}, but has shape {}".format((I, rank), (length_, rank_))
                    )
                elif rank != rank_ or (mode == 2 and length_ != K):
                    raise ValueError(
                        "Invalid shape for pre-specified auxiliary variable for mode 2"
                        "\nShould have shape {}, but has shape {}".format((K, rank), (length_, rank_))
                    )

                return self.aux_init
            elif mode in {0, 2} and isinstance(self.aux_init, list):
                raise TypeError("Cannot use list of matrices to initialize auxiliary matrices for mode 0 or 2.")
            elif mode == 1 and isinstance(self.aux_init, list):
                J_is = (tl.shape(matrix)[0] for matrix in matrices)
                shapes = ((J_i, rank) for J_i in J_is)
                if any(tl.shape(aux) != shape for aux, shape in zip(self.aux_init, shapes)):
                    raise ValueError(
                        "Invalid shape for at least one of matrices in the auxiliary variable list for mode 1."
                    )
                elif len(self.aux_init) != len(matrices):
                    raise ValueError(
                        "Different number of pre-specified auxiliary factor matrices for mode 1 "
                        "than the number of coupled matrices."
                    )

                return self.aux_init
            elif mode == 1 and tl.is_tensor(self.aux_init):
                raise TypeError(
                    "Cannot use a tensor (matrix) to initialize auxiliary matrices for mode 1. Must be a list instead."
                )
            else:
                raise ValueError("Unknown aux init: {}".format(self.aux_init))

    def init_dual(self, matrices, rank, mode, random_state=None):
        random_state = tl.check_random_state(random_state)

        if not isinstance(rank, int):
            raise TypeError("Rank must be int, not {}".format(type(rank)))
        if not isinstance(mode, int):
            raise TypeError("Mode must be int, not {}".format(type(mode)))
        elif mode not in [0, 1, 2]:
            raise ValueError("Mode must be 0, 1, or 2.")

        if (
            not isinstance(self.dual_init, str)
            and not tl.is_tensor(self.dual_init)
            and not isinstance(self.dual_init, list)
        ):
            raise TypeError(
                "self.dual_init must be a tensor, a list of tensors or a string specifiying init method, not {}".format(
                    type(self.dual_init)
                )
            )
        if self.dual_init == "random_uniform":
            if mode == 0:
                return random_state.uniform(size=(len(matrices), rank))
            elif mode == 1:
                return [random_state.uniform(size=(matrix.shape[0], rank)) for matrix in matrices]
            elif mode == 2:
                return random_state.uniform(size=(matrices[0].shape[1], rank))
        elif self.dual_init == "random_standard_normal":
            if mode == 0:
                return random_state.standard_normal(size=(len(matrices), rank))
            elif mode == 1:
                return [random_state.standard_normal(size=(matrix.shape[0], rank)) for matrix in matrices]
            elif mode == 2:
                return random_state.standard_normal(size=(matrices[0].shape[1], rank))
        elif self.dual_init == "zeros":
            if mode == 0:
                return tl.zeros(shape=(len(matrices), rank))
            elif mode == 1:
                return [tl.zeros(shape=(matrix.shape[0], rank)) for matrix in matrices]
            elif mode == 2:
                return tl.zeros(shape=(matrices[0].shape[1], rank))
        else:
            if mode in {0, 2} and tl.is_tensor(self.dual_init):
                length_, rank_ = tl.shape(self.dual_init)
                I = len(matrices)
                K = tl.shape(matrices[0])[1]
                if rank != rank_ or (mode == 0 and length_ != I):
                    raise ValueError(
                        "Invalid shape for pre-specified auxiliary variable for mode 0"
                        "\nShould have shape {}, but has shape {}".format((I, rank), (length_, rank_))
                    )
                elif rank != rank_ or (mode == 2 and length_ != K):
                    raise ValueError(
                        "Invalid shape for pre-specified auxiliary variable for mode 2"
                        "\nShould have shape {}, but has shape {}".format((K, rank), (length_, rank_))
                    )

                return self.dual_init
            elif mode in {0, 2} and isinstance(self.dual_init, list):
                raise TypeError("Cannot use list of matrices to initialize auxiliary matrices for mode 0 or 2.")
            elif mode == 1 and isinstance(self.dual_init, list):
                J_is = (tl.shape(matrix)[0] for matrix in matrices)
                shapes = ((J_i, rank) for J_i in J_is)
                if any(tl.shape(dual) != shape for dual, shape in zip(self.dual_init, shapes)):
                    raise ValueError(
                        "Invalid shape for at least one of matrices in the auxiliary variable list for mode 1."
                    )
                elif len(self.dual_init) != len(matrices):
                    raise ValueError(
                        "Different number of pre-specified auxiliary factor matrices for mode 1 "
                        "than the number of coupled matrices."
                    )

                return self.dual_init
            elif mode == 1 and tl.is_tensor(self.dual_init):
                raise TypeError(
                    "Cannot use a tensor (matrix) to initialize auxiliary matrices for mode 1. Must be a list instead."
                )
            else:
                raise ValueError("Unknown aux init: {}".format(self.dual_init))

    @abstractmethod
    def penalty(self, x):  # pragma: nocover
        # TODO: How to deal with penalties that go across matrices
        raise NotImplementedError

    def subtract_from_auxes(self, auxes, duals):
        return [self.subtract_from_aux(aux, dual) for aux, dual in zip(auxes, duals)]

    def subtract_from_aux(self, aux, dual):
        """Compute (aux - dual).
        """
        return aux - dual

    def aux_as_matrix(self, aux):
        return aux

    def auxes_as_matrices(self, auxes):
        return [self.aux_as_matrix(aux) for aux in auxes]


class RowVectorPenalty(ADMMPenalty):
    def factor_matrices_update(self, factor_matrices, feasibility_penalties, auxes):
        """Update a all factor matrices in given list.

        Parameters
        ----------
        factor_matrices : list of tl.tensor(ndim=2)
            List of factor matrix to update.
        feasibility_penalties : list of floats
            Penalty parameters for the feasibility gap of the different factor matrices.
        auxes : list of tl.tensor(ndim=2)
            List of auxiliary matrices, each element corresponds to the auxiliary factor matrix
            for the same element in ``factor_matrices``.
        """
        return [
            self.factor_matrix_update(fm, feasibility_penalty, aux)
            for fm, feasibility_penalty, aux in zip(factor_matrices, feasibility_penalties, auxes)
        ]

    def factor_matrix_update(self, factor_matrix, feasibility_penalty, aux):
        """Update a factor matrix.

        Parameters
        ----------
        factor_matrix : tl.tensor(ndim=2)
            Factor matrix to update.
        feasibility_penalty : float
            Penalty parameter for infeasible solutions.
        aux : tl.tensor(ndim=2)
            Auxiliary matrix that correspond to the factor matrix supplied to ``factor_matrix``.
        """
        out = tl.zeros(factor_matrix.shape)

        for row, factor_matrix_row in enumerate(factor_matrix):
            out[row] = self.factor_matrix_row_update(factor_matrix_row, feasibility_penalty, aux[row])

        return out

    @abstractmethod
    def factor_matrix_row_update(self, factor_matrix_row, feasibility_penalty, aux_row):  # pragma: nocover
        """Update a single row of a factor matrix.

        Parameters
        ----------
        factor_matrix_row : tl.tensor(ndim=1)
            Vector (first order tensor) that corresponds to the single row in the factor matrix
            we wish to update.
        feasibility_penalty : float
            Penalty parameter for infeasible solutions.
        aux_row : tl.tensor(ndim=1)
            Vector (first order tensor) that corresponds to the row in the auxiliary matrix that
            correspond to the row supplied to ``factor_matrix_row``.
        """
        raise NotImplementedError


class MatrixPenalty(ADMMPenalty):
    def factor_matrices_update(self, factor_matrices, feasibility_penalties, auxes):
        return [
            self.factor_matrix_update(fm, feasibility_penalty, aux)
            for fm, feasibility_penalty, aux in zip(factor_matrices, feasibility_penalties, auxes)
        ]

    @abstractmethod
    def factor_matrix_update(self, factor_matrix, feasibility_penalty, aux):  # pragma: nocover
        raise NotImplementedError


class MatricesPenalty(ADMMPenalty):
    @abstractmethod
    def factor_matrices_update(self, factor_matrices, feasibility_penalties, auxes):  # pragma: nocover
        raise NotImplementedError


class NonNegativity(RowVectorPenalty):
    r"""Impose non-negative values for the factor.

    The non-negativity constraint works element-wise, constraining the elements of a factor
    to satisfy :math:`0 \leq x, where :math:`x` represents a factor element

    Parameters
    ----------
    aux_init : {"random_uniform", "random_standard_normal"}
        Initialisation method for the auxiliary variables
    dual_init : {"random_uniform", "random_standard_normal"}
        Initialisation method for the auxiliary variables
    """

    def factor_matrix_row_update(self, factor_matrix_row, feasibility_penalty, aux_row):
        return tl.clip(factor_matrix_row, 0)

    def factor_matrix_update(self, factor_matrix, feasibility_penalty, aux):
        # Return elementwise maximum of zero and factor_matrix
        return tl.clip(factor_matrix, 0)

    def penalty(self, x):
        return 0


# TODO: unit tests
class BoxConstraint(RowVectorPenalty):
    r"""Set minimum and maximum value for the factor.

    A box constraint works element-wise, constraining the elements of a factor
    to satisfy :math:`l \leq x \leq u`, where :math:`x` represents the element
    of the factor and :math:`l` and :math:`u` are the lower and upper bound on
    the factor elements, respectively.

    Parameters
    ----------
    min_val : float
        Lower bound on the factor elements.
    max_val : float
        Upper bound on the factor elements
    aux_init : {"random_uniform", "random_standard_normal"}
        Initialisation method for the auxiliary variables
    dual_init : {"random_uniform", "random_standard_normal"}
        Initialisation method for the auxiliary variables
    """

    def __init__(self, min_val, max_val, aux_init="random_uniform", dual_init="random_uniform"):
        super().__init__(aux_init=aux_init, dual_init=dual_init)
        self.min_val = min_val
        self.max_val = max_val

    def factor_matrix_row_update(self, factor_matrix_row, feasibility_penalty, aux_row):
        """Update a single row of a factor matrix.

        Parameters
        ----------
        factor_matrix_row : tl.tensor(ndim=1)
            Vector (first order tensor) that corresponds to the single row in the factor matrix
            we wish to update.
        feasibility_penalty : float
            Penalty parameter for infeasible solutions.
        aux_row : tl.tensor(ndim=1)
            Vector (first order tensor) that corresponds to the row in the auxiliary matrix that
            correspond to the row supplied to ``factor_matrix_row``.
        """
        return tl.clip(factor_matrix_row, self.min_val, self.max_val)

    def factor_matrix_update(self, factor_matrix, feasibility_penalty, aux):
        """Update a factor matrix.

        Parameters
        ----------
        factor_matrix : tl.tensor(ndim=2)
            Factor matrix to update.
        feasibility_penalty : float
            Penalty parameter for infeasible solutions.  (Ignored)
        aux : tl.tensor(ndim=2)
            Auxiliary matrix that correspond to the factor matrix supplied to ``factor_matrix``.  (Ignored)
        """
        return tl.clip(factor_matrix, self.min_val, self.max_val)

    def penalty(self, x):
        return 0


class L1Penalty(RowVectorPenalty):
    r"""Add L1 (LASSO) regularisation on the factor elements.

    The L1 penalty is frequently used to obtain sparse components. That is,
    components with many zero-valued elements. To accomplish this, the L1 penalty
    adds a penalty term on the form :math:`\sum_{i r} \gamma |a_{ir}|`, where
    :math:`a_{ir}` is the :math:`(i,r)`-th element of the factor matrix.

    Parameters
    ----------
    reg_strength : float
        The regularisation strength, :math:`\gamma` in the equation above
    non_negativity : bool
        If ``True``, then non-negativity is also imposed on the factor elements.
    aux_init : {"random_uniform", "random_standard_normal"}
        Initialisation method for the auxiliary variables
    dual_init : {"random_uniform", "random_standard_normal"}
        Initialisation method for the auxiliary variables
    """
    # TODO: Different scaling versions
    def __init__(self, reg_strength, non_negativity=False, aux_init="random_uniform", dual_init="random_uniform"):
        super().__init__(aux_init=aux_init, dual_init=dual_init)
        if reg_strength < 0:
            raise ValueError("Regularization strength must be nonnegative.")
        self.reg_strength = reg_strength
        self.non_negativity = non_negativity

    def factor_matrix_row_update(self, factor_matrix_row, feasibility_penalty, aux_row):
        if not self.non_negativity:
            sign = tl.sign(factor_matrix_row)
            return sign * tl.clip(tl.abs(factor_matrix_row) - self.reg_strength / feasibility_penalty, 0)
        else:
            return tl.clip(factor_matrix_row - self.reg_strength / feasibility_penalty, 0)

    def factor_matrix_update(self, factor_matrix, feasibility_penalty, aux):
        if not self.non_negativity:
            sign = tl.sign(factor_matrix)
            return sign * tl.clip(tl.abs(factor_matrix) - self.reg_strength / feasibility_penalty, 0)
        else:
            return tl.clip(factor_matrix - self.reg_strength / feasibility_penalty, 0)

    def penalty(self, x):
        # TODO: return reg_strength*l1norm of x
        if tl.is_tensor(x):
            return tl.sum(tl.abs(x)) * self.reg_strength
        else:
            return sum(tl.sum(tl.abs(xi)) for xi in x) * self.reg_strength


class L2Ball(MatrixPenalty):
    def __init__(self, max_norm, aux_init="random_uniform", dual_init="random_uniform"):
        super().__init__(aux_init, dual_init)
        self.max_norm = max_norm

    def factor_matrix_update(self, factor_matrix, feasibility_penalty, aux):
        column_norms = tl.sqrt(tl.sum(factor_matrix ** 2, axis=0))
        column_norms = tl.clip(column_norms, self.max_norm, None)
        return factor_matrix * self.max_norm / column_norms

    def penalty(self, x):
        return 0


class UnitSimplex(MatrixPenalty):
    def compute_lagrange_multiplier(self, factor_matrix_column):
        """Compute lagrange multipliers for the equality constraint: sum(x) = 1 with x >= 0.

        Parameters
        ----------
        factor_matrix_column : ndarray
            Single column of a factor matrix

        Returns
        lagrange_multiplier : float
            The single lagrange multiplier for the simplex constraint.
        """
        # Inspired by https://math.stackexchange.com/questions/2402504/orthogonal-projection-onto-the-unit-simplex
        # But using bisection instead of Newton's method, since Newton's method requires a C2 function, and this is only a C0 function.
        # 0 = ∑_i[x_i] − 1 = ∑_i[min((yi−μ), 0)] - 1

        min_val = -tl.max(factor_matrix_column)
        max_val = tl.max(factor_matrix_column)

        def f(multiplier):
            return tl.sum(tl.clip(factor_matrix_column - multiplier, 0, None)) - 1

        return bisect(f, min_val, max_val)

    def factor_matrix_update(self, factor_matrix, feasibility_penalty, aux):
        output_factor_matrix = tl.zeros(tl.shape(factor_matrix))
        for r in range(tl.shape(factor_matrix)[1]):
            lagrange_multiplier = self.compute_lagrange_multiplier(factor_matrix[:, r])
            output_factor_matrix[:, r] = tl.clip(factor_matrix[:, r] - lagrange_multiplier, 0, None)

        return output_factor_matrix

    def penalty(self, x):
        return 0


class Parafac2(MatricesPenalty):
    r"""Impose the PARAFAC2 constraint on the uncoupled factor matrices.

    The PARAFAC2 constraint can only be imposed on the uncoupled :math:`\mathbf{B}_i`-matrices, and
    states that

    .. math::

        \mathbf{B}_{i_1}^\mathsf{T}\mathbf{B}_{i_1} = \mathbf{B}_{i_2}^\mathsf{T}\mathbf{B}_{i_2},

    for any :math:`i_1` and :math:`i_2`. This constraint ensures uniqueness on the components so
    long as the number of coupled matrices are sufficiently large. A sufficent condition is that
    there are :math:`R(R+1)(R+2)(R+3)/24` matrices, where :math:`R` is the rank of the
    decomposition :cite:p:`harshman1996uniqueness`. However, the true number of datasets required
    for uniqueness is typically lower, and it is conjectured that uniquenes for any :math:`R` holds
    in practice whenever there are four or more matrices :cite:p:`kiers1999parafac2`.

    Parameters
    ----------
    svd : str
        String that specifies which SVD algorithm to use. Valid strings are the keys of ``tensorly.SVD_FUNS``.
    aux_init : {"random_uniform", "random_standard_normal"}
        Initialisation method for the auxiliary variables
    dual_init : {"random_uniform", "random_standard_normal"}
        Initialisation method for the auxiliary variables
    """

    def __init__(self, svd="truncated_svd", aux_init="random_uniform", dual_init="random_uniform"):
        self.svd_fun = get_svd(svd)
        self.aux_init = aux_init
        self.dual_init = dual_init

    def init_aux(self, matrices, rank, mode, random_state=None):
        # TODO: Not provided random state
        if not isinstance(self.aux_init, (str, tuple)):
            raise TypeError(
                "Parafac2 auxiliary variables must be initialized using either a string"
                " or a tuple (containing the orthogonal basis matrices and the coordinate matrix)."
            )

        if not isinstance(rank, int):
            raise TypeError("Rank must be int, not {}".format(type(rank)))
        if not isinstance(mode, int):
            raise TypeError("Mode must be int, not {}".format(type(mode)))

        if mode != 1:
            raise ValueError("PARAFAC2 constraint can only be imposed with mode=1")

        if self.aux_init == "random_uniform":
            coordinate_matrix = random_state.uniform(size=(rank, rank))
            basis_matrices = [tl.eye(M.shape[0], rank) for M in matrices]

            return basis_matrices, coordinate_matrix
        if self.aux_init == "random_standard_normal":
            coordinate_matrix = random_state.standard_normal(size=(rank, rank))
            basis_matrices = [tl.eye(M.shape[0], rank) for M in matrices]

            return basis_matrices, coordinate_matrix
        if self.aux_init == "zeros":
            coordinate_matrix = tl.zeros(shape=(rank, rank))
            basis_matrices = [tl.eye(M.shape[0], rank) for M in matrices]

            return basis_matrices, coordinate_matrix
        elif isinstance(self.aux_init, tuple):
            basis_matrices, coordinate_matrix = self.aux_init

            if not isinstance(basis_matrices, list) or not tl.is_tensor(coordinate_matrix):
                raise TypeError(
                    "If self.aux_init is a tuple, then its first element must be a list of basis matrices "
                    "and second element the coordinate matrix."
                )

            if not len(tl.shape(coordinate_matrix)) == 2:
                raise ValueError(
                    "The coordinate matrix must have two modes, not {}".format(len(tl.shape(coordinate_matrix)))
                )

            if (
                tl.shape(coordinate_matrix)[0] != tl.shape(coordinate_matrix)[1]
                or tl.shape(coordinate_matrix)[0] != rank
            ):
                raise ValueError(
                    "The coordinate matrix must be rank x rank, with rank={}, not {}".format(
                        rank, tl.shape(coordinate_matrix)
                    )
                )

            for matrix, basis_matrix in zip(matrices, basis_matrices):
                if not tl.is_tensor(basis_matrix):
                    raise TypeError("Each basis matrix must be a tensorly tensor")
                if not len(tl.shape(basis_matrix)) == 2:
                    raise ValueError(
                        "Each basis matrix must be tensor with two modes, not {}".format(len(tl.shape(basis_matrix)))
                    )
                if tl.shape(matrix)[0] != tl.shape(basis_matrix)[0] or tl.shape(basis_matrix)[1] != rank:
                    raise ValueError(
                        "The i-th basis matrix must have shape J_i x rank, where J_i is the number of "
                        "rows in the i-th matrix."
                    )

                cross_product = tl.dot(tl.transpose(basis_matrix), basis_matrix)
                if not tl.sum((cross_product - tl.eye(rank)) ** 2) < 1e-8:
                    raise ValueError("The basis matrices must be orthogonal")

            if len(basis_matrices) != len(matrices):
                raise ValueError("There must be as many basis matrices as there are matrices")

            return self.aux_init
        else:
            raise ValueError(f"Unknown aux init: {self.aux_init}")

    def factor_matrices_update(self, factor_matrices, feasibility_penalties, auxes):
        # TODO: docstring
        # TODO: Unit test: Check if PARAFAC2 factor is unchanged
        basis_matrices, coord_mat = auxes
        basis_matrices = []  # To prevent inplace editing of basis matrices
        for fm in factor_matrices:
            U, s, Vh = self.svd_fun(fm @ coord_mat.T, n_eigenvecs=tl.shape(coord_mat)[0])
            basis_matrices.append(U @ Vh)

        coordinate_matrix = 0  # TODO: Project all factor matrices and compute weighted mean
        for fm, basis_matrix, feasibility_penalty in zip(factor_matrices, basis_matrices, feasibility_penalties):
            coordinate_matrix += feasibility_penalty * basis_matrix.T @ fm
        coordinate_matrix /= sum(feasibility_penalties)

        return basis_matrices, coordinate_matrix

    # TODO: change to mixin class
    def subtract_from_aux(self, aux, dual):
        raise TypeError("The PARAFAC2 constraint cannot shift a single factor matrix.")

    def subtract_from_auxes(self, auxes, duals):
        # TODO: Docstrings
        P_is, coord_mat = auxes
        return [tl.dot(P_i, coord_mat) - dual for P_i, dual in zip(P_is, duals)]

    def aux_as_matrix(self, aux):
        raise TypeError("The PARAFAC2 constraint cannot convert a single aux to a matrix")

    def auxes_as_matrices(self, auxes):
        P_is, coord_mat = auxes
        return [tl.dot(P_i, coord_mat) for P_i in P_is]

    def penalty(self, x):
        if not isinstance(x, list):
            raise TypeError("Cannot compute PARAFAC2 penalty of other types than a list of tensors")
        return 0
