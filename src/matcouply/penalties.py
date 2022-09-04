# MIT License: Copyright (c) 2022, Marie Roald.
# See the LICENSE file in the root directory for full license text.

from abc import ABC, abstractmethod

try:
    import condat_tv

    HAS_TV = True
except ImportError:
    HAS_TV = False
import numpy as np
import tensorly as tl
from scipy.optimize import bisect

from ._doc_utils import InheritableDocstrings, copy_ancestor_docstring
from ._unimodal_regression import unimodal_regression
from ._utils import get_svd


class ADMMPenalty(ABC, metaclass=InheritableDocstrings):
    """Base class for all regularizers and constraints.

    Parameters
    ----------
    aux_init : {"random_uniform", "random_standard_normal", "zeros", tl.tensor(ndim=2), list of tl.tensor(ndim=2)}
        Initialisation method for the auxiliary variables
    dual_init : {"random_uniform", "random_standard_normal", "zeros", tl.tensor(ndim=2), list of tl.tensor(ndim=2)}
        Initialisation method for the auxiliary variables
    """

    def __init__(self, aux_init="random_uniform", dual_init="random_uniform"):
        self.aux_init = aux_init
        self.dual_init = dual_init

    def init_aux(self, matrices, rank, mode, random_state=None):
        """Initialize the auxiliary variables

        Initialization schemes

         * ``"random_uniform"``: The elements of the auxiliary variables
            are drawn from a uniform distribution between 0 and 1.
         * ``"random_standard_normal"``: The elements of the auxiliary
            variables are drawn from a standard normal distribution.
         * ``"zeros"``: The elements of the auxiliary variables are
            initialized as zero.
         * tl.tensor(ndim=2) : Pre-computed auxiliary variables (mode=0 or mode=2)
         * list of tl.tensor(ndim=2): Pre-computed auxiliary variables (mode=1)

        Parameters
        ----------
        matrices : list of tensor(ndim=2) or tensor(ndim=3)
            The data matrices represented by the coupled matrix factorization
            these auxiliary variables correspond to.
        rank : int
            Rank of the decomposition.
        mode : int
            The mode represented by the factor matrices that these
            auxiliary variables correspond to.
        random_state : RandomState
            TensorLy random state.

        Returns
        -------
        tl.tensor(ndim=2) or list of tl.tensor(ndim=2)
        """
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

        # Initialize using given aux variables
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

        # Generate initialization based on option
        if self.aux_init == "random_uniform":
            if mode == 0:
                return tl.tensor(random_state.uniform(size=(len(matrices), rank)))
            elif mode == 1:
                return [tl.tensor(random_state.uniform(size=(matrix.shape[0], rank))) for matrix in matrices]
            elif mode == 2:
                return tl.tensor(random_state.uniform(size=(matrices[0].shape[1], rank)))
        elif self.aux_init == "random_standard_normal":
            if mode == 0:
                return tl.tensor(random_state.standard_normal(size=(len(matrices), rank)))
            elif mode == 1:
                return [tl.tensor(random_state.standard_normal(size=(matrix.shape[0], rank))) for matrix in matrices]
            elif mode == 2:
                return tl.tensor(random_state.standard_normal(size=(matrices[0].shape[1], rank)))
        elif self.aux_init == "zeros":
            if mode == 0:
                return tl.zeros((len(matrices), rank))
            elif mode == 1:
                return [tl.zeros((matrix.shape[0], rank)) for matrix in matrices]
            elif mode == 2:
                return tl.zeros((matrices[0].shape[1], rank))
        else:
            raise ValueError("Unknown aux init: {}".format(self.aux_init))

    def init_dual(self, matrices, rank, mode, random_state=None):
        """Initialize the dual variables

        Initialization schemes

         * ``"random_uniform"``: The elements of the dual variables
            are drawn from a uniform distribution between 0 and 1.
         * ``"random_standard_normal"``: The elements of the dual
            variables are drawn from a standard normal distribution.
         * ``"zeros"``: The elements of the dual variables are
            initialized as zero.
         * tl.tensor(ndim=2) : Pre-computed dual variables (mode=0 or mode=2)
         * list of tl.tensor(ndim=2): Pre-computed dual variables (mode=1)

        Parameters
        ----------
        matrices : list of tensor(ndim=2) or tensor(ndim=3)
            The data matrices represented by the coupled matrix factorization
            these dual variables correspond to.
        rank : int
            Rank of the decomposition.
        mode : int
            The mode represented by the factor matrices that these
            dual variables correspond to.
        random_state : RandomState
            TensorLy random state.

        Returns
        -------
        tl.tensor(ndim=2) or list of tl.tensor(ndim=2)
        """
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

        # Initialize using given dual variables
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

        # Generate initialization based on option
        if self.dual_init == "random_uniform":
            if mode == 0:
                return tl.tensor(random_state.uniform(size=(len(matrices), rank)))
            elif mode == 1:
                return [tl.tensor(random_state.uniform(size=(matrix.shape[0], rank))) for matrix in matrices]
            elif mode == 2:
                return tl.tensor(random_state.uniform(size=(matrices[0].shape[1], rank)))
        elif self.dual_init == "random_standard_normal":
            if mode == 0:
                return tl.tensor(random_state.standard_normal(size=(len(matrices), rank)))
            elif mode == 1:
                return [tl.tensor(random_state.standard_normal(size=(matrix.shape[0], rank))) for matrix in matrices]
            elif mode == 2:
                return tl.tensor(random_state.standard_normal(size=(matrices[0].shape[1], rank)))
        elif self.dual_init == "zeros":
            if mode == 0:
                return tl.zeros((len(matrices), rank))
            elif mode == 1:
                return [tl.zeros((matrix.shape[0], rank)) for matrix in matrices]
            elif mode == 2:
                return tl.zeros((matrices[0].shape[1], rank))
        else:
            raise ValueError("Unknown aux init: {}".format(self.dual_init))

    @abstractmethod
    def penalty(self, x):  # pragma: nocover
        """Compute the penalty for the given factor matrix or list of factor matrices."""
        raise NotImplementedError

    def subtract_from_auxes(self, auxes, duals):
        """Compute (aux - dual) for each auxiliary- and dual-factor matrix for mode=1.

        For some penalties, the aux is not a list of factor matrices but rather some
        other parametrization of a list of factor matrices. This function is used so
        the AO-ADMM procedure can work with any auxiliary-variable parametrization
        seamlessly.

        Parameters
        ----------
        auxes : list of tl.tensor(ndim=2)
            Auxiliary variables
        duals : list of tl.tensor(ndim=2)
            Dual variables (or other variable to subtract from the auxes)

        Returns
        -------
        list of tl.tensor(ndim=2)
            The list of differences
        """
        return [self.subtract_from_aux(aux, dual) for aux, dual in zip(auxes, duals)]

    def subtract_from_aux(self, aux, dual):
        """Compute (aux - dual) for mode=0 and mode=2.

        For some penalties, the aux is not a factor matrix but rather some other
        parametrization of a matrix. This function is used so the AO-ADMM procedure
        can work with any auxiliary-variable parametrization seamlessly.

        Parameters
        ----------
        aux : tl.tensor(ndim=2)
            Auxiliary variables
        dual : tl.tensor(ndim=2)
            Dual variables (or other variable to subtract from the auxes)

        Returns
        -------
        tl.tensor(ndim=2)
            The list of differences
        """
        return aux - dual

    def aux_as_matrix(self, aux):
        """Convert an auxiliary variable to a matrix (mode=0 and mode=2).

        This is an identity function that just returns its input. However,
        it is required for AO-ADMM to seamlessly work when the auxiliary variable
        is a parametrization of a matrix.

        Parameters
        ----------
        aux : tl.tensor(ndim=2)

        Returns
        -------
        tl.tensor(ndim=2)
        """
        return aux

    def auxes_as_matrices(self, auxes):
        """Convert a list of auxiliary variables to a list of matrices (mode=1).

        This is an identity function that just returns its input. However,
        it is required for AO-ADMM to seamlessly work when the auxiliary variable
        is a parametrization of a matrix.

        Parameters
        ----------
        auxes : list of tl.tensor(ndim=2)

        Returns
        -------
        list of tl.tensor(ndim=2)
        """
        return [self.aux_as_matrix(aux) for aux in auxes]

    def _auto_add_param_to_repr(self, param):
        if param.startswith("_"):
            return False
        elif param in {"aux_init", "dual_init"}:
            return False
        return True

    def __repr__(self):
        param_strings = [
            f"{key}={repr(value)}" for key, value in self.__dict__.items() if self._auto_add_param_to_repr(key)
        ]
        if isinstance(self.aux_init, str):
            param_strings.append(f"aux_init='{self.aux_init}'")
        else:
            param_strings.append("aux_init=given_init")
        if isinstance(self.dual_init, str):
            param_strings.append(f"dual_init='{self.dual_init}'")
        else:
            param_strings.append("dual_init=given_init")

        params = ", ".join(param_strings)
        return f"<'{self.__module__}.{type(self).__name__}' with {params})>"


class MatricesPenalty(ADMMPenalty):
    """Base class for penalties that are applied to a list of factor matrices simultaneously."""

    @abstractmethod
    def factor_matrices_update(self, factor_matrices, feasibility_penalties, auxes):  # pragma: nocover
        """Update all factor matrices in given list according to this penalty.

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
        raise NotImplementedError


class MatrixPenalty(MatricesPenalty):
    """Base class for penalties that can be applied to a single factor matrix at a time."""

    def factor_matrices_update(self, factor_matrices, feasibility_penalties, auxes):
        """Update all factor matrices in given list according to this penalty.

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

    @abstractmethod
    def factor_matrix_update(self, factor_matrix, feasibility_penalty, aux):  # pragma: nocover
        """Update a factor matrix according to this penalty.

        Parameters
        ----------
        factor_matrix : tl.tensor(ndim=2)
            Factor matrix to update.
        feasibility_penalty : float
            Penalty parameter for infeasible solutions.
        aux : tl.tensor(ndim=2)
            Auxiliary matrix that correspond to the factor matrix supplied to ``factor_matrix``.
        """
        raise NotImplementedError


class RowVectorPenalty(MatrixPenalty):
    """Base class for penalties that can be applied to one row of a factor matrix at a time."""

    def factor_matrix_update(self, factor_matrix, feasibility_penalty, aux):
        """Update a factor matrix according to this penalty.

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
        """Update a single row of a factor matrix according to this constraint.

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


class HardConstraintMixin(metaclass=InheritableDocstrings):
    """Mixin for hard constraints."""

    def penalty(self, x):
        """Returns 0 as there is no penalty for hard constraints.

        Hard constraints are always penalised with 0 even when the components are infeasible.
        Slightly infeasible components would otherwise result in an infinite penalty because
        the penalty function of hard constraints is 0 for feasible solutions and infinity for
        infeasible solutions. An infinite penalty would stop all convergence checking and not
        provide any information on the quality of the components. To ensure that the hard
        constraints are sufficiently imposed, it is recommended to examine the feasibility gap
        instead of the penalty and ensure that the feasibility gap is low.

        Parameters
        ----------
        x : tl.tensor(ndim=2) or list of tl.tensor(ndim=2)
            Factor matrix or list of factor matrices.
        """
        return 0


class NonNegativity(HardConstraintMixin, RowVectorPenalty):
    r"""Impose non-negative values for the factor.

    The non-negativity constraint works element-wise, constraining the elements of a factor
    to satisfy :math:`0 \leq x`, where :math:`x` represents a factor element

    Parameters
    ----------
    aux_init : {"random_uniform", "random_standard_normal", "zeros", tl.tensor(ndim=2), list of tl.tensor(ndim=2)}
        Initialisation method for the auxiliary variables
    dual_init : {"random_uniform", "random_standard_normal", "zeros", tl.tensor(ndim=2), list of tl.tensor(ndim=2)}
        Initialisation method for the auxiliary variables
    """

    @copy_ancestor_docstring
    def factor_matrix_row_update(self, factor_matrix_row, feasibility_penalty, aux_row):
        return tl.clip(factor_matrix_row, 0, float("inf"))

    @copy_ancestor_docstring
    def factor_matrix_update(self, factor_matrix, feasibility_penalty, aux):
        return tl.clip(factor_matrix, 0, float("inf"))


class BoxConstraint(HardConstraintMixin, RowVectorPenalty):
    r"""Set minimum and maximum value for the factor.

    A box constraint works element-wise, constraining the elements of a factor
    to satisfy :math:`l \leq x \leq u`, where :math:`x` represents the element
    of the factor and :math:`l` and :math:`u` are the lower and upper bound on
    the factor elements, respectively.

    Parameters
    ----------
    min_val : float or None
        Lower bound on the factor elements.
    max_val : float or None
        Upper bound on the factor elements
    aux_init : {"random_uniform", "random_standard_normal", "zeros", tl.tensor(ndim=2), list of tl.tensor(ndim=2)}
        Initialisation method for the auxiliary variables
    dual_init : {"random_uniform", "random_standard_normal", "zeros", tl.tensor(ndim=2), list of tl.tensor(ndim=2)}
        Initialisation method for the auxiliary variables
    """

    def __init__(self, min_val, max_val, aux_init="random_uniform", dual_init="random_uniform"):
        super().__init__(aux_init=aux_init, dual_init=dual_init)
        self.min_val = min_val
        self.max_val = max_val

    @copy_ancestor_docstring
    def factor_matrix_row_update(self, factor_matrix_row, feasibility_penalty, aux_row):
        return tl.clip(factor_matrix_row, self.min_val, self.max_val)

    @copy_ancestor_docstring
    def factor_matrix_update(self, factor_matrix, feasibility_penalty, aux):
        return tl.clip(factor_matrix, self.min_val, self.max_val)


class L1Penalty(RowVectorPenalty):
    r"""Add L1 (LASSO) regularization on the factor elements.

    The L1 penalty is frequently used to obtain sparse components. That is,
    components with many zero-valued elements. To accomplish this, the L1 penalty
    adds a penalty term on the form :math:`\sum_{i r} \gamma |a_{ir}|`, where
    :math:`a_{ir}` is the :math:`(i,r)`-th element of the factor matrix.

    Parameters
    ----------
    reg_strength : float
        The regularization strength, :math:`\gamma` in the equation above
    non_negativity : bool
        If ``True``, then non-negativity is also imposed on the factor elements.
    aux_init : {"random_uniform", "random_standard_normal", "zeros", tl.tensor(ndim=2), list of tl.tensor(ndim=2)}
        Initialisation method for the auxiliary variables
    dual_init : {"random_uniform", "random_standard_normal", "zeros", tl.tensor(ndim=2), list of tl.tensor(ndim=2)}
        Initialisation method for the auxiliary variables
    """

    def __init__(self, reg_strength, non_negativity=False, aux_init="random_uniform", dual_init="random_uniform"):
        super().__init__(aux_init=aux_init, dual_init=dual_init)
        if reg_strength < 0:
            raise ValueError("Regularization strength must be nonnegative.")
        self.reg_strength = reg_strength
        self.non_negativity = non_negativity

    @copy_ancestor_docstring
    def factor_matrix_row_update(self, factor_matrix_row, feasibility_penalty, aux_row):
        if self.non_negativity:
            return tl.clip(factor_matrix_row - self.reg_strength / feasibility_penalty, 0, float("inf"))

        sign = tl.sign(factor_matrix_row)
        return sign * tl.clip(tl.abs(factor_matrix_row) - self.reg_strength / feasibility_penalty, 0, float("inf"))

    @copy_ancestor_docstring
    def factor_matrix_update(self, factor_matrix, feasibility_penalty, aux):
        if self.non_negativity:
            return tl.clip(factor_matrix - self.reg_strength / feasibility_penalty, 0, float("inf"))

        sign = tl.sign(factor_matrix)
        return sign * tl.clip(tl.abs(factor_matrix) - self.reg_strength / feasibility_penalty, 0, float("inf"))

    @copy_ancestor_docstring
    def penalty(self, x):
        if tl.is_tensor(x):
            return tl.sum(tl.abs(x)) * self.reg_strength
        return sum(tl.sum(tl.abs(xi)) for xi in x) * self.reg_strength


class GeneralizedL2Penalty(MatrixPenalty):
    r"""Penalty on the form :math:`\mathbf{x}^\mathsf{T} \mathbf{Mx}`, where :math:`\mathbf{M}` is any symmetric positive semidefinite matrix.

    The generalized L2 penalty adds a squared (semi-)norm penalty on the form

    .. math::

        g(\mathbf{x}) = \mathbf{x}^\mathsf{T} \mathbf{Mx}

    where :math:`\mathbf{M}` is a symmetric positive semidefinite matrix and :math:`\mathbf{x}`
    is a vector. This penalty is imposed for all columns of the factor matrix (or matrices for
    the :math:`\mathbf{B}^{(i)}`-s). Note that the regular L2-penalty (or Ridge penalty) is a special
    case of the generalised L2 penalty, which we obtain by setting :math:`\mathbf{M}=\mathbf{I}`.
    Also, this penalty is a squared seminorm penalty since

    .. math::

        g(\mathbf{x}) = \mathbf{x}^\mathsf{T} \mathbf{Mx} = \| \mathbf{Lx} \|_2^2,

    where :math:`\mathbf{L}` is a Cholesky factorization of :math:`\mathbf{M}`. However, the
    formulation with :math:`\mathbf{M}` is more practical than the formulation with :math:`\mathbf{L}`,
    since

    .. math::

        \mathbf{M} = \mathbf{L}^\mathsf{T}\mathbf{L}

    is easy to compute with :math:`\mathbf{L}` known, but not wise-versa (e.g. if
    :math:`\mathbf{M}` is indefinite).

    **Graph Laplacian penalty**

    A special case of the generalized L2 penalty is the graph Laplacian penalty. This penalty
    is on the form

    .. math::

        g(\mathbf{x}) = \sum_{m=1}^M \sum_{n=1}^N w_{mn} (x_m - x_n)^2.

    That is, squared differences between the different component vector elements are penalised.
    Graph laplacian penalties are useful, for example, in image processing, where :math:`w_{mn}`
    is a high number for vector elements that represent pixels that are close to each other and
    a low number for vector elements that represent pixels that are far apart.

    To transform the graph Laplacian penalty into a generalised L2 penalty, we consider the
    component vector elements as nodes in a graph and :math:`w_{mn}` as the edge weight between
    node :math:`m` and node :math:`m`. Then, we set :math:`\mathbf{M}` equal to the graph Laplacian
    of this graph. That is

    .. math::

        m_{mn} = \begin{cases}
            -w_{mn}   & m \neq n \\
            \sum_m  & m = n
        \end{cases}.

    **The proximal operator:**

    The proximal operator of the generalised L2 penalty is obtained by solving

    .. math::

        \text{prox}_{\mathbf{x}^\mathsf{T} \mathbf{Mx}}(\mathbf{x})
        = \left(\mathbf{M} + 0.5\rho\mathbf{I}\right)^{-1}0.5\rho\mathbf{x},

    where :math:`\rho` is the scale parameter. There are several ways to solve this equation.
    One of which is via the SVD. Let :math:`\mathbf{M} = \mathbf{USU}^\mathsf{T}`, then, the
    proximal operator is given by

    .. math::

        \text{prox}_{\mathbf{x}^\mathsf{T} \mathbf{Mx}}(\mathbf{x})
        = (\mathbf{U}(\mathbf{S} + 0.5\rho\mathbf{I})^{-1}\mathbf{U}^\mathsf{T}) 0.5 \rho \mathbf{x}.

    This operation is fast once :math:`\mathbf{U}` and :math:`\mathbf{S}` are known, since solving the diagonal
    system :math:`(\mathbf{S} + 0.5\rho\mathbf{I})` is fast.

    Parameters
    ----------
    norm_matrix : tl.tensor(ndim=2)
        The :math:`\mathbf{M}`-matrix above
    svd : str
        String that specifies which SVD algorithm to use. Valid strings are the keys of ``tensorly.SVD_FUNS``.
    aux_init : {"random_uniform", "random_standard_normal", "zeros", tl.tensor(ndim=2), list of tl.tensor(ndim=2)}
        Initialisation method for the auxiliary variables
    dual_init : {"random_uniform", "random_standard_normal", "zeros", tl.tensor(ndim=2), list of tl.tensor(ndim=2)}
        Initialisation method for the auxiliary variables
    validate : bool (default=True)
        Enable naive validation of the norm matrix.

    Examples
    --------
    This example creates a generalised L2 penalty for a simple Laplacian penalty on
    the form :math:`\sum_{n} (x_n - x_{n-1})^2`.

    >>> import numpy as np
    >>> import tensorly as tl
    >>> from matcouply.penalties import GeneralizedL2Penalty
    >>> num_elements = 30
    >>> M = 2 * np.eye(num_elements) - np.eye(num_elements, k=1) - np.eye(num_elements, k=-1)
    >>> M[0, 0] = 1
    >>> M[-1, -1] = 1
    >>> M = tl.tensor(M)
    >>> penalty = GeneralizedL2Penalty(M)

    This penalty can now be added to ``matcouply.decomposition.cmf_aoadmm`` via the ``regs``-parameter.
    Alternatively, if the ``generalized_l2_penalty``-argument of ``matcouply.decomposition.cmf_aoadmm`` is
    used, then a ``GeneralizedL2Penalty`` is added with ``method="svd"``.
    """

    def __init__(
        self, norm_matrix, svd="truncated_svd", aux_init="random_uniform", dual_init="random_uniform", validate=True,
    ):
        super().__init__(aux_init, dual_init)
        self.norm_matrix = norm_matrix
        self.svd = svd  # Useful for the __repr__
        self.validate = validate  # Useful for the __repr__

        sign_matrix = -tl.ones(tl.shape(norm_matrix))
        sign_matrix = sign_matrix + 2 * tl.eye(tl.shape(norm_matrix)[0])
        if validate and (
            not tl.all(tl.transpose(norm_matrix) == norm_matrix)
            # FIXME: Validate eigenvalues also when/if tensorly gets eigvals function
        ):
            raise ValueError("The norm matrix should be symmetric positive semidefinite")
        if validate and tl.get_backend() == "numpy":
            eigvals = np.linalg.eigvals(norm_matrix)
            if np.any(eigvals < -1e-14):
                raise ValueError("The norm matrix should be symmetric positive semidefinite")

        self._U, self._s, _ = self.svd_fun(norm_matrix)  # Ignore Vh since norm matrix is symmetric

    @property
    def svd_fun(self):
        return get_svd(self.svd)

    @copy_ancestor_docstring
    def factor_matrix_update(self, factor_matrix, feasibility_penalty, aux):
        s_aug = self._s + 0.5 * feasibility_penalty
        tmp = 0.5 * feasibility_penalty * factor_matrix
        tmp = tl.dot(tl.transpose(self._U), tmp)
        tmp = tl.dot((self._U * (1 / s_aug)), tmp)
        return tmp

    def _penalty(self, x):
        return tl.trace(tl.dot(tl.dot(tl.transpose(x), self.norm_matrix), x))

    @copy_ancestor_docstring
    def penalty(self, x):
        if tl.is_tensor(x):
            return self._penalty(x)
        else:
            return sum(self._penalty(xi) for xi in x)


class TotalVariationPenalty(MatrixPenalty):
    r"""Impose piecewise constant components

    Total variation regularization imposes piecewise constant components by
    obtaining components whose derivative is sparse. This sparsity is obtained
    using an L1 penalty. That is

    .. math::

        g(\mathbf{x}) = \alpha \|\nabla \mathbf{x}\|_1 = \alpha \sum_{n=1}^{N} |x_n - x_{n-1}|,

    where :math:`\alpha` is a regularization coefficient that controls the sparsity
    level of the gradient and :math:`\nabla` is the finite difference operator.
    :math:`\mathbf{x}` is a column vector of a factor matrix, and all column vectors
    are penalised equally.

    The total variation penalty is compatible with the L1 penalty in the sense that
    it is easy to compute the proximal operator of

    .. math::

        g(\mathbf{x}) = \alpha \sum_{n=2}^{N} |x_n - x_{n-1}| + \sum_{n=1}^N \| x_n \|_1.

    Specifically, if we first evaluate the proximal operator for the total variation penalty,
    followed by the proximal operator of the L1 penalty, then that is equivalent to
    evaluating the proximal operator of the sum of a TV and an L1 penalty :cite:p:`friedman2007pathwise`.

    To evaluate the proximal operator, we use the improved direct total variation algorithm
    by Laurent Condat :cite:p:`condat2013direct` (C code of the improved version is available
    here: https://lcondat.github.io/publications.html).

    Parameters
    ----------
    reg_strength : float (> 0)
        The strength of the total variation regularization (:math:`\alpha` above)
    l1_strength : float (>= 0)
        The strength of the L1 penalty (:math:`\beta` above)
    aux_init : {"random_uniform", "random_standard_normal", "zeros", tl.tensor(ndim=2), list of tl.tensor(ndim=2)}
        Initialisation method for the auxiliary variables
    dual_init : {"random_uniform", "random_standard_normal", "zeros", tl.tensor(ndim=2), list of tl.tensor(ndim=2)}
        Initialisation method for the auxiliary variables

    Note
    ----
    The C-code this penalty is based in has a CeCILL lisence, which is compatible with GPL, but
    not MIT. This penalty is therefore not available with the default installation of
    MatCoupLy. To use this penalty, you need to install the GPL-lisenced ``condat-tv``.

    Note
    ----
    This penalty is only available with the numpy backend, since it is based on
    an external C-library.
    """

    def __init__(
        self, reg_strength, l1_strength=0, aux_init="random_uniform", dual_init="random_uniform",
    ):
        if not HAS_TV:
            raise ModuleNotFoundError(
                "Cannot use total variation penalty without the ``condat_tv`` package (GPL-3 lisenced). "
                "Install with ``pip install condat_tv``."
            )
        if reg_strength <= 0:
            raise ValueError("The TV regularization strength must be positive.")
        if l1_strength < 0:
            raise ValueError("The L1 regularization strength must be non-negative.")
        super().__init__(aux_init, dual_init)
        self.reg_strength = reg_strength
        self.l1_strength = l1_strength

    @copy_ancestor_docstring
    def factor_matrix_update(self, factor_matrix, feasibility_penalty, aux):
        X = tl.transpose(
            condat_tv.tv_denoise_matrix(tl.transpose(factor_matrix), self.reg_strength * 2 / feasibility_penalty)
        )
        if self.l1_strength:
            return tl.sign(X) * tl.clip(tl.abs(X) - self.l1_strength / feasibility_penalty, 0, float("inf"))
        else:
            return X

    def _penalty(self, x):
        penalty = self.reg_strength * tl.sum(tl.abs(np.diff(x, axis=0)))
        if self.l1_strength:
            penalty = penalty + self.l1_strength * tl.sum(tl.abs(x))
        return penalty

    @copy_ancestor_docstring
    def penalty(self, x):
        if tl.is_tensor(x):
            return self._penalty(x)
        else:
            return sum(self._penalty(xi) for xi in x)


class L2Ball(HardConstraintMixin, MatrixPenalty):
    r"""Ensure that the L2-norm of component vectors are less than a given scalar.

    This is a hard constraint on the L2-norm of the component vectors given by

    .. math::

        \|\mathbf{x}\|_2 \leq r,

    where :math:`\mathbf{x}` is a column vector for a factor matrix and
    :math:`r` is a positive constant.

    The L2-ball constraint is compatible with the non-negativity constraint.

    Parameters
    ----------
    norm_bound : float (> 0)
        Maximum L2-norm of the component vectors (:math:`r` above)
    non_negativity : float
        If true, then non-negativity is imposed too
    aux_init : {"random_uniform", "random_standard_normal", "zeros", tl.tensor(ndim=2), list of tl.tensor(ndim=2)}
        Initialisation method for the auxiliary variables
    dual_init : {"random_uniform", "random_standard_normal", "zeros", tl.tensor(ndim=2), list of tl.tensor(ndim=2)}
        Initialisation method for the auxiliary variables

    Note
    ----
    **Proof of compatibility with non-negativity constraints**

    The compatibility with non-negativity constraints can be obtained with the
    standard projection onto convex sets (POCS) algorithm, which states that the
    projection onto a convex set that is the intersection of two other convex sets
    :math:`\mathcal{C} = \mathcal{C}_1 \cap \mathcal{C}_2`,
    :math:`\mathcal{P}_\mathcal{C}` is given by

    .. math::
        \mathcal{P}_\mathcal{C} = \prod_{n=1}^\infty \mathcal{P}_{\mathcal{C}_1} \mathcal{P}_{\mathcal{C}_2},

    where :math:`\mathcal{P}_{\mathcal{C}_1}` and :math:`\mathcal{P}_{\mathcal{C}_2}`
    are the projections onto :math:`\mathcal{C}_1` and :math:`\mathcal{C}_2`,
    respectively. In other words, to project only :math:`\mathcal{C}`, we can
    alternatingly project onto :math:`\mathcal{C}_1` and :math:`\mathcal{C}_2`.

    We can now use this relation to prove that the projection onto the intersection
    of the L2 ball and the non-negative orthant can be obtained by first projecting
    onto the L2 ball followed by a projection onto the non-negative orthant. Consider
    any point :math:`\mathbf{x} \in \mathbb{R}^N`. The projection onto the L2 ball
    of radius :math:`r` is given by:

    .. math::

        \mathbf{x}^{(0.5)} = \frac{\min(\|\mathbf{x}\|_2, r) \mathbf{x}}{\|\mathbf{x}\|_2}.

    If any entries in :math:`\mathbf{x}` are negative, then their sign will not change.
    Next, we project :math:`\mathbf{x}^{(0.5)}` onto the non-negative orthant:

    .. math::

        x_n^{(1)} = \max(x_n^{(0.5)}, 0).

    This operation has the property that :math:`\|\mathbf{x}^{(1)}\|_2 \leq \|\mathbf{x}^{(0.5)}\|_2 \leq`.
    Thus, any subsequent projection either onto the L2-ball of radius :math:`r` or
    the non-negative orthant will not change :math:`\mathbf{x}^{(1)}`, which means that
    :math:`\mathbf{x}^{(1)}` is the projection of :math:`\mathbf{x}` onto the intersection
    of the L2 ball of radius :math:`r` and the non-negative orthant.
    """

    def __init__(self, norm_bound, non_negativity=False, aux_init="random_uniform", dual_init="random_uniform"):
        super().__init__(aux_init, dual_init)
        self.norm_bound = norm_bound
        self.non_negativity = non_negativity

        if norm_bound <= 0:
            raise ValueError("The norm bound must be positive.")

    @copy_ancestor_docstring
    def factor_matrix_update(self, factor_matrix, feasibility_penalty, aux):
        if self.non_negativity:
            factor_matrix = tl.clip(factor_matrix, 0, float("inf"))
        column_norms = tl.sqrt(tl.sum(factor_matrix ** 2, axis=0))
        column_norms = tl.clip(column_norms, self.norm_bound, float("inf"))
        return factor_matrix * self.norm_bound / column_norms


class UnitSimplex(HardConstraintMixin, MatrixPenalty):
    """Constrain the component-vectors so they are non-negative and sum to 1.

    This is a hard constraint which is useful when the component-vectors represent
    probabilities.

    Parameters
    ----------
    aux_init : {"random_uniform", "random_standard_normal", "zeros", tl.tensor(ndim=2), list of tl.tensor(ndim=2)}
        Initialisation method for the auxiliary variables
    dual_init : {"random_uniform", "random_standard_normal", "zeros", tl.tensor(ndim=2), list of tl.tensor(ndim=2)}
        Initialisation method for the auxiliary variables
    """

    def _compute_lagrange_multiplier(self, factor_matrix_column):
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
        # But using bisection instead of Newton's method, since Newton's method requires a C2 function,
        # and this is only a C0 function.
        # 0 = ∑_i[x_i] − 1 = ∑_i[min((yi−μ), 0)] - 1
        min_val = tl.min(factor_matrix_column) - 1
        max_val = tl.max(factor_matrix_column)

        # Add a little buffer to the tolerance to account for floating point errors
        min_val -= 1e-5
        min_val = min(0.9 * min_val, 1.1 * min_val)

        max_val += 1e-5
        max_val = max(0.9 * max_val, 1.1 * max_val)

        def f(multiplier):
            return tl.sum(tl.clip(factor_matrix_column - multiplier, 0, None)) - 1

        return bisect(f, min_val, max_val)

    @copy_ancestor_docstring
    def factor_matrix_update(self, factor_matrix, feasibility_penalty, aux):
        output_factor_matrix = tl.zeros(tl.shape(factor_matrix))
        for r in range(tl.shape(factor_matrix)[1]):
            lagrange_multiplier = self._compute_lagrange_multiplier(factor_matrix[:, r])
            output_factor_matrix[:, r] = tl.clip(factor_matrix[:, r] - lagrange_multiplier, 0, None)

        return output_factor_matrix


class Unimodality(HardConstraintMixin, MatrixPenalty):
    r"""Constrain the component-vectors so they are unimodal.

    Unimodal vectors :math:`\mathbf{u} \in \mathbb{R}^n` have the property that

    .. math::
        u_1 \leq u_2 \leq ... \leq u_{t-1} \leq u_t \geq u_{t+1} \geq ... \geq u_{n-1} \geq u_n

    Projecting a general vector into the set of unimodal vectors (called unimodal regression) requires solving
    a set of isotonic regression problems (i.e. projections onto monotincally increasing or decreasing vectors).
    Two isotonic regression problems for each element in the vector. However, there is an incremental algorithm
    for fitting isotonic regression problems called *prefix isotonic regression* :cite:p:`stout2008unimodal`,
    which can be used to solve unimodal problems in linear time :cite:p:`stout2008unimodal`.

    Parameters
    ----------
    non_negativity : bool
        If True, then the components will also be non-negative
    aux_init : {"random_uniform", "random_standard_normal", "zeros", tl.tensor(ndim=2), list of tl.tensor(ndim=2)}
        Initialisation method for the auxiliary variables
    dual_init : {"random_uniform", "random_standard_normal", "zeros", tl.tensor(ndim=2), list of tl.tensor(ndim=2)}
        Initialisation method for the auxiliary variables
    """

    def __init__(self, non_negativity=False, aux_init="random_uniform", dual_init="random_uniform"):
        if tl.get_backend() != "numpy":
            raise RuntimeError("Unimodality is only supported with the Numpy backend")
        super().__init__(aux_init, dual_init)
        self.non_negativity = non_negativity

    @copy_ancestor_docstring
    def factor_matrix_update(self, factor_matrix, feasibility_penalty, aux):
        return unimodal_regression(factor_matrix, non_negativity=self.non_negativity)


class Parafac2(MatricesPenalty):
    r"""Impose the PARAFAC2 constraint on the uncoupled factor matrices.

    The PARAFAC2 constraint can only be imposed on the uncoupled :math:`\mathbf{B}^{(i)}`-matrices, and
    states that

    .. math::

        {\mathbf{B}^{(i_1)}}^{\mathsf{T}}\mathbf{B}^{(i_1)} = {\mathbf{B}^{(i_2)}}^{\mathsf{T}}\mathbf{B}^{(i_2)},

    for any :math:`i_1` and :math:`i_2`. This constraint ensures uniqueness on the components so
    long as the number of coupled matrices are sufficiently large. A sufficent condition is that
    there are :math:`R(R+1)(R+2)(R+3)/24` matrices, where :math:`R` is the rank of the
    decomposition :cite:p:`harshman1996uniqueness`. However, the true number of datasets required
    for uniqueness is typically lower, and it is conjectured that uniquenes for any :math:`R` holds
    in practice whenever there are four or more matrices :cite:p:`kiers1999parafac2`.

    **Parametrization of matrix-collections that satisfy the PARAFAC2 constraint**

    The standard way of parametrizing collections of matrices that satisfy the PARAFAC2 constraint
    is due to Kiers et al. :cite:p:`kiers1999parafac2`. If :math:`\{\mathbf{B}^{(i)}\}_{i=1}^I` satsifies
    the PARAFAC2 constraint, then there exists a matrix :math:`\mathbf{\Delta} \in \mathbb{R}^{R \times R}`
    (the coordinate matrix) and a collection of orthonormal matrices :math:`\{\mathbf{P}^{(i)}\}_{i=1}^I`
    (the orthogonal basis matrices) such that

    .. math::

        \mathbf{B}^{(i)} = \mathbf{P}^{(i)} \mathbf{\Delta}.

    For this implementation, we use the above parametrization of the auxiliary variables, which
    is a tuple whose first element is a list of orthogonal basis matrices and second element
    is the coordinate matrix.

    **The proximal operator**

    To evaluate the proximal operator, we use the projection scheme presented in
    :cite:p:`roald2021admm,roald2021parafac2`. Specifically, we project with a coordinate descent
    scheme, where we first update the basis matrices and then update the coordinate matrix. It
    has been observed that only one iteration of this coordinate descent scheme is sufficient
    for fitting PARAFAC2 models with AO-ADMM :cite:p:`roald2021admm,roald2021parafac2`.

    To project :math:`\{\mathbf{X}^{(i)}\}_{i=1}^I` onto the set of collections of matrices that
    satisfy the PARAFAC2 constraint, we first update the orthogonal basis matrices by

    .. math::

        \mathbf{P}^{(i)} = \mathbf{U}^{(i)} {\mathbf{V}^{(i)}}^\mathsf{T}

    where :math:`\mathbf{U}^{(i)}` and :math:`\mathbf{V}^{(i)}` contain the left and right singular vectors
    of :math:`\mathbf{X}^{(i)} \mathbf{\Delta}^\mathsf{T}`.

    Then we update the coordinate matrix by

    .. math::

        \mathbf{\Delta}
        = \frac{1}{\sum_{i=1}^I \rho_i}\sum_{i=1}^I \rho_i {\mathbf{P}^{(i)}}^\mathsf{T}\mathbf{X}^{(i)},

    where :math:`\rho_i` is the feasibility penalty (which parameterizes the norm of the projection)
    for the :math:`i`-th factor matrix.

    Parameters
    ----------
    svd : str
        String that specifies which SVD algorithm to use. Valid strings are the keys of ``tensorly.SVD_FUNS``.
    n_iter : int
        Number of iterations for the coordinate descent scheme
    aux_init : {"random_uniform", "random_standard_normal", "zeros", tl.tensor(ndim=2), list of tl.tensor(ndim=2)}
        Initialisation method for the auxiliary variables
    dual_init : {"random_uniform", "random_standard_normal", "zeros", tl.tensor(ndim=2), list of tl.tensor(ndim=2)}
        Initialisation method for the auxiliary variables
    """

    def __init__(
        self,
        svd="truncated_svd",
        n_iter=1,
        update_basis_matrices=True,
        update_coordinate_matrix=True,
        aux_init="random_uniform",
        dual_init="random_uniform",
    ):
        self.svd = svd
        self.aux_init = aux_init
        self.dual_init = dual_init
        self.update_basis_matrices = update_basis_matrices
        self.update_coordinate_matrix = update_coordinate_matrix
        self.n_iter = n_iter

    @property
    def svd_fun(self):
        return get_svd(self.svd)

    def init_aux(self, matrices, rank, mode, random_state=None):
        r"""Initialize the auxiliary variables

        For all initialization schemes, the orthogonal basis matrices are initialized
        using the first :math:`R` rows of an identity matrix.

        **Coordinate matrix initialization schemes**

         * ``"random_uniform"``: The elements of the coordinate matrix
            are drawn from a uniform distribution between 0 and 1.
         * ``"random_standard_normal"``: The elements of the coordinate matrix
            are drawn from a standard normal distribution.
         * ``"zeros"``: The elements of the coordinate matrix are
            initialized as zero.
         * tl.tensor(ndim=2) : Pre-computed coordinate matrix (mode=0 or mode=2)
         * list of tl.tensor(ndim=2): Pre-computed coordinate matrix (mode=1)

        Parameters
        ----------
        matrices : list of tensor(ndim=2) or tensor(ndim=3)
            The data matrices represented by the coupled matrix factorization
            these auxiliary variables correspond to.
        rank : int
            Rank of the decomposition.
        mode : int
            The mode represented by the factor matrices that these
            auxiliary variables correspond to.
        random_state : RandomState
            TensorLy random state.

        Returns
        -------
        tuple
            Tuple whose first element is the :math:`R \times R` coordinate matrix
            and second element is the list of the orthogonal basis matrices.
        """
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
            coordinate_matrix = tl.tensor(random_state.uniform(size=(rank, rank)))
            basis_matrices = [tl.eye(M.shape[0], rank) for M in matrices]

            return basis_matrices, coordinate_matrix
        if self.aux_init == "random_standard_normal":
            coordinate_matrix = tl.tensor(random_state.standard_normal(size=(rank, rank)))
            basis_matrices = [tl.eye(M.shape[0], rank) for M in matrices]

            return basis_matrices, coordinate_matrix
        if self.aux_init == "zeros":
            coordinate_matrix = tl.zeros((rank, rank))
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

    @copy_ancestor_docstring
    def factor_matrices_update(self, factor_matrices, feasibility_penalties, auxes):
        basis_matrices, coordinate_matrix = auxes
        R = tl.shape(coordinate_matrix)[0]

        for it in range(self.n_iter):
            # Update orthogonal basis matrices
            if self.update_basis_matrices:
                basis_matrices = []  # To prevent inplace editing of basis matrices
                for fm in factor_matrices:
                    U, s, Vh = self.svd_fun(tl.matmul(fm, tl.transpose(coordinate_matrix)), n_eigenvecs=R)
                    basis_matrices.append(tl.matmul(U, Vh))

            if self.update_coordinate_matrix:
                # Project all factor matrices onto the space spanned by the orthogonal basis matrices
                # and compute weighted mean
                coordinate_matrix = 0
                for fm, basis_matrix, feasibility_penalty in zip(
                    factor_matrices, basis_matrices, feasibility_penalties
                ):
                    coordinate_matrix += feasibility_penalty * basis_matrix.T @ fm
                coordinate_matrix /= sum(feasibility_penalties)

            if (not self.update_coordinate_matrix) or (not self.update_basis_matrices):
                break

        return basis_matrices, coordinate_matrix

    def subtract_from_aux(self, aux, dual):
        """Raises TypeError since the PARAFAC2 constraint only works with mode=1."""
        raise TypeError("The PARAFAC2 constraint cannot shift a single factor matrix.")

    def subtract_from_auxes(self, auxes, duals):
        r"""Compute (aux - dual) for each auxiliary- and dual-factor matrix for mode=1.

        Computing the difference between the auxiliary variables and the dual variables
        is an essential part of ADMM. However, the auxiliary variables is not a
        list of factor matrices but rather a coordinate matrix and a collection of
        orthogonal basis matrices, so this difference cannot be computed by simply subtracting
        one from the other. First auxiliary matrices must be computed by multiplying
        the basis matrices with the coordinate matrix and then the difference can
        be computed.

        Parameters
        ----------
        auxes : tuple
            Tuple whose first element is the :math:`R \times R` coordinate matrix
            and second element is the list of the orthogonal basis matrices.
        duals : list of tl.tensor(ndim=2)
            Dual variables (or other variable to subtract from the auxes)

        Returns
        -------
        list of tl.tensor(ndim=2)
            The list of differences
        """
        P_is, coord_mat = auxes
        return [tl.dot(P_i, coord_mat) - dual for P_i, dual in zip(P_is, duals)]

    def aux_as_matrix(self, aux):
        """Raises TypeError since the PARAFAC2 constraint only works with mode=1."""
        raise TypeError("The PARAFAC2 constraint cannot convert a single aux to a matrix")

    def auxes_as_matrices(self, auxes):
        """Convert a the auxiliary variables into a list of matrices (mode=1).

        This function computes the list of matrices parametrized by the coordinate
        matrix and orthogonal basis matrices by multiplying them together.

        Parameters
        ----------
        auxes : tuple
            Tuple whose first element is the :math:`R \times R` coordinate matrix
            and second element is the list of the orthogonal basis matrices.

        Returns
        -------
        list of tl.tensor(ndim=2)
        """
        P_is, coord_mat = auxes
        return [tl.dot(P_i, coord_mat) for P_i in P_is]

    def penalty(self, x):
        """Returns 0 as there is no penalty for hard constraints.

        Hard constraints are always penalised with 0 even when the components are infeasible.
        Slightly infeasible components would otherwise result in an infinite penalty because
        the penalty function of hard constraints is 0 for feasible solutions and infinity for
        infeasible solutions. An infinite penalty would stop all convergence checking and not
        provide any information on the quality of the components. To ensure that the hard
        constraints are sufficiently imposed, it is recommended to examine the feasibility gap
        instead of the penalty and ensure that the feasibility gap is low.

        Parameters
        ----------
        x : list of tl.tensor(ndim=2)
            List of factor matrices.
        """
        if not isinstance(x, list):
            raise TypeError("Cannot compute PARAFAC2 penalty of other types than a list of tensors")
        return 0
