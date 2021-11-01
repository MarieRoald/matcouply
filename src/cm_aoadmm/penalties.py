from abc import ABC, abstractmethod

import tensorly as tl

from ._utils import get_svd
# TODO: Maybe remove compute_feasibility_gap and only use shift_aux
# TODO: Maybe rename shift_aux to subtract_from_aux
# TODO: Maybe add mixin classes for some of the functionality
# TODO: For all penalties with __init__, make sure they call super().__init__ and have the aux_init and dual_init arguments
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
        # TODO: Not provided random state
        if self.aux_init == "random_uniform":
            if mode == 0:
                return random_state.uniform(size=(len(matrices), rank))
            elif mode == 1:
                return [random_state.uniform(size=(matrix.shape[0], rank)) for matrix in matrices]
            elif mode == 2:
                return random_state.uniform(size=(matrices[0].shape[1], rank))
            else:
                raise ValueError("Mode must be 0, 1, or 2.")
        elif self.aux_init == "random_standard_normal": # TODO: test
            if mode == 0:
                return random_state.standard_normal(size=(len(matrices), rank))
            elif mode == 1:
                return [random_state.standard_normal(size=(matrix.shape[0], rank)) for matrix in matrices]
            elif mode == 2:
                return random_state.standard_normal(size=(matrices[0].shape[1], rank))
            else:
                raise ValueError("Mode must be 0, 1, or 2.")
        else:
            raise ValueError(f"Unknown aux init: {self.aux_init}")

    def init_dual(self, matrices, rank, mode, random_state=None):
        # TODO: Not provided random state
        if self.dual_init == "random_uniform":
            if mode == 0:
                return random_state.uniform(size=(len(matrices), rank))
            elif mode == 1:
                return [random_state.uniform(size=(matrix.shape[0], rank)) for matrix in matrices]
            elif mode == 2:
                return random_state.uniform(size=(matrices[0].shape[1], rank))
            else:
                raise ValueError("Mode must be 0, 1, or 2.")
        elif self.dual_init == "random_standard_normal": # TODO: test
            if mode == 0:
                return random_state.standard_normal(size=(len(matrices), rank))
            elif mode == 1:
                return [random_state.standard_normal(size=(matrix.shape[0], rank)) for matrix in matrices]
            elif mode == 2:
                return random_state.standard_normal(size=(matrices[0].shape[1], rank))
            else:
                raise ValueError("Mode must be 0, 1, or 2.")
        elif self.dual_init == "zeros": # TODO: test
            if mode == 0:
                return tl.zeros(size=(len(matrices), rank))
            elif mode == 1:
                return [tl.zeros(size=(matrix.shape[0], rank)) for matrix in matrices]
            elif mode == 2:
                return tl.zeros(size=(matrices[0].shape[1], rank))
            else:
                raise ValueError("Mode must be 0, 1, or 2.")
        else:
            raise ValueError(f"Unknown dual init: {self.aux_init}")

    @abstractmethod
    def penalty(self, x):  # TODO: How to deal with penalties that go across matrices
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
        raise NotImplementedError


class MatrixPenalty(ADMMPenalty):
    def factor_matrices_update(self, factor_matrices, feasibility_penalties, auxes):
        return [
            self.factor_matrix_update(fm, feasibility_penalty, aux)
            for fm, feasibility_penalty, aux in zip(factor_matrices, feasibility_penalties, auxes)
        ]

    @abstractmethod
    def factor_matrix_update(self, factor_matrix, feasibility_penalty, aux):
        raise NotImplementedError


class MatricesPenalty(ADMMPenalty):
    @abstractmethod
    def factor_matrices_update(self, factor_matrices, feasibility_penalties, auxes):
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
    def __init__(self, reg_strength, non_negativity=False):
        self.reg_strength = reg_strength
        self.non_negativity = non_negativity

    def factor_matrix_row_update(self, factor_matrix_row, feasibility_penalty, aux_row):
        if not self.non_negativity:
            sign = tl.sign(factor_matrix_row)
            return sign * tl.clip(tl.abs(factor_matrix_row) - self.reg_strength/feasibility_penalty, 0)
        else:
            return tl.clip(factor_matrix_row - self.reg_strength/feasibility_penalty, 0)

    def factor_matrix_update(self, factor_matrix, feasibility_penalty, aux):
        if not self.non_negativity:
            sign = tl.sign(factor_matrix)
            return sign * tl.clip(tl.abs(factor_matrix) - self.reg_strength/feasibility_penalty, 0)
        else:
            return tl.clip(factor_matrix - self.reg_strength/feasibility_penalty, 0)

    def penalty(self, x):
        # TODO: return reg_strength*l1norm of x
        if tl.is_tensor(x):
            return tl.sum(tl.abs(x))*self.reg_strength
        else:
            return sum(tl.sum(tl.abs(xi)) for xi in x)*self.reg_strength


class Parafac2(MatricesPenalty):
    def __init__(self, svd="truncated_svd", aux_init="random_uniform", dual_init="random_uniform"):
        self.svd_fun = get_svd(svd)
        self.aux_init = aux_init
        self.dual_init = dual_init

    def init_aux(self, matrices, rank, mode, random_state=None):
        # TODO: Not provided random state
        if self.aux_init == "random_uniform":
            if mode != 1:
                raise ValueError("PARAFAC2 constraint can only be imposed with mode=1")
            else:
                coordinate_matrix = random_state.uniform(size=(rank, rank))
                basis_matrices = [tl.eye(M.shape[0], rank) for M in matrices]

                return basis_matrices, coordinate_matrix
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
        return 0
