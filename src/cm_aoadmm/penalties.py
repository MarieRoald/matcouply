import tensorly as tl
# TODO: Maybe remove compute_feasibility_gap and only use shift_aux
# TODO: Maybe rename shift_aux to subtract_from_aux
# TODO: Maybe add mixin classes for some of the functionality

TensorType = type(tl.zeros(2))  # TODO: Add TensorType to tensorly


class ADMMPenalty:
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

    def penalty(self, x):
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
        return [
            self.factor_matrix_update(fm, feasibility_penalty, aux)
            for fm, feasibility_penalty, aux in zip(factor_matrices, feasibility_penalties, auxes)
        ]

    def factor_matrix_update(self, factor_matrix, feasibility_penalty, aux):
        out = tl.empty(factor_matrix.shape)

        for row, factor_matrix_row in enumerate(factor_matrix):
            out[row] = self.factor_matrix_row_update(factor_matrix_row, feasibility_penalty, aux[row])
        
        return out

    def factor_matrix_row_update(self, factor_matrix_row, feasibility_penalty, aux_row):
        raise NotImplementedError


class MatrixPenalty(ADMMPenalty):        
    def factor_matrices_update(self, factor_matrices, feasibility_penalties, auxes):
        return [
            self.factor_matrix_update(fm, feasibility_penalty, aux)
            for fm, feasibility_penalty, aux in zip(factor_matrices, feasibility_penalties, auxes)
        ]

    def factor_matrix_update(self, factor_matrix, feasibility_penalty, aux):
        raise NotImplementedError


class NonNegativity(RowVectorPenalty):
    def factor_matrix_row_update(self, factor_matrix_row, feasibility_penalty, aux_row):
        return tl.clip(factor_matrix_row, 0)

    def factor_matrix_update(self, factor_matrix, feasibility_penalty, aux):
        # Return elementwise maximum of zero and factor_matrix
        return tl.clip(factor_matrix, 0)

    def penalty(self, x):
        return 0


# TODO: unit tests
class BoxConstraint(RowVectorPenalty):
    def __init__(self, min_val, max_val):
        self.min_val = min_val
        self.max_val = max_val
    def factor_matrix_row_update(self, factor_matrix_row, feasibility_penalty, aux_row):
        return tl.clip(factor_matrix_row, self.min_val, self.max_val)

    def factor_matrix_update(self, factor_matrix, feasibility_penalty, aux):
        return tl.clip(factor_matrix, self.min_val, self.max_val)

    def penalty(self, x):
        return 0



class L1Penalty(RowVectorPenalty):
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
        if isinstance(x, TensorType):
            return tl.sum(tl.abs(x))*self.reg_strength
        else:
            return sum(tl.sum(tl.abs(xi)) for xi in x)*self.reg_strength


class Parafac2(ADMMPenalty):
    def __init__(self, svd=None, svd_fun=None, aux_init="random_uniform", dual_init="random_uniform"):
        if svd is None and svd_fun is None:
            raise ValueError("Either the svd method or a function to compute the svd must be provided")
        self.svd_fun = svd_fun
        self.aux_init = aux_init
        self.dual_init = dual_init
        # TODO: Parse svd similar to cmf_aoadmm

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
            U, s, Vh = self.svd_fun(fm @ coord_mat.T, full_matrices=False)
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
