# MIT License: Copyright (c) 2022, Marie Roald.
# See the LICENSE file in the root directory for full license text.

from copy import copy
from typing import NamedTuple, Optional

import numpy as np
import tensorly as tl
import tensorly.decomposition

from . import penalties
from ._utils import create_padded_tensor, get_shapes, get_svd, is_iterable
from .coupled_matrices import CoupledMatrixFactorization, cmf_to_matrices

__all__ = ["compute_feasibility_gaps", "ADMMVars", "DiagnosticMetrics", "cmf_aoadmm", "parafac2_aoadmm"]


def initialize_cmf(matrices, rank, init, svd_fun, random_state=None, init_params=None):
    random_state = tl.check_random_state(random_state)

    # Start by checking if init is a valid factorization. If so, use it
    if isinstance(init, (tuple, list, CoupledMatrixFactorization)):
        weights, (A, B_is, C) = init
        if weights is not None:
            scaled_A = weights * A
            return CoupledMatrixFactorization((None, (scaled_A, B_is, C)))
        else:
            return CoupledMatrixFactorization(init)

    # Random uniform initialisation
    if init == "random":
        I = len(matrices)
        K = tl.shape(matrices[0])[1]

        A = tl.tensor(random_state.uniform(size=(I, rank)))
        C = tl.tensor(random_state.uniform(size=(K, rank)))
        B_is = [tl.tensor(random_state.uniform(size=(tl.shape(matrix)[0], rank))) for matrix in matrices]

        return CoupledMatrixFactorization((None, [A, B_is, C]))

    # SVD and thresholded SVD initialisation
    if init == "svd" or init == "threshold_svd":
        I = len(matrices)
        A = tl.ones((I, rank))
        B_is = [svd_fun(matrix, n_eigenvecs=rank)[0] for matrix in matrices]
        C = tl.transpose(svd_fun(tl.concatenate(matrices, 0), n_eigenvecs=rank)[2])
        if init == "svd":
            return CoupledMatrixFactorization((None, [A, B_is, C]))

        A = tl.ones((I, rank))
        B_is = [tl.clip(B_i, 0, float("inf")) for B_i in B_is]
        C = tl.clip(C, 0, float("inf"))
        return CoupledMatrixFactorization((None, [A, B_is, C]))

    # PARAFAC and PARAFAC2 initialisation:
    if init_params is None:
        init_params = {}
    if "n_iter_max" not in init_params:
        init_params["n_iter_max"] = 50

    if init == "parafac2_als":
        pf2 = tl.decomposition.parafac2(matrices, rank, **init_params, random_state=random_state)
        return CoupledMatrixFactorization.from_Parafac2Tensor(pf2)

    # PARAFAC init: Work with padded tensor
    tensor = create_padded_tensor(matrices)
    shapes = get_shapes(matrices)
    if init == "cp_als" or init == "parafac_als":
        cp = tl.decomposition.parafac(tensor, rank, **init_params, random_state=random_state)
        return CoupledMatrixFactorization.from_CPTensor(cp, shapes=shapes)
    if init == "cp_hals" or init == "parafac_hals":
        cp = tl.decomposition.non_negative_parafac_hals(tensor, rank, **init_params, random_state=random_state)
        return CoupledMatrixFactorization.from_CPTensor(cp, shapes=shapes)

    raise ValueError('Initialization method "{}" not recognized'.format(init))


def initialize_aux(matrices, rank, reg, random_state):
    A_aux_list = [A_reg.init_aux(matrices, rank, 0, random_state=random_state) for A_reg in reg[0]]
    B_aux_list = [B_reg.init_aux(matrices, rank, 1, random_state=random_state) for B_reg in reg[1]]
    C_aux_list = [C_reg.init_aux(matrices, rank, 2, random_state=random_state) for C_reg in reg[2]]
    return A_aux_list, B_aux_list, C_aux_list


def initialize_dual(matrices, rank, reg, random_state):
    A_dual_list = [A_reg.init_dual(matrices, rank, 0, random_state=random_state) for A_reg in reg[0]]
    B_dual_list = [B_reg.init_dual(matrices, rank, 1, random_state=random_state) for B_reg in reg[1]]
    C_dual_list = [C_reg.init_dual(matrices, rank, 2, random_state=random_state) for C_reg in reg[2]]
    return A_dual_list, B_dual_list, C_dual_list


def _check_inner_convergence(factor_matrix, old_factor_matrix, cmf, reg_list, aux_list, mode, inner_tol):
    if not inner_tol or inner_tol < 0:
        return False

    if mode == 1:
        norm = _root_sum_squared_list(factor_matrix)
        change = _root_sum_squared_list([B_i - prev_B_i for B_i, prev_B_i in zip(factor_matrix, old_factor_matrix)])
    else:
        norm = tl.norm(factor_matrix)
        change = tl.norm(factor_matrix - old_factor_matrix)

    if change > inner_tol * norm:
        return False

    if len(reg_list) == 0:
        return True

    # Create regs-list and auxes-list so only feasibility gaps for current mode is computed
    regs = [[], [], []]
    regs[mode] = reg_list
    auxes = [[], [], []]
    auxes[mode] = aux_list
    gaps = compute_feasibility_gaps(cmf, regs, *auxes)[mode]

    return max(gaps) < inner_tol


# TODO (Improvement): Add option to scale the l2_penalty based on size (mostly relevant for B)
def admm_update_A(
    matrices,
    reg,
    cmf,
    A_aux_list,
    A_dual_list,
    l2_penalty,
    inner_n_iter_max,
    inner_tol,
    feasibility_penalty_scale,
    constant_feasibility_penalty,
    svd_fun,
):
    weights, (A, B_is, C) = cmf

    # Compute lhs and rhs
    lhses = []
    rhses = []
    CtC = tl.dot(tl.transpose(C), C)
    for matrix, B_i in zip(matrices, B_is):
        BtX = tl.dot(tl.transpose(B_i), matrix)
        rhses.append(tl.diag(tl.dot(BtX, C)))
        lhses.append(tl.dot(tl.transpose(B_i), B_i) * CtC)

    # Multiply with 0.5 since this function minimizes 0.5||Ax - b||^2
    # while in the PARAFAC2 AO-ADMM paper ||Ax - b||^2 is minimzed
    feasibility_penalties = [0.5 * tl.trace(lhs) * feasibility_penalty_scale for lhs in lhses]
    if constant_feasibility_penalty:
        max_feasibility_penalty = max(feasibility_penalties)
        feasibility_penalties = [max_feasibility_penalty for _ in feasibility_penalties]

    lhses = [
        lhs + tl.eye(tl.shape(A)[1]) * (feasibility_penalty * len(reg) + l2_penalty)
        for lhs, feasibility_penalty in zip(lhses, feasibility_penalties)
    ]
    svds = [svd_fun(lhs) for lhs in lhses]

    old_A = tl.copy(A)
    for inner_it in range(inner_n_iter_max):
        old_A, A = A, old_A

        shifted_A_auxes = [
            single_reg.subtract_from_aux(A_aux, A_dual)
            for single_reg, A_aux, A_dual in zip(reg, A_aux_list, A_dual_list)
        ]

        for i, matrix in enumerate(matrices):
            U, s, Uh = svds[i]

            sum_shifted_aux_A_row = 0
            for shifted_A_aux in shifted_A_auxes:
                sum_shifted_aux_A_row += shifted_A_aux[i]

            tl.index_update(
                A,
                tl.index[i, :],
                tl.dot(tl.dot(feasibility_penalties[i] * sum_shifted_aux_A_row + rhses[i], U / s), Uh),
            )

        for reg_num, single_reg in enumerate(reg):
            A_aux = A_aux_list[reg_num]
            A_dual = A_dual_list[reg_num]
            shifted_A = A + A_dual

            if constant_feasibility_penalty:
                A_aux_list[reg_num] = single_reg.factor_matrix_update(shifted_A, max_feasibility_penalty, A_aux)

            else:
                for i, feasibility_penalty in enumerate(feasibility_penalties):
                    tl.index_update(
                        A_aux,
                        tl.index[i, :],
                        single_reg.factor_matrix_row_update(shifted_A[i], feasibility_penalty, A_aux[i]),
                    )
            shifted_A_auxes[reg_num] = single_reg.subtract_from_aux(A_aux_list[reg_num], A_dual)
            A_dual_list[reg_num] = A - shifted_A_auxes[reg_num]  # A - (A_aux - A_dual) = A - A_aux + A_dual

        cmf = weights, (A, B_is, C)
        if _check_inner_convergence(A, old_A, cmf, reg, A_aux_list, mode=0, inner_tol=inner_tol):
            break

    return (None, [A, B_is, C]), A_aux_list, A_dual_list


def admm_update_B(
    matrices,
    reg,
    cmf,
    B_is_aux_list,
    B_is_dual_list,
    l2_penalty,
    inner_n_iter_max,
    inner_tol,
    feasibility_penalty_scale,
    constant_feasibility_penalty,
    svd_fun,
):
    weights, (A, B_is, C) = cmf

    # Compute lhs and rhs
    lhses = []
    rhses = []
    CtC = tl.dot(tl.transpose(C), C)
    for matrix, a_row in zip(matrices, A):
        rhses.append(tl.dot(matrix, C * a_row))
        lhses.append(tl.transpose(tl.transpose(CtC * a_row) * a_row))

    # Multiply with 0.5 since this function minimizes 0.5||Ax - b||^2
    # while in the PARAFAC2 AO-ADMM paper ||Ax - b||^2 is minimzed
    feasibility_penalties = [0.5 * tl.trace(lhs) * feasibility_penalty_scale for lhs in lhses]
    if constant_feasibility_penalty:
        max_feasibility_penalty = max(feasibility_penalties)
        feasibility_penalties = [max_feasibility_penalty for _ in feasibility_penalties]

    lhses = [
        lhs + tl.eye(tl.shape(A)[1]) * (feasibility_penalty * len(reg) + l2_penalty)
        for lhs, feasibility_penalty in zip(lhses, feasibility_penalties)
    ]
    svds = [svd_fun(lhs) for lhs in lhses]

    old_B_is = [tl.copy(B_i) for B_i in B_is]
    for inner_it in range(inner_n_iter_max):
        old_B_is, B_is = B_is, old_B_is
        shifted_auxes_B_is = [
            single_reg.subtract_from_auxes(B_is_aux, B_is_dual)
            for single_reg, B_is_aux, B_is_dual in zip(reg, B_is_aux_list, B_is_dual_list)
        ]

        for i, matrix in enumerate(matrices):
            U, s, Uh = svds[i]

            sum_shifted_aux_B_i = 0
            for shifted_aux_B_is in shifted_auxes_B_is:
                sum_shifted_aux_B_i += shifted_aux_B_is[i]  # B_is_aux[i] - B_is_dual[i]

            B_is[i] = tl.dot(tl.dot(feasibility_penalties[i] * sum_shifted_aux_B_i + rhses[i], U / s), Uh)

        for reg_num, single_reg in enumerate(reg):
            B_is_aux = B_is_aux_list[reg_num]
            B_is_dual = B_is_dual_list[reg_num]
            shifted_B_is = [B_i + B_i_dual for B_i, B_i_dual in zip(B_is, B_is_dual)]

            B_is_aux_list[reg_num] = single_reg.factor_matrices_update(shifted_B_is, feasibility_penalties, B_is_aux)

            shifted_auxes_B_is[reg_num] = single_reg.subtract_from_auxes(B_is_aux_list[reg_num], B_is_dual)
            B_is_dual_list[reg_num] = [
                B_i - shifted_aux_B_i for B_i, shifted_aux_B_i in zip(B_is, shifted_auxes_B_is[reg_num])
            ]

        cmf = weights, (A, B_is, C)

        if _check_inner_convergence(B_is, old_B_is, cmf, reg, B_is_aux_list, mode=1, inner_tol=inner_tol):
            break

    return (None, [A, B_is, C]), B_is_aux_list, B_is_dual_list


def admm_update_C(
    matrices,
    reg,
    cmf,
    C_aux_list,
    C_dual_list,
    l2_penalty,
    inner_n_iter_max,
    inner_tol,
    feasibility_penalty_scale,
    svd_fun,
):
    weights, (A, B_is, C) = cmf

    # Compute lhs and rhs
    lhs = 0
    rhs = 0
    for matrix, B_i, A_i in zip(matrices, B_is, A):
        B_iA_i = B_i * A_i
        lhs += tl.dot(tl.transpose(B_iA_i), B_iA_i)
        rhs += tl.dot(tl.transpose(matrix), B_iA_i)

    # Multiply with 0.5 since this function minimizes 0.5||Ax - b||^2
    # while in the PARAFAC2 AO-ADMM paper ||Ax - b||^2 is minimzed
    feasibility_penalty = 0.5 * tl.trace(lhs) * feasibility_penalty_scale
    lhs += tl.eye(tl.shape(C)[1]) * (feasibility_penalty * len(reg) + l2_penalty)
    U, s, Uh = svd_fun(lhs)

    old_C = tl.copy(C)
    # ADMM iterations
    for inner_it in range(inner_n_iter_max):
        old_C, C = C, old_C

        sum_shifted_aux_C = 0
        for single_reg, C_aux, C_dual in zip(reg, C_aux_list, C_dual_list):
            sum_shifted_aux_C += single_reg.subtract_from_aux(C_aux, C_dual)
        C = tl.dot(tl.dot(sum_shifted_aux_C * feasibility_penalty + rhs, U / s), Uh)

        for reg_num, single_reg in enumerate(reg):
            C_aux = C_aux_list[reg_num]
            C_dual = C_dual_list[reg_num]

            C_aux_list[reg_num] = single_reg.factor_matrix_update(C + C_dual, feasibility_penalty, C_aux)
            C_dual_list[reg_num] = C - single_reg.subtract_from_aux(C_aux_list[reg_num], C_dual)

        cmf = weights, (A, B_is, C)
        if _check_inner_convergence(C, old_C, cmf, reg, C_aux_list, mode=2, inner_tol=inner_tol):
            break

    return (None, [A, B_is, C]), C_aux_list, C_dual_list


def _root_sum_squared_list(x_list):
    return tl.sqrt(sum(tl.sum(x ** 2) for x in x_list))


def compute_feasibility_gaps(cmf, regs, A_aux_list, B_aux_list, C_aux_list):
    r"""Compute all feasibility gaps.

    The feasibility gaps for AO-ADMM are given by

    .. math::

        \frac{\|\mathbf{x} - \mathbf{z}\|_2}{\|\mathbf{z}\|_2},

    where :math:`\mathbf{x}` is a component vector and :math:`\mathbf{z}` is an auxiliary
    variable that represents a component vector. If a decomposition obtained with AO-ADMM
    is valid, then all feasibility gaps should be small compared to the components. If any
    of the feasibility penalties are large, then the decomposition may not satisfy the
    imposed constraints.

    To compute the feasibility gap for the :math:`\mathbf{A}` and :math:`\mathbf{C}`
    matrices, we use the frobenius norm, and to compute the feasibility gap for the
    :math:`\mathbf{B}^{(i)}`-matrices, we compute :math:`\sqrt{\sum_i \|\mathbf{B}^{(i)} - \mathbf{Z}^{(\mathbf{B}^{(i)})}\|^2}`,
    where :math:`\mathbf{Z}^{(\mathbf{B}^{(i)})}\|^2` is the auxiliary variable for
    :math:`\mathbf{B}^{(i)}`.

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

    regs : list of list of matcouply.penalties.ADMMPenalty
        The regs list should have three elements, the first being a list of penalties imposed
        on mode 0, the second being a list of penalties imposed on mode 1 and the last being
        a list of penalties imposed on mode 2.
    A_aux_list : list
        A list of all auxiliary variables for the A-matrix
    B_aux_list : list
        A list of all auxiliary variables for the B_is-matrices
    C_aux_list : list
        A list of all auxiliary variables for the C-matrix

    Returns
    -------
    list
        Feasibility gaps for A
    list
        Feasibility gaps for B_is
    list
        Veasibility gaps for C
    """
    weights, (A, B_is, C) = cmf

    A_norm = tl.norm(A)
    B_norm = _root_sum_squared_list(B_is)
    C_norm = tl.norm(C)

    A_gaps = [tl.norm(A_reg.subtract_from_aux(A_aux, A)) / A_norm for A_reg, A_aux in zip(regs[0], A_aux_list)]
    B_gaps = [
        _root_sum_squared_list(B_reg.subtract_from_auxes(B_is_aux, B_is)) / B_norm
        for B_reg, B_is_aux in zip(regs[1], B_aux_list)
    ]
    C_gaps = [tl.norm(C_reg.subtract_from_aux(C_aux, C)) / C_norm for C_reg, C_aux in zip(regs[2], C_aux_list)]

    return A_gaps, B_gaps, C_gaps


def _cmf_reconstruction_error(matrices, cmf):
    estimated_matrices = cmf_to_matrices(cmf, validate=False)
    return _root_sum_squared_list([X - Xhat for X, Xhat in zip(matrices, estimated_matrices)])


def _listify(input_value, param_name):
    if hasattr(input_value, "get"):
        return [input_value.get(i, None) for i in range(3)]
    elif not is_iterable(input_value):
        return [input_value] * 3
    else:
        out = list(input_value)
        if not len(out) == 3:
            raise ValueError(
                "All parameters must be a dictionary, non-iterable value or non-dictionary iterable of length 3."
                f" {param_name} is iterable of length {len(out)}."
            )
        return out


def _parse_all_penalties(
    non_negative,
    lower_bound,
    upper_bound,
    l2_norm_bound,
    unimodal,
    parafac2,
    l1_penalty,
    tv_penalty,
    generalized_l2_penalty,
    svd,
    regs,
    dual_init,
    aux_init,
    verbose,
):
    if regs is None:
        regs = [[], [], []]
    elif is_iterable(regs):
        for modereg in regs:
            if not is_iterable(modereg):
                raise TypeError(
                    "regs should contain an iterable of iterables containting "
                    "matcouply.penalties.ADMMMPenalty instances at least one of the"
                    f"elements in regs were not iterable (regs={regs})"
                )
            else:
                for reg in modereg:
                    if not isinstance(reg, penalties.ADMMPenalty):
                        raise TypeError(
                            "regs should contain an iterable of iterables containting "
                            "matcouply.penalties.ADMMMPenalty instances at least one of the"
                            f"elements in regs contained something other than an ADMMPenalty (regs={regs})"
                        )

    regs = [copy(reg_list) for reg_list in regs]  # To avoid side effects where the input lists are modified

    non_negative = _listify(non_negative, "non_negative")
    upper_bound = _listify(upper_bound, "upper_bound")
    lower_bound = _listify(lower_bound, "lower_bound")
    l2_norm_bound = _listify(l2_norm_bound, "l2_norm_bound")
    unimodal = _listify(unimodal, "unimodal")
    if parafac2:
        parafac2 = [False, True, False]
    else:
        parafac2 = [False, False, False]
    l1_penalty = _listify(l1_penalty, "l1_penalty")
    generalized_l2_penalty = _listify(generalized_l2_penalty, "generalized_l2_penalty")
    tv_penalty = _listify(tv_penalty, "tv_penalty")

    for mode in range(3):
        parsed_regs = _parse_mode_penalties(
            non_negative=non_negative[mode],
            lower_bound=lower_bound[mode],
            upper_bound=upper_bound[mode],
            l2_norm_bound=l2_norm_bound[mode],
            unimodal=unimodal[mode],
            parafac2=parafac2[mode],
            l1_penalty=l1_penalty[mode],
            tv_penalty=tv_penalty[mode],
            generalized_l2_penalty=generalized_l2_penalty[mode],
            svd=svd,
            dual_init=dual_init,
            aux_init=aux_init,
        )

        regs[mode] = parsed_regs + regs[mode]

    if verbose:
        print(f"All regularization penalties (including regs list):")
        for mode, reg in enumerate(regs):
            print(f"* Mode {mode}:")
            if len(reg) == 0:
                print("   - (no regularization added)")
            for single_reg in reg:
                print(f"   - {single_reg}")
    return regs


def _parse_mode_penalties(
    non_negative,
    lower_bound,
    upper_bound,
    l2_norm_bound,
    unimodal,
    parafac2,
    l1_penalty,
    tv_penalty,
    generalized_l2_penalty,
    svd,
    dual_init,
    aux_init,
):

    if not l1_penalty:
        l1_penalty = 0

    regs = []

    skip_non_negative = False

    if parafac2:
        regs.append(penalties.Parafac2(svd=svd, aux_init=aux_init, dual_init=dual_init))

    if unimodal:
        regs.append(penalties.Unimodality(non_negativity=non_negative, aux_init=aux_init, dual_init=dual_init))
        skip_non_negative = True

    if (
        generalized_l2_penalty is not None and generalized_l2_penalty is not False
    ):  # generalized_l2_penalty is None or matrix
        regs.append(
            penalties.GeneralizedL2Penalty(generalized_l2_penalty, aux_init=aux_init, dual_init=dual_init, svd=svd)
        )

    if l2_norm_bound:
        regs.append(
            penalties.L2Ball(l2_norm_bound, non_negativity=non_negative, aux_init=aux_init, dual_init=dual_init)
        )
        skip_non_negative = True

    if tv_penalty:
        regs.append(
            penalties.TotalVariationPenalty(tv_penalty, l1_strength=l1_penalty, aux_init=aux_init, dual_init=dual_init)
        )
        l1_penalty = 0  # Already included in the total variation penalty

    if l1_penalty:
        regs.append(
            penalties.L1Penalty(l1_penalty, non_negativity=non_negative, aux_init=aux_init, dual_init=dual_init)
        )
        skip_non_negative = True

    if lower_bound is not None or upper_bound is not None:
        if lower_bound is None:
            lower_bound = -float("inf")
        if non_negative:
            lower_bound = max(lower_bound, 0)
        regs.append(penalties.BoxConstraint(lower_bound, upper_bound, aux_init=aux_init, dual_init=dual_init))
        skip_non_negative = True

    if non_negative and not skip_non_negative:
        regs.append(penalties.NonNegativity(aux_init=aux_init, dual_init=dual_init))

    return regs


def _compute_l2_penalty(cmf, l2_parameters):
    weights, (A, B_is, C) = cmf
    l2reg = 0
    if l2_parameters[0]:
        l2reg += 0.5 * l2_parameters[0] * tl.sum(A ** 2)
    if l2_parameters[1]:
        l2reg += 0.5 * l2_parameters[1] * sum(tl.sum(B_i ** 2) for B_i in B_is)
    if l2_parameters[2]:
        l2reg += 0.5 * l2_parameters[2] * tl.sum(C ** 2)

    return l2reg


def _check_feasibility(feasibility_gaps, feasibility_tol):
    A_gaps, B_gaps, C_gaps = feasibility_gaps
    max_feasibility_gap = -float("inf")
    if len(A_gaps):
        max_feasibility_gap = max((max(A_gaps), max_feasibility_gap))
    if len(B_gaps):
        max_feasibility_gap = max((max(B_gaps), max_feasibility_gap))
    if len(C_gaps):
        max_feasibility_gap = max((max(C_gaps), max_feasibility_gap))

    return max_feasibility_gap < feasibility_tol


class ADMMVars(NamedTuple):
    auxes: tuple  #: Length three tuple containing a list of auxiliary factor matrices for each mode
    duals: tuple  #: Length three tuple containing a list of dual variables for each mode


class DiagnosticMetrics(NamedTuple):
    rec_errors: list  #: List of length equal to the number of iterations plus one containing the reconstruction errors
    feasibility_gaps: list  #: List of length equal to the number of iterations plus one containing the feasibility gaps
    regularized_loss: list  #: List of length equal to the number of iterations plus one containing the regularized loss
    satisfied_stopping_condition: Optional[
        bool
    ]  #: Boolean specifying whether the stopping conditions were satisfied, None if no tolerance is set
    satisfied_feasibility_condition: Optional[
        bool
    ]  #: Boolean specifying whether the feasibility conditions were satisfied, None if no tolerance is set
    n_iter: int  #: Number of iterations ran
    message: str  #: Convergence message


def cmf_aoadmm(
    matrices,
    rank,
    init="random",
    n_iter_max=1000,
    l2_penalty=None,
    tv_penalty=None,
    l1_penalty=None,
    non_negative=None,
    unimodal=None,
    generalized_l2_penalty=None,
    l2_norm_bound=None,
    lower_bound=None,
    upper_bound=None,
    parafac2=None,
    regs=None,
    feasibility_penalty_scale=1,
    constant_feasibility_penalty=False,
    aux_init="random_uniform",
    dual_init="random_uniform",
    svd="truncated_svd",
    init_params=None,
    random_state=None,
    tol=1e-8,
    absolute_tol=1e-10,
    feasibility_tol=1e-4,
    inner_tol=None,
    inner_n_iter_max=5,
    update_A=True,
    update_B_is=True,
    update_C=True,
    return_admm_vars=False,
    return_errors=False,
    verbose=False,
):
    r"""Fit a regularized coupled matrix factorization model with AO-ADMM

    Regularization parameters can be:

     * ``None`` (no regularization)
     * A single regularization parameter (same regularization in all modes)
     * A list with specific regularization parameters (one for each mode)
     * A dictionary with mode-index (0, 1 or 2) as keys and regularization parameter as values
       (regularization only in specific modes)

    :math:`\mathbf{A}` is represented by mode-index 0, :math:`\{\mathbf{B}^{(i)}\}_{i=1}^I` is
    represented by mode-index 1 and :math:`\mathbf{C}` is represented by mode-index 2.

    Parameters
    ----------
    matrices : list of ndarray
        Matrices that the coupled matrix factorization should model
    rank : int
        The rank of the model
    init : {"random", "svd", "threshold_svd", "parafac_als", "parafac_hals", "parafac2_als"} (default="random")
        Initialization method. If equal to ``"parafac_als"``, ``"parafac_hals"`` or ``"parafac2_als"``,
        then the corresponding methods in TensorLy are used to initialize the model. In that case,
        you can pass additional keyword-arguments to the decomposition function with the ``init_params``
        parameter.
    n_iter_max : int (default=1000)
        Maximum number of iterations.
    l2_penalty : Regularization parameter (default=None)
        Strength of the L2 penalty, imposed as ``0.5 * l2_penalty * tl.sum(M**2)``, where ``M``
        represents a single factor matrix (note that this differs by a factor :math:`0.5` compared
        to the expression in :cite:p:`roald2021parafac2,roald2021admm`).
    tv_penalty : Regularization parameter (default=None)
        Strength of the TV penalty. To use this regularizer, you must have the GPL-lisenced library:
        ``condat_tv`` installed.
    l1_penalty : Regularization parameter (default=None)
        Strength of the sparsity inducing regularization
    non_negative : Regularization parameter (default=None)
        If True, then the corresponding factor matrices are non-negative
    unimodal : Regularization parameter (default=None)
        If True, then the corresponding factor matrices have unimodal columns
    generalized_l2_penalty : Regularization parameter (must be ``None``, list or dict) (default=None)
        List or dict containing square matrices that parametrize a generalized L2 norm.
    l2_norm_bound : Regularization parameter (default=None)
        Maximum L2 norm of the columns
    lower_bound : Regularization parameter (default=None)
        Lower value of the columns
    upper_bound : Regularization parameter (default=None)
        Upper value of the columns
    parafac2 : bool (default=False)
        If ``True``, then the :math:`\mathbf{B}^{(i)}`-matrices follow the PARAFAC2 constraint
    regs : List of lists of matcouply.penalties.ADMMPenalty (optional, default=None)
        The first element of this list contains a list of ``matcouply.penalties.ADMMPenalty``-instances
        for :math:`\mathbf{A}`, the second for :math:`\{\mathbf{B}^{(i)}\}_{i=1}^I` and the third for
        :math:`\mathbf{C}`.
    feasibility_penalty_scale : float (default=1)
        What function to multiply the automatically computed feasibility penalty parameter by
        (see :cite:p:`roald2021admm` for details)
    constant_feasibility_penalty : bool, "A" or "B" (default=False)
        If True, then all rows of :math:`\mathbf{A}` will have the same feasibility penalty parameter
        and all :math:`\mathbf{B}^{(i)}`-matrices will have the same feasibility penalty parameter.
        This makes it possible to use matrix-penalties for :math:`\mathbf{A}` and multi-matrix penalties
        that require the same penalty parameter for all :math:`\mathbf{B}^{(i)}`-matrices.

        The maximum penalty parameter for all rows of :math:`\mathbf{A}` are used as the penalty
        parameter for :math:`\mathbf{A}` and the maximum penalty parameter among all :math:`\mathbf{B}^{(i)}`-matrices
        are used as the penalty parameter for :math:`\{\mathbf{B}^{(i)}\}_{i=1}^I`.

        If ``"A"`` or ``"B"``, then this option is only enabled for the specified parameters.
    aux_init : str (default="random_uniform")
        Initialization scheme for the auxiliary variables. See :meth:`matcouply.penalties.ADMMPenalty.init_aux`
        for possible options.
    dual_init : str (default="random_uniform")
        Initialization scheme for the dual variables. See :meth:`matcouply.penalties.ADMMPenalty.init_dual`
        for possible options.
    svd : str (default="truncated_svd")
        String that specifies which SVD algorithm to use. Valid strings are the keys of ``tensorly.SVD_FUNS``.
    init_params : dict
        Keyword arguments to use for the initialization.
    random_state : None, int or valid tensorly random state
    tol : float (default=1e-8)
        Relative loss decrease condition. For stopping, the optimization requires
        that ``abs(losses[-2] - losses[-1]) / losses[-2] < tol`` or that
        ``losses[-2] < absolute_tol``.
    absolute_tol : float (default=1e-10)
        Absolute loss value condition. For stopping, the optimization requires
        that ``abs(losses[-2] - losses[-1]) / losses[-2] < tol`` or that
        ``losses[-2] < absolute_tol``.
    feasibility_tol : float (default=1e-4)
        If any feasibility gap is greater than feasibility_tol, then the optimization
        will not stop (unless the maximum number of iterations are reached).
    inner_tol : float (optional, default=None)
        If set, then this specifies the stopping condition for the inner subproblems,
        solved with ADMM. If it is None, then the ADMM algorithm will always run for
        ``inner_n_iter_max``-iterations.
    inner_n_iter_max : int (default=5)
        Maximum number of iterations for the ADMM subproblems
    update_A : bool (default=True)
        If ``False``, then :math:`\mathbf{A}` will not be updated.
    update_B_is : bool (default=True)
        If ``False``, then the :math:`\mathbf{B}^{(i)}`-matrices will not be updated.
    update_C : bool (default=True)
        If ``False``, then :math:`\mathbf{C}` will not be updated.
    return_admm_vars : bool (default=False)
        If ``True``, then the auxiliary and dual variables will also be returned.
    return_errors : bool (default=False)
        If ``True``, then the reconstruction error, feasibility gaps and loss per iteration
        will be returned.
    verbose : bool or int (default=False)
        If ``True``, then a message with convergence info will be printed out every iteration.
        If ``int > 1``, then a message with convergence info will be printed out ever ``verbose`` iteration.

    Returns
    -------
    CoupledMatrixFactorization
        The fitted coupled matrix factorization
    ADMMVars
        (only returned if ``return_admm_vars=True``) NamedTuple containing the auxuliary
        and dual variables. The feasibility gaps are computed by taking the relative normed
        differences between the auxiliary variables and the factor matrices of the model.
    DiagnosticMetrics
        (only returned if ``return_errors=True``) NamedTuple containing lists of the relative
        norm-error, the feasibility gaps and the regularized loss for each iteration, each with
        length equal to the number of iterations plus one (the initial values), a stopping
        message and two boolean values, one signifying whether the stopping conditions (including
        feasibility gap) were satisfied and one signifying whether the feasibility gaps were
        sufficiently low (according to ``feasibility_tol``).

    Note
    ----
    If the maximum number of iterations is reached, then you should check that the feasibility
    gaps are sufficiently small (see :doc:`../optimization` for more details)

    Note
    ----
    If you use norm-dependent regularization (e.g. ``generalized_l2_penalty``) in one mode,
    then you must use norm-dependent regularization in all modes. You may, for example, use
    ``l2_penalty``, ``norm_bound``, ``l1_penalty`` or ``lower_bound`` and ``upper_bound``.
    See :cite:p:`roald2021admm` for more details.

    Note
    ----
    For simplicity, the model used here is a permuted version of that in
    :cite:p:`roald2021parafac2,roald2021admm` (where :math:`\mathbf{B_k}`-matrices
    vary over the rows in :math:`\mathbf{C}` instead of the rows in :math:`\mathbf{A}`).

    Examples
    --------
    Here is a small example that shows what the diagnostic metrics contain. For more detailed examples, see :ref:`examples`

    >>> import matcouply
    >>> data = matcouply.random.random_coupled_matrices([(10, 5), (12, 5), (11, 5)], rank=2, full=True, random_state=1)
    >>> cmf, diagnostics = matcouply.decomposition.cmf_aoadmm(data, 2, n_iter_max=4, non_negative=True, return_errors=True, random_state=10)
    >>> diagnostics.message
    'MAXIMUM NUMBER OF ITERATIONS REACHED'

    >>> diagnostics.satisfied_stopping_condition
    False

    >>> diagnostics.satisfied_feasibility_condition
    True

    >>> len(diagnostics.regularized_loss)
    5

    We see that the length of the regularized loss list is the number of iterations plus one. This is because the initial
    decomposition is included in this.

    >>> len(diagnostics.feasibility_gaps[0])
    3

    The feasibility gaps list contain tuples (one for each mode) of tuples (one for each penalty for the given mode).

    >>> len(diagnostics.feasibility_gaps[0][0])
    1
    """
    random_state = tl.check_random_state(random_state)
    svd_fun = get_svd(svd)
    cmf = initialize_cmf(matrices, rank, init, svd_fun=svd_fun, random_state=random_state, init_params=init_params)

    # Parse constraints
    l2_penalty = _listify(l2_penalty, "l2_penalty")
    l2_penalty = [l2 if l2 is not None else 0 for l2 in l2_penalty]

    regs = _parse_all_penalties(
        non_negative=non_negative,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        l2_norm_bound=l2_norm_bound,
        unimodal=unimodal,
        parafac2=parafac2,
        l1_penalty=l1_penalty,
        tv_penalty=tv_penalty,
        generalized_l2_penalty=generalized_l2_penalty,
        svd=svd,
        regs=regs,
        dual_init=dual_init,
        aux_init=aux_init,
        verbose=verbose,
    )

    if not update_A:
        regs[0] = []
    if not update_B_is:
        regs[1] = []
    if not update_C:
        regs[2] = []

    # TODO  (Improvement): Include cmf to initialize functions in case other init schemes require that?
    (A_aux_list, B_is_aux_list, C_aux_list,) = initialize_aux(matrices, rank, regs, random_state=random_state)
    (A_dual_list, B_is_dual_list, C_dual_list,) = initialize_dual(matrices, rank, regs, random_state=random_state)
    norm_matrices = _root_sum_squared_list(matrices)
    rec_errors = []
    feasibility_gaps = []

    rec_error = _cmf_reconstruction_error(matrices, cmf)
    rec_error /= norm_matrices
    rec_errors.append(rec_error)
    losses = []
    reg_penalty = (
        +sum(A_reg.penalty(cmf[1][0]) for A_reg in regs[0])
        + sum(B_reg.penalty(cmf[1][1]) for B_reg in regs[1])
        + sum(C_reg.penalty(cmf[1][2]) for C_reg in regs[2])
    )
    l2_reg = _compute_l2_penalty(cmf, l2_penalty)
    losses.append(0.5 * rec_error ** 2 + l2_reg + reg_penalty)

    A_gaps, B_gaps, C_gaps = compute_feasibility_gaps(cmf, regs, A_aux_list, B_is_aux_list, C_aux_list)
    feasibility_gaps.append((A_gaps, B_gaps, C_gaps))
    if verbose and verbose > 0:
        print("Feasibility gaps for A: {}".format(A_gaps))
        print("Feasibility gaps for the Bi-matrices: {}".format(B_gaps))
        print("Feasibility gaps for C: {}".format(C_gaps))

    # Default values for diagnostics
    satisfied_stopping_condition = False
    message = "MAXIMUM NUMBER OF ITERATIONS REACHED"
    if isinstance(constant_feasibility_penalty, str) and constant_feasibility_penalty not in {"A", "B"}:
        raise ValueError(
            f"If `constant_feasibility_penalty` is a string, it must be 'A' or 'B', not {constant_feasibility_penalty}"
        )
    constant_A_feasibility = (
        constant_feasibility_penalty and not isinstance(constant_feasibility_penalty, str)
    ) or constant_feasibility_penalty == "A"
    constant_B_feasibility = (
        constant_feasibility_penalty and not isinstance(constant_feasibility_penalty, str)
    ) or constant_feasibility_penalty == "B"

    it = -1  # Needed if n_iter_max <= 0
    for it in range(n_iter_max):
        if update_B_is:
            cmf, B_is_aux_list, B_is_dual_list = admm_update_B(
                matrices=matrices,
                reg=regs[1],
                cmf=cmf,
                B_is_aux_list=B_is_aux_list,
                B_is_dual_list=B_is_dual_list,
                l2_penalty=l2_penalty[1],
                inner_n_iter_max=inner_n_iter_max,
                inner_tol=inner_tol,
                feasibility_penalty_scale=feasibility_penalty_scale,
                constant_feasibility_penalty=constant_B_feasibility,
                svd_fun=svd_fun,
            )
        if update_A:
            cmf, A_aux_list, A_dual_list = admm_update_A(
                matrices=matrices,
                reg=regs[0],
                cmf=cmf,
                A_aux_list=A_aux_list,
                A_dual_list=A_dual_list,
                l2_penalty=l2_penalty[0],
                inner_n_iter_max=inner_n_iter_max,
                inner_tol=inner_tol,
                feasibility_penalty_scale=feasibility_penalty_scale,
                constant_feasibility_penalty=constant_A_feasibility,
                svd_fun=svd_fun,
            )
        if update_C:
            cmf, C_aux_list, C_dual_list = admm_update_C(
                matrices=matrices,
                reg=regs[2],
                cmf=cmf,
                C_aux_list=C_aux_list,
                C_dual_list=C_dual_list,
                l2_penalty=l2_penalty[2],
                inner_n_iter_max=inner_n_iter_max,
                inner_tol=inner_tol,
                feasibility_penalty_scale=feasibility_penalty_scale,
                svd_fun=svd_fun,
            )

        if tol or absolute_tol or return_errors:
            curr_feasibility_gaps = compute_feasibility_gaps(cmf, regs, A_aux_list, B_is_aux_list, C_aux_list)
            feasibility_gaps.append(curr_feasibility_gaps)

            if tol or absolute_tol:
                # Compute stopping criterions
                feasibility_criterion = feasibility_tol and _check_feasibility(curr_feasibility_gaps, feasibility_tol)

                if not feasibility_criterion and not return_errors:
                    if verbose and it % verbose == 0 and verbose > 0:
                        print(
                            "Coupled matrix factorization iteration={}, ".format(it)
                            + "reconstruction error=NOT COMPUTED, "
                            + "regularized loss=NOT COMPUTED, "
                            + "regularized loss variation=NOT COMPUTED."
                        )
                        print("Feasibility gaps for A: {}".format(A_gaps))
                        print("Feasibility gaps for the Bi-matrices: {}".format(B_gaps))
                        print("Feasibility gaps for C: {}".format(C_gaps))

                    continue

            rec_error = _cmf_reconstruction_error(matrices, cmf)
            rec_error /= norm_matrices
            rec_errors.append(rec_error)
            reg_penalty = (
                +sum(A_reg.penalty(cmf[1][0]) for A_reg in regs[0])
                + sum(B_reg.penalty(cmf[1][1]) for B_reg in regs[1])
                + sum(C_reg.penalty(cmf[1][2]) for C_reg in regs[2])
            )

            l2_reg = _compute_l2_penalty(cmf, l2_penalty)
            losses.append(0.5 * rec_error ** 2 + l2_reg + reg_penalty)

            if verbose and it % verbose == 0 and verbose > 0:
                A_gaps, B_gaps, C_gaps = curr_feasibility_gaps
                print(
                    "Coupled matrix factorization iteration={}, ".format(it)
                    + "reconstruction error={}, ".format(rec_errors[-1])
                    + "regularized loss={} ".format(losses[-1])
                    + "regularized loss variation={}.".format(abs(losses[-2] - losses[-1]) / losses[-2])
                )
                print("Feasibility gaps for A: {}".format(A_gaps))
                print("Feasibility gaps for the Bi-matrices: {}".format(B_gaps))
                print("Feasibility gaps for C: {}".format(C_gaps))

            if tol:
                # Compute rest of stopping criterions
                rel_loss_criterion = abs(losses[-2] - losses[-1]) < (tol * losses[-2])
                abs_loss_criterion = losses[-1] < absolute_tol

                if feasibility_criterion and rel_loss_criterion:
                    satisfied_stopping_condition = True
                    message = "FEASIBILITY GAP CRITERION AND RELATIVE LOSS CRITERION SATISFIED"
                    if verbose:
                        print("converged in {} iterations: {}".format(it, message))
                    break
                elif feasibility_criterion and abs_loss_criterion:
                    satisfied_stopping_condition = True
                    message = "FEASIBILITY GAP CRITERION AND ABSOLUTE LOSS CRITERION SATISFIED"
                    if verbose:
                        print("converged in {} iterations: {}".format(it, message))
                    break

        elif verbose and it % verbose == 0 and verbose > 0:
            print("Coupled matrix factorization iteration={}".format(it))
    else:
        if verbose:
            print("REACHED MAXIMUM NUMBER OF ITERATIONS")

    # Compute feasibility gaps to return with diagnostics
    # If the feasibility tolerance is set, but no loss tolerance then the feasibility criterion is not
    # computed in the AO-ADMM loop. Likewise, if n_iter_max <= 0, the feasibility tolerance
    # is not computed. This is only relevant if we should return errors, but it is a fast operation
    # so we do it anyways.
    if feasibility_tol and return_errors:
        curr_feasibility_gaps = compute_feasibility_gaps(cmf, regs, A_aux_list, B_is_aux_list, C_aux_list)

        feasibility_criterion = _check_feasibility(curr_feasibility_gaps, feasibility_tol)
    elif not feasibility_tol:
        feasibility_criterion = None

    # Save as validated factorization instead of tuple
    cmf = CoupledMatrixFactorization(cmf)

    out = [cmf]
    if return_admm_vars:
        admm_vars = ADMMVars(
            auxes=(A_aux_list, B_is_aux_list, C_aux_list), duals=(A_dual_list, B_is_dual_list, C_dual_list)
        )
        out.append(admm_vars)
    if return_errors:
        if not satisfied_stopping_condition and not (tol or absolute_tol):
            satisfied_stopping_condition = None

        diagnostic_metrics = DiagnosticMetrics(
            rec_errors=rec_errors,
            feasibility_gaps=feasibility_gaps,
            regularized_loss=losses,
            satisfied_stopping_condition=satisfied_stopping_condition,
            satisfied_feasibility_condition=feasibility_criterion,
            message=message,
            n_iter=it + 1,  # Plus one since this is the number of iterations, not the iteration number
        )
        out.append(diagnostic_metrics)

    if len(out) == 1:
        return out[0]
    else:
        return tuple(out)


def parafac2_aoadmm(
    matrices,
    rank,
    init="random",
    n_iter_max=1000,
    l2_penalty=0,
    tv_penalty=None,
    l1_penalty=None,
    non_negative=None,
    unimodal=None,
    generalized_l2_penalty=None,
    l2_norm_bound=None,
    lower_bound=None,
    upper_bound=None,
    regs=None,
    feasibility_penalty_scale=1,
    constant_feasibility_penalty=False,
    aux_init="random_uniform",
    dual_init="random_uniform",
    svd="truncated_svd",
    init_params=None,
    random_state=None,
    tol=1e-8,
    absolute_tol=1e-10,
    feasibility_tol=1e-4,
    inner_tol=None,
    inner_n_iter_max=5,
    update_A=True,
    update_B_is=True,
    update_C=True,
    return_errors=False,
    return_admm_vars=False,
    verbose=False,
):
    """Alias for cmf_aoadmm with PARAFAC2 constraint (constant cross-product) on mode 1 (B mode)

    See also
    --------
    matcouply.decomposition.cmf_aoadmm : General coupled matrix factorization with AO-ADMM
    matcouply.penalties.Parafac2 : Class for PARAFAC2 constraint with more information about its properties
    """

    return cmf_aoadmm(
        matrices=matrices,
        rank=rank,
        init=init,
        n_iter_max=n_iter_max,
        l2_penalty=l2_penalty,
        tv_penalty=tv_penalty,
        l1_penalty=l1_penalty,
        non_negative=non_negative,
        unimodal=unimodal,
        generalized_l2_penalty=generalized_l2_penalty,
        l2_norm_bound=l2_norm_bound,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        parafac2=True,
        regs=regs,
        feasibility_penalty_scale=feasibility_penalty_scale,
        constant_feasibility_penalty=constant_feasibility_penalty,
        aux_init=aux_init,
        dual_init=dual_init,
        svd=svd,
        init_params=init_params,
        random_state=random_state,
        tol=tol,
        absolute_tol=absolute_tol,
        feasibility_tol=feasibility_tol,
        inner_tol=inner_tol,
        inner_n_iter_max=inner_n_iter_max,
        update_A=update_A,
        update_B_is=update_B_is,
        update_C=update_C,
        return_errors=return_errors,
        return_admm_vars=return_admm_vars,
        verbose=verbose,
    )
