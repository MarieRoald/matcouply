import numpy as np
import tensorly as tl

from . import penalties
from ._utils import get_svd, is_iterable
from .coupled_matrices import CoupledMatrixFactorization, cmf_to_matrices

# TODO: Document all update steps, they might be slightly different from paper (e.g. new transposes)
# TODO: Document l2_penalty as 0.5||A||^2, etc. Not ||A||^2


def initialize_cmf(matrices, rank, init, svd_fun, random_state=None, init_params=None):
    random_state = tl.check_random_state(random_state)
    if init_params is None:
        init_params = {}
    if init == "random":
        I = len(matrices)
        K = tl.shape(matrices[0])[1]

        A = random_state.uniform(size=(I, rank))
        C = random_state.uniform(size=(K, rank))
        B_is = [random_state.uniform(size=(matrix.shape[0], rank)) for matrix in matrices]

        return CoupledMatrixFactorization((None, [A, B_is, C]))
    elif init == "svd":
        # TODO: TEST SVD init
        I = len(matrices)
        A = tl.ones((I, rank))
        B_is = [svd_fun(matrix, n_eigenvecs=rank)[0] for matrix in matrices]
        C = tl.transpose(svd_fun(tl.concatenate(matrices, 0), n_eigenvecs=rank)[2])
        return CoupledMatrixFactorization((None, [A, B_is, C]))
    elif init == "threshold_svd":
        # TODO: TEST Thresholded SVD init
        # TODO: Before thresholding: fix SVD sign
        I = len(matrices)
        A = tl.ones((I, rank))
        B_is = [tl.clip(svd_fun(matrix, n_eigenvecs=rank)[0], 0) for matrix in matrices]
        C = tl.clip(tl.transpose(svd_fun(tl.concatenate(matrices, 0), n_eigenvecs=rank)[2]), 0)
        return CoupledMatrixFactorization((None, [A, B_is, C]))
    elif init == "parafac2_als":
        # TODO: PARAFAC2 init
        pass
    elif init == "cp_als":
        # TODO: CP init
        pass
    elif isinstance(init, (tuple, list, CoupledMatrixFactorization)):
        return CoupledMatrixFactorization(init)

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


# TODO: For all updates: add l2_penalty. When computing svd, add 0.5*l2_penalty*identity_matrix
# TODO: Document the loss function we are optinmising, l2_penalty half its value in the paper
# TODO: Add option to scale the l2_penalty
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

    feasibility_penalties = [tl.trace(lhs) * feasibility_penalty_scale for lhs in lhses]
    if constant_feasibility_penalty:
        max_feasibility_penalty = max(feasibility_penalties)
        feasibility_penalties = [max_feasibility_penalty for _ in feasibility_penalties]

    lhses = [
        lhs + tl.eye(tl.shape(A)[1]) * 0.5 * (feasibility_penalty * len(reg) + l2_penalty)
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
                tl.dot(tl.dot(0.5 * feasibility_penalties[i] * sum_shifted_aux_A_row + rhses[i], U / s), Uh),
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

        if inner_tol:
            A_norm = tl.norm(A)
            A_change = tl.norm(A - old_A)
            A_gaps, _, _ = compute_feasibility_gaps(cmf, [reg, [], []], A_aux_list, [], [])
            A_gaps = [
                A_gap / tl.norm(single_reg.aux_as_matrix(A_aux))
                for single_reg, A_gap, A_aux in zip(reg, A_gaps, A_aux_list)
            ]

            dual_residual_criterion = len(A_gaps) == 0 or max(A_gaps) < inner_tol
            primal_residual_criterion = A_change < inner_tol * A_norm

            if primal_residual_criterion and dual_residual_criterion:
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

    feasibility_penalties = [tl.trace(lhs) * feasibility_penalty_scale for lhs in lhses]
    if constant_feasibility_penalty:
        max_feasibility_penalty = max(feasibility_penalties)
        feasibility_penalties = [max_feasibility_penalty for _ in feasibility_penalties]

    lhses = [
        lhs + tl.eye(tl.shape(A)[1]) * 0.5 * (feasibility_penalty * len(reg) + l2_penalty)
        for lhs, feasibility_penalty in zip(lhses, feasibility_penalties)
    ]
    svds = [svd_fun(lhs) for lhs in lhses]

    old_B_is = [B_i.copy() for B_i in B_is]
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

            B_is[i] = tl.dot(tl.dot(0.5 * feasibility_penalties[i] * sum_shifted_aux_B_i + rhses[i], U / s), Uh)

        for reg_num, single_reg in enumerate(reg):
            B_is_aux = B_is_aux_list[reg_num]
            B_is_dual = B_is_dual_list[reg_num]
            shifted_B_is = [B_i + B_i_dual for B_i, B_i_dual in zip(B_is, B_is_dual)]

            B_is_aux_list[reg_num] = single_reg.factor_matrices_update(shifted_B_is, feasibility_penalties, B_is_aux)

            shifted_auxes_B_is[reg_num] = single_reg.subtract_from_auxes(B_is_aux_list[reg_num], B_is_dual)
            B_is_dual_list[reg_num] = [
                B_i - shifted_aux_B_i for B_i, shifted_aux_B_i in zip(B_is, shifted_auxes_B_is[reg_num])
            ]

        if inner_tol:
            # TODO: How to deal with feasibility gaps for all B_is?
            #   Currently we "stack" each B_is and compute their norm
            B_is_norm = _root_sum_squared_list(B_is)
            B_is_change = _root_sum_squared_list([B_i - prev_B_i for B_i, prev_B_i in zip(B_is, old_B_is)])
            _, B_gaps, _ = compute_feasibility_gaps(cmf, [[], reg, []], [], B_is_aux_list, [])
            B_gaps = [
                B_gap / _root_sum_squared_list(single_reg.auxes_as_matrices(B_is_aux))
                for single_reg, B_gap, B_is_aux in zip(reg, B_gaps, B_is_aux_list)
            ]

            dual_residual_criterion = len(B_gaps) == 0 or max(B_gaps) < inner_tol
            primal_residual_criterion = B_is_change < inner_tol * B_is_norm

            if primal_residual_criterion and dual_residual_criterion:
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
    svd_fun,  # TODO: Fix svd_fun for all
):
    weights, (A, B_is, C) = cmf

    # Compute lhs and rhs
    lhs = 0
    rhs = 0
    for matrix, B_i, A_i in zip(matrices, B_is, A):
        B_iA_i = B_i * A_i
        lhs += tl.dot(tl.transpose(B_iA_i), B_iA_i)
        rhs += tl.dot(tl.transpose(matrix), B_iA_i)

    feasibility_penalty = tl.trace(lhs) * feasibility_penalty_scale
    lhs += tl.eye(tl.shape(C)[1]) * 0.5 * (feasibility_penalty * len(reg) + l2_penalty)
    U, s, Uh = svd_fun(lhs)

    old_C = tl.copy(C)
    # ADMM iterations
    for inner_it in range(inner_n_iter_max):
        old_C, C = C, old_C

        sum_shifted_aux_C = 0
        for single_reg, C_aux, C_dual in zip(reg, C_aux_list, C_dual_list):
            sum_shifted_aux_C += single_reg.subtract_from_aux(C_aux, C_dual)
        C = tl.dot(tl.dot(sum_shifted_aux_C * 0.5 * feasibility_penalty + rhs, U / s), Uh)

        for reg_num, single_reg in enumerate(reg):
            C_aux = C_aux_list[reg_num]
            C_dual = C_dual_list[reg_num]

            C_aux_list[reg_num] = single_reg.factor_matrix_update(C + C_dual, feasibility_penalty, C_aux)
            C_dual_list[reg_num] = C - single_reg.subtract_from_aux(C_aux_list[reg_num], C_dual)

        # print("Inner iteration:", inner_it)
        # print("Feasibility penalty", feasibility_penalty)
        # print("C:", C)
        # print("C AUX:", C_aux_list[0])
        # print("C DUAL:", C_dual_list[0])

        if inner_tol:
            C_norm = tl.norm(C)
            C_change = tl.norm(C - old_C)
            _, _, C_gaps = compute_feasibility_gaps(cmf, [[], [], reg], [], [], C_aux_list)
            C_gaps = [
                C_gap / tl.norm(single_reg.aux_as_matrix(C_aux))
                for single_reg, C_gap, C_aux in zip(reg, C_gaps, C_aux_list)
            ]

            dual_residual_criterion = len(C_gaps) == 0 or max(C_gaps) < inner_tol
            primal_residual_criterion = C_change < inner_tol * C_norm

            if primal_residual_criterion and dual_residual_criterion:
                break

    return (None, [A, B_is, C]), C_aux_list, C_dual_list


def _root_sum_squared_list(x_list):
    return tl.sqrt(sum(tl.sum(x ** 2) for x in x_list))


def compute_feasibility_gaps(cmf, regs, A_aux_list, B_aux_list, C_aux_list):
    r"""Compute all feasibility gaps.

    The feasibility gaps for AO-ADMM are given by

    .. math::

        \|\mathbf{x} - \mathbf{z}\|_2,
    
    where :math:`\mathbf{x}` is a component vector and :math:`\mathbf{z}` is an auxiliary
    variable that represents a component vector. If a decomposition obtained with AO-ADMM
    is valid, then all feasibility gaps should be small compared to the components. If any
    of the feasibility penalties are large, then the decomposition may not satisfy the
    imposed constraints.

    To compute the feasibility gap for the :math:`\mathbf{A}` and :math:`\mathbf{C}`
    matrices, we use the frobenius norm, and to compute the feasibility gap for the
    :math:`B_{i}`-matrices, we compute :math:`\sqrt{\sum_i \|\mathbf{B}_i - \mathbf{Z}^{(\mathbf{B}_i)}\|^2}`,
    where :math:`\mathbf{Z}^{(\mathbf{B}_i)}\|^2` is the auxiliary variable for
    :math:`\mathbf{B}_i`.

    Parameters
    ----------
    cmf: CoupledMatrixFactorization - (weight, factors)

        * weights : 1D array of shape (rank, )
            weights of the factors
        * factors : List of factors of the coupled matrix decomposition
            Containts the matrices :math:`\mathbf{A}`, :math:`\mathbf{B}_i` and
            :math:`\mathbf{C}` described above

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
    A_gaps = [tl.norm(A_reg.subtract_from_aux(A_aux, A)) for A_reg, A_aux in zip(regs[0], A_aux_list)]
    B_gaps = [
        _root_sum_squared_list(B_reg.subtract_from_auxes(B_is_aux, B_is))
        for B_reg, B_is_aux in zip(regs[1], B_aux_list)
    ]
    C_gaps = [tl.norm(C_reg.subtract_from_aux(C_aux, C)) for C_reg, C_aux in zip(regs[2], C_aux_list)]

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
        parsed_regs, message = _parse_mode_penalties(
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
            print(f"Added mode {mode} penalties and constraints:{message}")
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
    if unimodal:
        raise NotImplementedError("Unimodality is not yet implemented")
    if not l1_penalty:
        l1_penalty = 0

    description_str = ""
    regs = []

    skip_non_negative = False

    if parafac2:
        regs.append(penalties.Parafac2(svd=svd, aux_init=aux_init, dual_init=dual_init))
        description_str += "\n * PARAFAC2"

    if (
        generalized_l2_penalty is not None and generalized_l2_penalty is not False
    ):  # generalized_l2_penalty is None or matrix
        regs.append(
            penalties.GeneralizedL2Penalty(generalized_l2_penalty, aux_init=aux_init, dual_init=dual_init, svd=svd)
        )
        description_str += "\n * Generalized L2 penalty"

    if l2_norm_bound:
        regs.append(
            penalties.L2Ball(l2_norm_bound, non_negativity=non_negative, aux_init=aux_init, dual_init=dual_init)
        )
        if non_negative:
            description_str += "\n * L2 ball constraint (with non-negativity)"
        else:
            description_str += "\n * L2 ball constraint"
        skip_non_negative = True

    if tv_penalty:
        regs.append(
            penalties.TotalVariationPenalty(tv_penalty, l1_strength=l1_penalty, aux_init=aux_init, dual_init=dual_init)
        )
        if l1_penalty:
            description_str += "\n * Total Variation penalty (with L1)"
        else:
            description_str += "\n * Total Variation penalty"
        l1_penalty = 0  # Already included in the total variation penalty

    if l1_penalty:
        regs.append(
            penalties.L1Penalty(l1_penalty, non_negativity=non_negative, aux_init=aux_init, dual_init=dual_init)
        )
        description_str += "\n * L1 penalty"
        skip_non_negative = True

    if lower_bound is not None or upper_bound is not None:
        if lower_bound is None:
            lower_bound = -float("inf")
        if non_negative:
            lower_bound = max(lower_bound, 0)
        regs.append(penalties.BoxConstraint(lower_bound, upper_bound, aux_init=aux_init, dual_init=dual_init))
        description_str += f"\n * Box constraints ({lower_bound} < x < {upper_bound})"
        skip_non_negative = True

    if non_negative and not skip_non_negative:
        regs.append(penalties.NonNegativity(aux_init=aux_init, dual_init=dual_init))
        description_str += "\n * Non negativity constraints"

    if len(description_str) == 0:
        description_str += "\n (no additional regularisation added)"
    return regs, description_str


# TODO: add alias for parafac2 decomposition


def cmf_aoadmm(
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
    return_errors=False,
    verbose=False,
):
    r"""Fit a regularized coupled matrix factorization model with AO-ADMM

    A coupled matrix factorization model decomposes a collection of matrices,
    :math:`\{\mathbf{X}_i\}_{i=1}^I` with :math:`\mathbf{X}_i \in \mathbb{R}^{J_i \times K}`
    into a sum of low rank components. An :math:`R`-component model is defined as

    .. math::

        \mathbf{X}_i \approx \mathbf{B}_i \mathbf{D}_i \mathbf{C}^\mathsf{T},

    where :math:`\mathbf{B}_i \in \mathbb{R}^{J_i \times R}` and
    :math:`\mathbf{C} \in \mathbb{R}^{K \times R}` are factor matrices and
    :math:`\mathbf{D}_k \in \mathbb{R}^{R \times R}` is a diagonal matrix. Here, we
    collect the diagonal entries in all :math:`\mathbf{D}_i`-matrices into a
    third factor matrix, :math:`\mathbf{A} \in \mathbb{R}^{I \times R}` whose :math:`i`-th row
    consists of the diagonal entries of :math:`\mathbf{D}_i`.

    Optimization problem
    ^^^^^^^^^^^^^^^^^^^^

    Fitting coupled matrix factorizations involve solving the following optimization problem

    .. math::

        \min_{\mathbf{A}, \{\mathbf{B}_i\}_{i=1}^I, \mathbf{C}}
        \sum_{i=1}^I \| \mathbf{B}_i \mathbf{D}_i \mathbf{C}^\mathsf{T} - \mathbf{X}_i\|^2.
    
    However, this problem does not have a unique solution, and each time we fit a coupled matrx
    factorization, we may obtain different factor matrices. As a consequence, we cannot interpret
    the factor matrices. To circumvent this problem, it is common to add regularisation, forming
    the following optimisation problem

    .. math::

        \min_{\mathbf{A}, \{\mathbf{B}_i\}_{i=1}^I, \mathbf{C}}
        \sum_{i=1}^I \| \mathbf{B}_i \mathbf{D}_i \mathbf{C}^\mathsf{T} - \mathbf{X}_i\|^2.
        + \sum_{n=1}^{N_\mathbf{A}} g^{(A)}_n(\mathbf{A})
        + \sum_{n=1}^{N_\mathbf{B}} g^{(B)}_n(\{ \mathbf{B}_i \}_{i=1}^I)
        + \sum_{n=1}^{N_\mathbf{C}} g^{(C)}_n(\mathbf{C}),

    where the :math:`g`-functions are regularisation penalties, and :math:`N_\mathbf{A}, N_\mathbf{B}`
    and :math:`N_\mathbf{C}` are the number of regularisation penalties for 
    :math:`\mathbf{A}, \{\mathbf{B}\}_{i=1}^I` and :math:`\mathbf{C}`, respectively.

    The formulation above also encompasses hard constraints, such as :math:`a_{ir} \geq 0` for
    any index :math:`(i, r)`. To obtain such a constraint, we set 
    
    .. math::
        
        g^{(\mathbf{A})} = \begin{cases}
            0 & \text{if } a_{ir} \geq 0 \text{ for all } a_{ir} \\
            \infty & \text{otherwise}.
        \end{cases}

    Optimization
    ^^^^^^^^^^^^

    To solve the regularized least squares problem, we use AO-ADMM. AO-ADMM is a block 
    coordinate descent scheme, where the factor matrices for each mode is updated in an
    alternating fashion. To update these factor matrices we use (a few) iterations of ADMM.

    The benefit of AO-ADMM is its flexibility in terms of regularization and constraints. We
    can impose any regularization penalty or hard constraint so long as we have a way to
    evaluate the scaled proximal operator of the penalty function[TODO: CITE] or projection
    onto the feasible set of the hard constraint.

    For more information about AO-ADMM see :ref:`optimization`.

    Uniqueness
    ^^^^^^^^^^

    Unfortunately, the coupled matrix factorization model is not unique. To see this,
    we stack the data matrices, :math:`\{\mathbf{X}_i\}_{i=1}^I` into one large matrix,
    :math:`\tilde{\mathbf{X}} \in \mathbb{R}^{\sum_{i=1}^I J_i \times K}`. A coupled
    matrix factorization model is then equivalent to factorizing this stacked
    :math:`\tilde{\mathbf{X}}`-matrix. And since matrix factorizations are not unique without
    additional constraints, we see that the coupled matrix factorization model is not unique
    either.

    The lack of uniqueness means that [TODO: THIS PARAGRAPH]

    Constraints
    ^^^^^^^^^^^

    To combat the problem of uniqueness, we can constrain the model. Common constraints are
    non-negativity [TODO: CITE], sparsity  [TODO: CITE] and the *PARAFAC2-constraint*  [TODO: CITE].


    **Non-negativity**

    Non-negativity constraints work by requiring that (some of) the factor matrices
    contain only non-negative entries. This is both helpful for uniqueness and interpretability
    of the model. However, while non-negativity constraints improve the uniqueness properties
    of the coupled matrix factorization model, such constraints do not ensure that the decomposition
    is truly unique. It is therefore also common to combine non-negativity constraints with
    other constraints, such as sparsity or the PARAFAC2 constraint.

    **Sparsity**

    Sparsity constraints are used when we want one or more of the factor matrices to be sparse.
    This also improves the interpretability of the model, as it becomes easy to see the variables
    that affect each component vector. Sparsity is often imposed via an L1 (also known as LASSO)
    penalty on one of the factor matrices. This penalty function has the form

    .. math::

        \|\mathbf{M}\|_1 = \sum_{n m} |m_{nm}|.

    **PARAFAC2**

    Another way to impose uniqueness is via the PARAFAC2 model [TODO: CITE]. This model is equivalent
    to the coupled matrix factorization model, but has the additional constraint

    .. math::

        \mathbf{B}_{i_1}^\mathsf{T} \mathbf{B}_{i_1} = \mathbf{B}_{i_2}^\mathsf{T} \mathbf{B}_{i_2}

    for all pairs :math:`i_1, i_2`. This constraint yields a unique decomposition when there
    are enough data matrices (:math:`\mathbf{X}_{i}`-s). [TODO: REFERENCE parafac2 function]

    Optimization
    ^^^^^^^^^^^^

    To fit the coupled matrix factorization, we use AO-ADMM. 
    """
    # TODO: docstring
    random_state = tl.check_random_state(random_state)
    svd_fun = get_svd(svd)
    cmf = initialize_cmf(matrices, rank, init, svd_fun=svd_fun, random_state=random_state, init_params=init_params)

    # Parse constraints
    l2_penalty = _listify(l2_penalty, "l2_penalty")  # TODO: Make test that checks that listify is called on l2_penalty
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

    A_aux_list, B_is_aux_list, C_aux_list, = initialize_aux(
        matrices, rank, regs, random_state=random_state
    )  # TODO: Include cmf?
    A_dual_list, B_is_dual_list, C_dual_list, = initialize_dual(
        matrices, rank, regs, random_state=random_state
    )  # TODO: Include cmf?
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
    losses.append(0.5 * rec_error + reg_penalty)

    A_gaps, B_gaps, C_gaps = compute_feasibility_gaps(cmf, regs, A_aux_list, B_is_aux_list, C_aux_list)
    feasibility_gaps.append((A_gaps, B_gaps, C_gaps))
    if verbose:
        print("Duality gaps for A: {}".format(A_gaps))
        print("Duality gaps for the Bi-matrices: {}".format(B_gaps))
        print("Duality gaps for C: {}".format(C_gaps))

    for it in range(n_iter_max):
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
            constant_feasibility_penalty=constant_feasibility_penalty,
            svd_fun=svd_fun,
        )
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
            constant_feasibility_penalty=constant_feasibility_penalty,
            svd_fun=svd_fun,
        )
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

        # TODO: Do we want normalisation?
        if tol or return_errors:
            A_gaps, B_gaps, C_gaps = compute_feasibility_gaps(cmf, regs, A_aux_list, B_is_aux_list, C_aux_list)
            feasibility_gaps.append((A_gaps, B_gaps, C_gaps))

            if tol:
                max_feasibility_gap = -np.inf
                if len(A_gaps):
                    max_feasibility_gap = max((max(A_gaps), max_feasibility_gap))
                if len(B_gaps):
                    max_feasibility_gap = max((max(B_gaps), max_feasibility_gap))
                if len(C_gaps):
                    max_feasibility_gap = max((max(C_gaps), max_feasibility_gap))
                # max_feasibility_gap = max(max(A_gaps), max(B_gaps), max(C_gaps))

                # Compute stopping criterions
                feasibility_criterion = max_feasibility_gap < feasibility_tol

                if not feasibility_criterion and not return_errors:
                    if verbose and it % verbose == 0:
                        print(
                            "Coupled matrix factorization iteration={}, ".format(it)
                            + "reconstruction error=NOT COMPUTED, "
                            + "regularised loss=NOT COMPUTED, "
                            + "squared relative variation=NOT COMPUTED."
                        )
                        print("Duality gaps for A: {}".format(A_gaps))
                        print("Duality gaps for the Bi-matrices: {}".format(B_gaps))
                        print("Duality gaps for C: {}".format(C_gaps))

                    continue

            # TODO: Maybe not always compute this (to save computation)?
            # TODO: Include the regularisation
            rec_error = _cmf_reconstruction_error(matrices, cmf)
            rec_error /= norm_matrices
            rec_errors.append(rec_error)
            reg_penalty = (
                +sum(A_reg.penalty(cmf[1][0]) for A_reg in regs[0])
                + sum(B_reg.penalty(cmf[1][1]) for B_reg in regs[1])
                + sum(C_reg.penalty(cmf[1][2]) for C_reg in regs[2])
            )
            losses.append(0.5 * rec_error ** 2 + reg_penalty)

            if verbose and it % verbose == 0:
                print(
                    "Coupled matrix factorization iteration={}, ".format(it)
                    + "reconstruction error={}, ".format(rec_errors[-1])
                    + "regularised loss={} ".format(losses[-1])
                    + "squared relative variation={}.".format(abs(losses[-2] - losses[-1]) / losses[-2])
                )
                print("Duality gaps for A: {}".format(A_gaps))
                print("Duality gaps for the Bi-matrices: {}".format(B_gaps))
                print("Duality gaps for C: {}".format(C_gaps))

            if tol:
                # Compute rest of stopping criterions
                rel_loss_criterion = abs(losses[-2] - losses[-1]) < (tol * losses[-2])
                abs_loss_criterion = losses[-1] < absolute_tol
                if feasibility_criterion and (rel_loss_criterion or abs_loss_criterion):
                    if verbose:
                        print("converged in {} iterations.".format(it))
                        # TODO: print information about what stopped it?
                    break
        elif verbose and it % verbose == 0:
            print("Coupled matrix factorization iteration={}".format(it))

    # Save as validated factorization instead of tuple
    cmf = CoupledMatrixFactorization(cmf)

    # TODO: Check return when only one constrain on B
    if return_errors:
        return (
            cmf,
            (A_aux_list, B_is_aux_list, C_aux_list),
            (A_dual_list, B_is_dual_list, C_dual_list),
            rec_errors,
            feasibility_gaps,
        )
    else:
        return (
            cmf,
            (A_aux_list, B_is_aux_list, C_aux_list),
            (A_dual_list, B_is_dual_list, C_dual_list),
        )


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
    return_errors=False,
    verbose=False,
):
    """Alias for cmf_aoadmm with PARAFAC2 constraint (constant cross-product) on mode 1 (B mode)  

    See also
    --------
    cmf_aoadmm : General coupled matrix factorization with AO-ADMM
    penalties.Parafac2 : Class for PARAFAC2 constraint with more information about its properties
    """  # FIXME: crossref penalties.Parafac2 correctly

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
        return_errors=return_errors,
        verbose=verbose,
    )
