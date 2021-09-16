import numpy as np
import tensorly as tl
from .coupled_matrices import CoupledMatrixFactorization, cmf_to_matrices
from ._utils import is_iterable

# TODO: Document all update steps, they might be slightly different from paper (e.g. new transposes)
# TODO: Document l2_penalty as 0.5||A||^2, etc. Not ||A||^2


# TODO: fast
def initialize_cmf(matrices, rank, init, svd_fun, random_state=None, init_params=None):
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
        # TODO: SVD init (A: Ones, Bi: singular vectors, C: svd of stacked matrix)
        pass
    elif init == "threshold_svd":
        # TODO: Thresholded SVD init - First SVD, then clip at zero
        pass
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

    feasibility_penalties = [np.trace(lhs) * feasibility_penalty_scale for lhs in lhses]
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
                A, tl.index[i, :], tl.dot(tl.dot(0.5 * feasibility_penalties[i] * sum_shifted_aux_A_row + rhses[i], U / s), Uh)
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
                        A_aux, tl.index[i, :], single_reg.factor_matrix_row_update(shifted_A[i], feasibility_penalty, A_aux[i])
                    )
            shifted_A_auxes[reg_num] = single_reg.subtract_from_aux(A_aux_list[reg_num], A_dual)
            A_dual_list[reg_num] = A - shifted_A_auxes[reg_num]  # A - (A_aux - A_dual) = A - A_aux + A_dual

        if inner_tol:
            A_norm = tl.norm(A)
            A_change = tl.norm(A - old_A)
            A_gaps, _, _ = compute_feasibility_gaps(cmf, [reg, [], []], A_aux_list, [], [])
            A_gaps = [A_gap / tl.norm(single_reg.aux_as_matrix(A_aux)) for single_reg, A_gap, A_aux in zip(reg, A_gaps, A_aux_list)]

            dual_residual_criterion = max(A_gaps) < inner_tol
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
    # TODO: implement B update
    weights, (A, B_is, C) = cmf

    # Compute lhs and rhs
    lhses = []
    rhses = []
    CtC = tl.dot(tl.transpose(C), C)
    for matrix, a_row in zip(matrices, A):
        rhses.append(tl.dot(matrix, C*a_row))
        lhses.append(tl.transpose(tl.transpose(CtC*a_row)*a_row))

    # TODO: trace in backends?
    feasibility_penalties = [np.trace(lhs) * feasibility_penalty_scale for lhs in lhses]
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
            # TODO: How to deal with feasibility gaps for all B_is? Currently we "stack" each B_is and compute their norm
            B_is_norm = _root_sum_squared_list(B_is)
            B_is_change = _root_sum_squared_list([B_i - prev_B_i for B_i, prev_B_i in zip(B_is, old_B_is)])
            _, B_gaps, _ = compute_feasibility_gaps(cmf, [[], reg, []], [], B_is_aux_list, [])
            B_gaps = [
                B_gap / _root_sum_squared_list(single_reg.auxes_as_matrices(B_is_aux))
                for single_reg, B_gap, B_is_aux in zip(reg, B_gaps, B_is_aux_list)
            ]

            dual_residual_criterion = max(B_gaps) < inner_tol
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

    # TODO: trace in backends?
    feasibility_penalty = np.trace(lhs) * feasibility_penalty_scale
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
            C_gaps = [C_gap / tl.norm(single_reg.aux_as_matrix(C_aux)) for single_reg, C_gap, C_aux in zip(reg, C_gaps, C_aux_list)]

            dual_residual_criterion = max(C_gaps) < inner_tol
            primal_residual_criterion = C_change < inner_tol * C_norm

            if primal_residual_criterion and dual_residual_criterion:
                break

    return (None, [A, B_is, C]), C_aux_list, C_dual_list


def _root_sum_squared_list(x_list):
    return tl.sqrt(sum(tl.sum(x ** 2) for x in x_list))


def compute_feasibility_gaps(cmf, reg, A_aux_list, B_aux_list, C_aux_list):
    # TODO: Docstring for compute_feasibility_gaps
    weights, (A, B_is, C) = cmf
    A_gaps = [tl.norm(A_reg.subtract_from_aux(A_aux, A)) for A_reg, A_aux in zip(reg[0], A_aux_list)]
    B_gaps = [
        _root_sum_squared_list(B_reg.subtract_from_auxes(B_is_aux, B_is))
        for B_reg, B_is_aux in zip(reg[1], B_aux_list)
    ]
    C_gaps = [tl.norm(C_reg.subtract_from_aux(C_aux, C)) for C_reg, C_aux in zip(reg[2], C_aux_list)]

    #C_gaps = [C_reg.compute_feasibility_gap(C, C_aux) for C_reg, C_aux in zip(reg[2], C_aux_list)]
    return A_gaps, B_gaps, C_gaps


def _cmf_reconstruction_error(matrices, cmf):
    estimated_matrices = cmf_to_matrices(cmf, validate=False)
    return _root_sum_squared_list([X - Xhat for X, Xhat in zip(matrices, estimated_matrices)])


def _get_svd(svd):
    if svd in tl.SVD_FUNS:
        return tl.SVD_FUNS[svd]
    else:
        message = "Got svd={}. However, for the current backend ({}), the possible choices are {}".format(
            svd, tl.get_backend(), tl.SVD_FUNS
        )
        raise ValueError(message)


def cmf_aoadmm(
    matrices,
    rank,
    init="random",
    n_iter_max=1000,
    l2_penalty=0,
    tv_penalty=None,
    l1_penalty=None,
    non_negativity=None,
    unimodality=None,
    # group_lasso_penalty=None,  # No reasonable default
    # group_lasso_groups,  # No reasonable default
    smoothness_penalty=None,  # SVD trick default
    l2_ball_constraint=None,
    l1_ball_constraint=None,  # Maybe implement?
    box_constraint=None,  # Maybe implement?
    parafac2_constraint=None,
    log_det=None,
    reg=None,
    feasibility_penalty_scale=10,
    constant_feasibility_penalty=False,
    aux_init=None,
    dual_init=None,
    svd="numpy_svd",
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
    # TODO: docstring
    random_state = tl.check_random_state(random_state)
    svd_fun = _get_svd(svd)
    cmf = initialize_cmf(matrices, rank, init, svd_fun=svd_fun, random_state=random_state, init_params=init_params)

    # Parse constraints
    if not is_iterable(l2_penalty):
        l2_penalty = [l2_penalty]*3
    l2_penalty = [p if not p is None else 0 for p in l2_penalty]
    # TODO: Parse constraints to generate appropriate proxes

    A_aux_list, B_is_aux_list, C_aux_list, = initialize_aux(
        matrices, rank, reg, random_state=random_state
    )  # TODO: Include cmf?
    A_dual_list, B_is_dual_list, C_dual_list, = initialize_dual(
        matrices, rank, reg, random_state=random_state
    )  # TODO: Include cmf?
    norm_matrices = _root_sum_squared_list(matrices)  # TODO: calculate norm of matrices
    rec_errors = []
    feasibility_gaps = []

    rec_error = _cmf_reconstruction_error(matrices, cmf)
    rec_error /= norm_matrices
    rec_errors.append(0.5*rec_error)
    A_gaps, B_gaps, C_gaps = compute_feasibility_gaps(cmf, reg, A_aux_list, B_is_aux_list, C_aux_list)
    feasibility_gaps.append((A_gaps, B_gaps, C_gaps))
    if verbose:
        print("Duality gaps for A: {}".format(A_gaps))
        print("Duality gaps for the Bi-matrices: {}".format(B_gaps))
        print("Duality gaps for C: {}".format(C_gaps))


    for it in range(n_iter_max):
        cmf, B_is_aux_list, B_is_dual_list = admm_update_B(
            matrices=matrices,
            reg=reg[1],
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
            reg=reg[0],
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
            reg=reg[2],
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
            # TODO: Maybe not always compute this (to save computation)?
            # TODO: Include the regularisation
            rec_error = _cmf_reconstruction_error(matrices, cmf)
            rec_error /= norm_matrices
            rec_errors.append(0.5*rec_error)
            A_gaps, B_gaps, C_gaps = compute_feasibility_gaps(cmf, reg, A_aux_list, B_is_aux_list, C_aux_list)
            feasibility_gaps.append((A_gaps, B_gaps, C_gaps))

            if it >= 1:
                if verbose:
                    print(
                        "Coupled matrix factorization reconstruction error={}, variation={}.".format(
                            rec_errors[-1], rec_errors[-2] - rec_errors[-1]
                        )
                    )
                    print("Duality gaps for A: {}".format(A_gaps))
                    print("Duality gaps for the Bi-matrices: {}".format(B_gaps))
                    print("Duality gaps for C: {}".format(C_gaps))

                if tol:
                    max_feasibility_gap = -np.inf
                    if len(A_gaps):
                        max_feasibility_gap = max((max(A_gaps), max_feasibility_gap))
                    if len(B_gaps):
                        max_feasibility_gap = max((max(B_gaps), max_feasibility_gap))
                    if len(C_gaps):
                        max_feasibility_gap = max((max(C_gaps), max_feasibility_gap))
                    #max_feasibility_gap = max(max(A_gaps), max(B_gaps), max(C_gaps))

                    # Compute stopping criterions
                    feasibility_criterion = max_feasibility_gap < feasibility_tol
                    rel_loss_criterion = abs(rec_errors[-2] ** 2 - rec_errors[-1] ** 2) < (tol * rec_errors[-2] ** 2)
                    abs_loss_criterion = rec_errors[-1] ** 2 < absolute_tol
                    if feasibility_criterion and (rel_loss_criterion or abs_loss_criterion):
                        if verbose:
                            print("converged in {} iterations.".format(it))
                        break

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
