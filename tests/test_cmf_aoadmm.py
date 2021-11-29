import inspect
import itertools
from copy import copy
from unittest.mock import patch

import numpy as np
import pytest
import tensorly as tl
from tensorly.metrics.factors import congruence_coefficient
from tensorly.testing import assert_array_almost_equal, assert_array_equal

import matcouply
from matcouply import cmf_aoadmm, coupled_matrices, penalties, random
from matcouply._utils import get_svd
from matcouply.coupled_matrices import CoupledMatrixFactorization
from matcouply.penalties import NonNegativity


def all_combinations(*args):
    """All combinations of the input iterables.

    Each argument must be an iterable.

    Examples:
    ---------
    >>> all_combinations([1, 2], ["ab", "cd"])
    [(1, 'ab'), (1, 'cd'), (2, 'ab'), (2, 'cd')]
    """
    return list(itertools.product(*args))


def normalize(X):
    return X / tl.sqrt(tl.sum(X ** 2, axis=0, keepdims=True))


@pytest.mark.parametrize(
    "rank,init",
    all_combinations(
        [1, 2, 5],
        ["random", "svd", "threshold_svd", "parafac2_als", "parafac_als", "cp_als", "parafac_hals", "cp_hals"],
    ),
)
def test_initialize_cmf(rng, rank, init):
    shapes = ((5, 10), (10, 10), (15, 10))
    matrices = [rng.random_sample(shape) for shape in shapes]

    svd_fun = get_svd("truncated_svd")
    cmf = cmf_aoadmm.initialize_cmf(matrices, rank, init, svd_fun, random_state=None, init_params=None)
    init_matrices = coupled_matrices.cmf_to_matrices(cmf)
    for matrix, init_matrix in zip(matrices, init_matrices):
        assert matrix.shape == init_matrix.shape


@pytest.mark.parametrize(
    "rank", [1, 2, 5],
)
def test_initialize_aux(rng, rank):
    shapes = ((5, 10), (10, 10), (15, 10))
    matrices = [rng.random(shape) for shape in shapes]

    reg = [[NonNegativity(), NonNegativity(), NonNegativity()], [NonNegativity()], []]
    A_aux_list, B_aux_list, C_aux_list = cmf_aoadmm.initialize_aux(matrices, rank, reg, rng)
    assert len(A_aux_list) == 3
    assert len(B_aux_list) == 1
    assert len(C_aux_list) == 0

    for A_aux in A_aux_list:
        assert tl.shape(A_aux) == (len(shapes), rank)
    for B_is_aux in B_aux_list:
        for B_i_aux, shape in zip(B_is_aux, shapes):
            assert tl.shape(B_i_aux) == (shape[0], rank)
        assert len(B_is_aux) == len(shapes)
    for C_aux in C_aux_list:
        assert tl.shape(C_aux) == (shapes[0][1], rank)


@pytest.mark.parametrize(
    "rank", [1, 2, 5],
)
def test_initialize_dual(rng, rank):
    shapes = ((5, 10), (10, 10), (15, 10))
    matrices = [rng.random(shape) for shape in shapes]

    reg = [[NonNegativity(), NonNegativity(), NonNegativity()], [NonNegativity()], []]
    A_dual_list, B_dual_list, C_dual_list = cmf_aoadmm.initialize_dual(matrices, rank, reg, rng)
    assert len(A_dual_list) == 3
    assert len(B_dual_list) == 1
    assert len(C_dual_list) == 0

    for A_dual in A_dual_list:
        assert tl.shape(A_dual) == (len(shapes), rank)
    for B_is_dual in B_dual_list:
        for B_i_dual, shape in zip(B_is_dual, shapes):
            assert tl.shape(B_i_dual) == (shape[0], rank)
        assert len(B_is_dual) == len(shapes)
    for C_dual in C_dual_list:
        assert tl.shape(C_dual) == (shapes[0][1], rank)


def test_cmf_reconstruction_error(rng, random_ragged_cmf):
    cmf, shapes, rank = random_ragged_cmf
    matrices = cmf.to_matrices()

    # Add random noise
    noise = [rng.standard_normal(size=shape) for shape in shapes]
    noisy_matrices = [matrix + n for matrix, n in zip(matrices, noise)]
    noise_norm = tl.sqrt(sum(tl.sum(n ** 2) for n in noise))

    # Check that the error is equal to the noise magnitude
    error = cmf_aoadmm._cmf_reconstruction_error(noisy_matrices, cmf)
    assert error == pytest.approx(noise_norm)


def test_listify(rng):
    param_name = "testparameter"

    # Check that you can send in iterable of length 3
    three_elements_list = [1, 2, 3]
    out = cmf_aoadmm._listify(three_elements_list, param_name)
    assert len(out) == 3
    assert out[0] == 1
    assert out[1] == 2
    assert out[2] == 3

    # Check that when given an iterable with missnig modes, it returns a list with None for missing modes
    just_two_elements_dict = {0: 1, 2: 2}
    out = cmf_aoadmm._listify(just_two_elements_dict, param_name)
    assert len(out) == 3
    assert out[0] == 1
    assert out[1] is None
    assert out[2] == 2

    just_one_elements_dict = {1: 1}
    out = cmf_aoadmm._listify(just_one_elements_dict, param_name)
    assert len(out) == 3
    assert out[0] is None
    assert out[1] == 1
    assert out[2] is None

    # Check that you can send in a non-iterable
    out = cmf_aoadmm._listify(1, param_name)  # int
    assert len(out) == 3
    assert out[0] == 1
    assert out[1] == 1
    assert out[2] == 1
    out = cmf_aoadmm._listify(0.33, param_name)  # float
    assert len(out) == 3
    assert out[0] == 0.33
    assert out[1] == 0.33
    assert out[2] == 0.33

    # Check that you cannot send in an non-dictonary iterable of length other than 3
    just_two_elements_list = [1, 2]
    with pytest.raises(ValueError):
        out = cmf_aoadmm._listify(just_two_elements_list, param_name)

    four_elements_list = [1, 2, 3, 4]
    with pytest.raises(ValueError):
        out = cmf_aoadmm._listify(four_elements_list, param_name)


def test_compute_feasibility_gaps(rng, random_ragged_cmf):
    cmf, shapes, rank = random_ragged_cmf
    weights, (A, B_is, C) = cmf

    # Create list with random noise
    A_noise_list = [rng.standard_normal(A.shape) for _ in range(4)]
    C_noise_list = [rng.standard_normal(C.shape) for _ in range(3)]
    B_is_noise_list = [[rng.standard_normal(B_i.shape) for B_i in B_is] for _ in range(2)]

    # Create copy of factors with noise added
    A_aux_list = [A + A_noise for A_noise in A_noise_list]
    C_aux_list = [C + C_noise for C_noise in C_noise_list]
    B_aux_list = [[B_i + B_i_noise for B_i, B_i_noise in zip(B_is, B_is_noise)] for B_is_noise in B_is_noise_list]

    # Create regularizers and compute feasibility gap
    regs = [
        [NonNegativity() for _ in A_noise_list],
        [NonNegativity() for _ in B_is_noise_list],
        [NonNegativity() for _ in C_noise_list],
    ]
    A_gap_list, B_gap_list, C_gap_list = cmf_aoadmm.compute_feasibility_gaps(
        cmf, regs, A_aux_list, B_aux_list, C_aux_list
    )

    # Check that feasibility gap is correct
    for A_noise, A_gap in zip(A_noise_list, A_gap_list):
        assert tl.norm(A_noise) == pytest.approx(A_gap)
    for C_noise, C_gap in zip(C_noise_list, C_gap_list):
        assert tl.norm(C_noise) == pytest.approx(C_gap)
    B_concatenated_noise_list = [tl.concatenate(B_is_noise) for B_is_noise in B_is_noise_list]
    for B_noise, B_gap in zip(B_concatenated_noise_list, B_gap_list):
        assert tl.norm(B_noise) == pytest.approx(B_gap)


def test_parse_all_penalties():
    # Check that regularisation is added
    regs = cmf_aoadmm._parse_all_penalties(
        non_negative=1,
        lower_bound=2,
        upper_bound=3,
        l2_norm_bound=4,
        unimodal=None,  # TODO: FIXME
        parafac2=False,
        l1_penalty=7,
        tv_penalty=8,
        generalized_l2_penalty=None,
        svd="truncated_svd",
        dual_init="random_uniform",
        aux_init="random_uniform",
        verbose=False,
        regs=None,
    )
    num_reg = len(regs[0])
    for reg_list in regs:
        assert len(reg_list) == num_reg

    # Check that parafac2 is only applied on second mode
    regs = cmf_aoadmm._parse_all_penalties(
        non_negative=1,
        lower_bound=2,
        upper_bound=3,
        l2_norm_bound=4,
        unimodal=None,  # TODO: FIXME
        parafac2=True,
        l1_penalty=7,
        tv_penalty=8,
        generalized_l2_penalty=None,
        svd="truncated_svd",
        dual_init="random_uniform",
        aux_init="random_uniform",
        verbose=False,
        regs=None,
    )
    num_reg = len(regs[0])
    assert len(regs[1]) == num_reg + 1
    assert len(regs[2]) == num_reg

    # Check that we only have one reg when only non-negativity is imposed
    regs = cmf_aoadmm._parse_all_penalties(
        non_negative=[True, True, True],
        lower_bound=None,
        upper_bound=None,
        l2_norm_bound=None,
        unimodal=None,
        parafac2=False,
        l1_penalty=None,
        tv_penalty=None,
        generalized_l2_penalty=None,
        svd="truncated_svd",
        dual_init="random_uniform",
        aux_init="random_uniform",
        verbose=False,
        regs=None,
    )
    assert len(regs[0]) == 1
    assert len(regs[1]) == 1
    assert len(regs[2]) == 1

    assert isinstance(regs[0][0], NonNegativity)
    assert isinstance(regs[1][0], NonNegativity)
    assert isinstance(regs[2][0], NonNegativity)

    # Check that we can impose non-negativity on only two modes
    regs = cmf_aoadmm._parse_all_penalties(
        non_negative=[True, False, True],
        lower_bound=None,
        upper_bound=None,
        l2_norm_bound=None,
        unimodal=None,
        parafac2=False,
        l1_penalty=None,
        tv_penalty=None,
        generalized_l2_penalty=None,
        svd="truncated_svd",
        dual_init="random_uniform",
        aux_init="random_uniform",
        verbose=False,
        regs=None,
    )
    assert len(regs[0]) == 1
    assert len(regs[1]) == 0
    assert len(regs[2]) == 1

    assert isinstance(regs[0][0], NonNegativity)
    assert isinstance(regs[2][0], NonNegativity)

    # Check that we can use dict also
    regs = cmf_aoadmm._parse_all_penalties(
        non_negative={0: True, 2: True},
        lower_bound=None,
        upper_bound=None,
        l2_norm_bound=None,
        unimodal=None,
        parafac2=False,
        l1_penalty=None,
        tv_penalty=None,
        generalized_l2_penalty=None,
        svd="truncated_svd",
        dual_init="random_uniform",
        aux_init="random_uniform",
        verbose=False,
        regs=None,
    )
    assert len(regs[0]) == 1
    assert len(regs[1]) == 0
    assert len(regs[2]) == 1

    assert isinstance(regs[0][0], NonNegativity)
    assert isinstance(regs[2][0], NonNegativity)

    regs = cmf_aoadmm._parse_all_penalties(
        non_negative={2: True},
        lower_bound=None,
        upper_bound=None,
        l2_norm_bound=None,
        unimodal=None,
        parafac2=False,
        l1_penalty=None,
        tv_penalty=None,
        generalized_l2_penalty=None,
        svd="truncated_svd",
        dual_init="random_uniform",
        aux_init="random_uniform",
        verbose=False,
        regs=None,
    )
    assert len(regs[0]) == 0
    assert len(regs[1]) == 0
    assert len(regs[2]) == 1

    assert isinstance(regs[2][0], NonNegativity)

    regs = cmf_aoadmm._parse_all_penalties(
        non_negative={0: True, 1: True, 2: True},
        lower_bound=None,
        upper_bound=None,
        l2_norm_bound=None,
        unimodal=None,
        parafac2=False,
        l1_penalty=None,
        tv_penalty=None,
        generalized_l2_penalty=None,
        svd="truncated_svd",
        dual_init="random_uniform",
        aux_init="random_uniform",
        verbose=False,
        regs=None,
    )
    assert len(regs[0]) == 1
    assert len(regs[1]) == 1
    assert len(regs[2]) == 1

    assert isinstance(regs[0][0], NonNegativity)
    assert isinstance(regs[1][0], NonNegativity)
    assert isinstance(regs[2][0], NonNegativity)

    # Check that we can change the init methods
    regs = cmf_aoadmm._parse_all_penalties(
        non_negative=[True, True, True],
        lower_bound=None,
        upper_bound=None,
        l2_norm_bound=None,
        unimodal=None,
        parafac2=False,
        l1_penalty=None,
        tv_penalty=None,
        generalized_l2_penalty=None,
        svd="truncated_svd",
        dual_init="random_standard_normal",
        aux_init="random_standard_normal",
        verbose=False,
        regs=None,
    )
    for reg in regs:
        assert reg[0].dual_init == "random_standard_normal"
        assert reg[0].aux_init == "random_standard_normal"


def test_parse_all_penalties_verbose(capfd):
    # Check nothing is printed without verbose
    _ = cmf_aoadmm._parse_all_penalties(
        non_negative=[None, None, None],
        lower_bound=1,
        upper_bound=2,
        l2_norm_bound=3,
        unimodal=None,
        parafac2=True,
        l1_penalty=4,
        tv_penalty=5,
        generalized_l2_penalty=[None, np.eye(10), None],
        svd="truncated_svd",
        dual_init="random_uniform",
        aux_init="random_uniform",
        verbose=False,
        regs=None,
    )
    out, err = capfd.readouterr()
    assert len(out) == 0

    # Check that text is printed out when verbose is True
    _ = cmf_aoadmm._parse_all_penalties(
        non_negative=[None, None, None],
        lower_bound=1,
        upper_bound=2,
        l2_norm_bound=3,
        unimodal=None,
        parafac2=True,
        l1_penalty=4,
        tv_penalty=5,
        generalized_l2_penalty=[None, np.eye(10), None],
        svd="truncated_svd",
        dual_init="random_uniform",
        aux_init="random_uniform",
        verbose=True,
        regs=None,
    )
    out, err = capfd.readouterr()
    assert len(out) > 0

    # Check excact output for a small testcase with one penalty
    _ = cmf_aoadmm._parse_all_penalties(
        non_negative={2: True},
        lower_bound=None,
        upper_bound=None,
        l2_norm_bound=None,
        unimodal=None,
        parafac2=False,
        l1_penalty=None,
        tv_penalty=None,
        generalized_l2_penalty=None,
        svd="truncated_svd",
        dual_init="random_uniform",
        aux_init="random_uniform",
        verbose=True,
        regs=None,
    )
    out, err = capfd.readouterr()
    assert len(out) > 0
    message = (
        "Added mode 0 penalties and constraints:"
        + "\n"
        + " (no additional regularisation added)\n"
        + "Added mode 1 penalties and constraints:"
        + "\n"
        + " (no additional regularisation added)\n"
        + "Added mode 2 penalties and constraints:"
        + "\n"
        + " * Non negativity constraints\n"
    )
    assert out == message


@pytest.mark.parametrize(
    "dual_init, aux_init",
    all_combinations(["random_uniform", "random_standard_normal"], ["random_uniform", "random_standard_normal"]),
)
def test_parse_mode_penalties(dual_init, aux_init):
    # Check that dual and aux init is set correctly
    out, verbosity_str = cmf_aoadmm._parse_mode_penalties(
        non_negative=True,
        lower_bound=0,
        upper_bound=1,
        l2_norm_bound=2,
        unimodal=None,
        parafac2=None,
        l1_penalty=1,
        tv_penalty=2,
        generalized_l2_penalty=None,
        svd="truncated_svd",
        dual_init=dual_init,
        aux_init=aux_init,
    )
    for reg in out:
        assert reg.dual_init == dual_init
        assert reg.aux_init == aux_init

    for svd in ["truncated_svd", "numpy_svd"]:
        out, verbosity_str = cmf_aoadmm._parse_mode_penalties(
            non_negative=None,
            lower_bound=None,
            upper_bound=None,
            l2_norm_bound=None,
            unimodal=None,
            parafac2=True,
            l1_penalty=None,
            tv_penalty=None,
            generalized_l2_penalty=tl.eye(30),
            svd=svd,
            dual_init=dual_init,
            aux_init=aux_init,
        )
        for reg in out:
            assert reg.svd_fun == get_svd(svd)

    # Check that no penalty gives length 0
    out, verbosity_str = cmf_aoadmm._parse_mode_penalties(
        non_negative=None,
        lower_bound=None,
        upper_bound=None,
        l2_norm_bound=None,
        unimodal=None,
        parafac2=None,
        l1_penalty=None,
        tv_penalty=None,
        generalized_l2_penalty=None,
        svd="truncated_svd",
        dual_init=dual_init,
        aux_init=aux_init,
    )
    assert len(out) == 0
    assert verbosity_str == "\n (no additional regularisation added)"

    # Check that non-negativity gives length one
    out, verbosity_str = cmf_aoadmm._parse_mode_penalties(
        non_negative=True,
        lower_bound=None,
        upper_bound=None,
        l2_norm_bound=None,
        unimodal=None,
        parafac2=None,
        l1_penalty=None,
        tv_penalty=None,
        generalized_l2_penalty=None,
        svd="truncated_svd",
        dual_init=dual_init,
        aux_init=aux_init,
    )
    assert len(out) == 1
    assert isinstance(out[0], penalties.NonNegativity)

    # Test parsing of NN
    # ------------------
    # NN + lower bound
    out, verbosity_str = cmf_aoadmm._parse_mode_penalties(
        non_negative=True,
        lower_bound=5,
        upper_bound=None,
        l2_norm_bound=None,
        unimodal=None,
        parafac2=None,
        l1_penalty=None,
        tv_penalty=None,
        generalized_l2_penalty=None,
        svd="truncated_svd",
        dual_init=dual_init,
        aux_init=aux_init,
    )
    assert len(out) == 1
    assert isinstance(out[0], penalties.BoxConstraint)
    assert out[0].min_val == 5

    out, verbosity_str = cmf_aoadmm._parse_mode_penalties(
        non_negative=True,
        lower_bound=-1,
        upper_bound=None,
        l2_norm_bound=None,
        unimodal=None,
        parafac2=None,
        l1_penalty=None,
        tv_penalty=None,
        generalized_l2_penalty=None,
        svd="truncated_svd",
        dual_init=dual_init,
        aux_init=aux_init,
    )
    assert len(out) == 1
    assert isinstance(out[0], penalties.BoxConstraint)
    assert out[0].min_val == 0

    # NN + upper bound
    out, verbosity_str = cmf_aoadmm._parse_mode_penalties(
        non_negative=True,
        lower_bound=None,
        upper_bound=5,
        l2_norm_bound=None,
        unimodal=None,
        parafac2=None,
        l1_penalty=None,
        tv_penalty=None,
        generalized_l2_penalty=None,
        svd="truncated_svd",
        dual_init=dual_init,
        aux_init=aux_init,
    )
    assert len(out) == 1
    assert isinstance(out[0], penalties.BoxConstraint)
    assert out[0].max_val == 5
    assert out[0].min_val == 0

    # NN + lower and upper bound
    out, verbosity_str = cmf_aoadmm._parse_mode_penalties(
        non_negative=True,
        lower_bound=-1,
        upper_bound=5,
        l2_norm_bound=None,
        unimodal=None,
        parafac2=None,
        l1_penalty=None,
        tv_penalty=None,
        generalized_l2_penalty=None,
        svd="truncated_svd",
        dual_init=dual_init,
        aux_init=aux_init,
    )
    assert len(out) == 1
    assert isinstance(out[0], penalties.BoxConstraint)
    assert out[0].max_val == 5
    assert out[0].min_val == 0

    # NN + L1
    out, verbosity_str = cmf_aoadmm._parse_mode_penalties(
        non_negative=True,
        lower_bound=None,
        upper_bound=None,
        l2_norm_bound=None,
        unimodal=None,
        parafac2=None,
        l1_penalty=42,
        tv_penalty=None,
        generalized_l2_penalty=None,
        svd="truncated_svd",
        dual_init=dual_init,
        aux_init=aux_init,
    )
    assert len(out) == 1
    assert isinstance(out[0], penalties.L1Penalty)
    assert out[0].non_negativity
    assert out[0].reg_strength == 42

    # NN + L1 + lower bound
    out, verbosity_str = cmf_aoadmm._parse_mode_penalties(
        non_negative=True,
        lower_bound=-1,
        upper_bound=None,
        l2_norm_bound=None,
        unimodal=None,
        parafac2=None,
        l1_penalty=42,
        tv_penalty=None,
        generalized_l2_penalty=None,
        svd="truncated_svd",
        dual_init=dual_init,
        aux_init=aux_init,
    )
    assert len(out) == 2
    if isinstance(out[0], penalties.BoxConstraint):
        box, l1 = out
    else:
        l1, box = out

    assert isinstance(box, penalties.BoxConstraint)
    assert box.min_val == 0

    assert isinstance(l1, penalties.L1Penalty)
    assert l1.non_negativity
    assert l1.reg_strength == 42
    # NN + ball

    # NN + L1 + lower bound
    out, verbosity_str = cmf_aoadmm._parse_mode_penalties(
        non_negative=True,
        lower_bound=None,
        upper_bound=None,
        l2_norm_bound=42,
        unimodal=None,
        parafac2=None,
        l1_penalty=None,
        tv_penalty=None,
        generalized_l2_penalty=None,
        svd="truncated_svd",
        dual_init=dual_init,
        aux_init=aux_init,
    )
    assert len(out) == 1
    assert isinstance(out[0], penalties.L2Ball)
    assert out[0].non_negativity

    # Test parsing of Parafac2
    # Parafac2
    out, verbosity_str = cmf_aoadmm._parse_mode_penalties(
        non_negative=None,
        lower_bound=None,
        upper_bound=None,
        l2_norm_bound=None,
        unimodal=None,
        parafac2=True,
        l1_penalty=None,
        tv_penalty=None,
        generalized_l2_penalty=None,
        svd="truncated_svd",
        dual_init=dual_init,
        aux_init=aux_init,
    )
    assert len(out) == 1
    assert isinstance(out[0], penalties.Parafac2)
    # NN + Parafac2 (len=2, contains both a NN and a Parafac2)
    out, verbosity_str = cmf_aoadmm._parse_mode_penalties(
        non_negative=True,
        lower_bound=None,
        upper_bound=None,
        l2_norm_bound=None,
        unimodal=None,
        parafac2=True,
        l1_penalty=None,
        tv_penalty=None,
        generalized_l2_penalty=None,
        svd="truncated_svd",
        dual_init=dual_init,
        aux_init=aux_init,
    )
    assert len(out) == 2
    if isinstance(out[0], penalties.Parafac2):
        pf2, nn = out
    else:
        nn, pf2 = out
    assert isinstance(pf2, penalties.Parafac2)
    assert isinstance(nn, penalties.NonNegativity)

    # Test parsing of L1
    # L1
    out, verbosity_str = cmf_aoadmm._parse_mode_penalties(
        non_negative=None,
        lower_bound=None,
        upper_bound=None,
        l2_norm_bound=None,
        unimodal=None,
        parafac2=False,
        l1_penalty=1,
        tv_penalty=None,
        generalized_l2_penalty=None,
        svd="truncated_svd",
        dual_init=dual_init,
        aux_init=aux_init,
    )
    assert len(out) == 1
    assert isinstance(out[0], penalties.L1Penalty)
    # L1 + TV
    out, verbosity_str = cmf_aoadmm._parse_mode_penalties(
        non_negative=None,
        lower_bound=None,
        upper_bound=None,
        l2_norm_bound=None,
        unimodal=None,
        parafac2=False,
        l1_penalty=1,
        tv_penalty=2,
        generalized_l2_penalty=None,
        svd="truncated_svd",
        dual_init=dual_init,
        aux_init=aux_init,
    )
    assert len(out) == 1
    assert isinstance(out[0], penalties.TotalVariationPenalty)
    assert out[0].reg_strength == 2
    assert out[0].l1_strength == 1
    assert verbosity_str == "\n * Total Variation penalty (with L1)"
    # L1 + upper bound
    out, verbosity_str = cmf_aoadmm._parse_mode_penalties(
        non_negative=None,
        lower_bound=None,
        upper_bound=10,
        l2_norm_bound=None,
        unimodal=None,
        parafac2=False,
        l1_penalty=1,
        tv_penalty=None,
        generalized_l2_penalty=None,
        svd="truncated_svd",
        dual_init=dual_init,
        aux_init=aux_init,
    )
    assert len(out) == 2
    if isinstance(out[0], penalties.L1Penalty):
        l1, box = out
    else:
        box, l1 = out
    assert isinstance(l1, penalties.L1Penalty)
    assert isinstance(box, penalties.BoxConstraint)
    assert box.max_val == 10

    # Test parsing of TV
    # TV
    out, verbosity_str = cmf_aoadmm._parse_mode_penalties(
        non_negative=None,
        lower_bound=None,
        upper_bound=None,
        l2_norm_bound=None,
        unimodal=None,
        parafac2=False,
        l1_penalty=None,
        tv_penalty=1,
        generalized_l2_penalty=None,
        svd="truncated_svd",
        dual_init=dual_init,
        aux_init=aux_init,
    )
    assert len(out) == 1
    assert isinstance(out[0], penalties.TotalVariationPenalty)
    assert out[0].reg_strength == 1
    assert verbosity_str == "\n * Total Variation penalty"
    # TV + Parafac2
    out, verbosity_str = cmf_aoadmm._parse_mode_penalties(
        non_negative=None,
        lower_bound=None,
        upper_bound=None,
        l2_norm_bound=None,
        unimodal=None,
        parafac2=True,
        l1_penalty=None,
        tv_penalty=1,
        generalized_l2_penalty=None,
        svd="truncated_svd",
        dual_init=dual_init,
        aux_init=aux_init,
    )
    assert len(out) == 2
    if isinstance(out[0], penalties.Parafac2):
        pf2, tv = out
    else:
        tv, pf2 = out
    assert isinstance(pf2, penalties.Parafac2)
    assert isinstance(tv, penalties.TotalVariationPenalty)
    assert tv.reg_strength == 1

    # Test parsing of ball constraints
    # Ball
    out, verbosity_str = cmf_aoadmm._parse_mode_penalties(
        non_negative=None,
        lower_bound=None,
        upper_bound=None,
        l2_norm_bound=1,
        unimodal=None,
        parafac2=None,
        l1_penalty=None,
        tv_penalty=None,
        generalized_l2_penalty=None,
        svd="truncated_svd",
        dual_init=dual_init,
        aux_init=aux_init,
    )
    assert len(out) == 1
    assert isinstance(out[0], penalties.L2Ball)
    assert out[0].norm_bound == 1
    # Ball + TV
    out, verbosity_str = cmf_aoadmm._parse_mode_penalties(
        non_negative=None,
        lower_bound=None,
        upper_bound=None,
        l2_norm_bound=2,
        unimodal=None,
        parafac2=None,
        l1_penalty=None,
        tv_penalty=1,
        generalized_l2_penalty=None,
        svd="truncated_svd",
        dual_init=dual_init,
        aux_init=aux_init,
    )
    assert len(out) == 2
    if isinstance(out[0], penalties.L2Ball):
        ball, tv = out
    else:
        tv, ball = out
    assert isinstance(ball, penalties.L2Ball)
    assert isinstance(tv, penalties.TotalVariationPenalty)
    assert tv.reg_strength == 1
    assert ball.norm_bound == 2

    # Test parsing of generalized L2
    # Generalized L2
    out, verbosity_str = cmf_aoadmm._parse_mode_penalties(
        non_negative=None,
        lower_bound=None,
        upper_bound=None,
        l2_norm_bound=None,
        unimodal=None,
        parafac2=None,
        l1_penalty=None,
        tv_penalty=None,
        generalized_l2_penalty=tl.eye(10),
        svd="truncated_svd",
        dual_init=dual_init,
        aux_init=aux_init,
    )
    assert len(out) == 1
    assert isinstance(out[0], penalties.GeneralizedL2Penalty)
    assert_array_equal(out[0].norm_matrix, tl.eye(10))
    # Generalized L2 + L1
    out, verbosity_str = cmf_aoadmm._parse_mode_penalties(
        non_negative=None,
        lower_bound=None,
        upper_bound=None,
        l2_norm_bound=None,
        unimodal=None,
        parafac2=None,
        l1_penalty=2,
        tv_penalty=None,
        generalized_l2_penalty=tl.eye(10),
        svd="truncated_svd",
        dual_init=dual_init,
        aux_init=aux_init,
    )
    assert len(out) == 2
    if isinstance(out[0], penalties.GeneralizedL2Penalty):
        generalized_l2, l1 = out
    else:
        l1, generalized_l2 = out
    assert isinstance(generalized_l2, penalties.GeneralizedL2Penalty)
    assert_array_equal(generalized_l2.norm_matrix, tl.eye(10))
    assert isinstance(l1, penalties.L1Penalty)
    assert l1.reg_strength == 2


@pytest.mark.parametrize(
    "feasibility_penalty_scale,constant_feasibility_penalty", all_combinations([0.5, 1, 10], [True, False])
)
def test_admm_update_A(rng, random_ragged_cmf, feasibility_penalty_scale, constant_feasibility_penalty):
    cmf, shapes, rank = random_ragged_cmf
    weights, (A, B_is, C) = cmf

    # Test that ADMM update with no constraints finds the correct A-matrix
    # when Bi-s and C is correct but A is incorrect
    A2 = rng.uniform(size=tl.shape(A))
    modified_cmf = coupled_matrices.CoupledMatrixFactorization((weights, (A2, B_is, C)))

    out = cmf_aoadmm.admm_update_A(
        matrices=cmf.to_matrices(),
        reg=[],
        cmf=modified_cmf,
        A_aux_list=[],
        A_dual_list=[],
        l2_penalty=0,
        inner_n_iter_max=1000,
        inner_tol=-1,
        feasibility_penalty_scale=feasibility_penalty_scale,
        constant_feasibility_penalty=constant_feasibility_penalty,
        svd_fun=get_svd("truncated_svd"),
    )
    out_cmf, out_A_auxes, out_A_duals = out
    out_A_normalized = normalize(out_cmf[1][0])
    A_normalized = normalize(A)
    assert_array_almost_equal(out_A_normalized, A_normalized)

    # Test that ADMM update with NN constraints finds the correct A-matrix
    # when Bi-s and C is correct but A is incorrect
    A = tl.clip(A, 0, None)
    B_is = [tl.clip(B_i, 0, None) for B_i in B_is]
    C = tl.clip(C, 0, None)

    A2 = rng.uniform(size=tl.shape(A))
    modified_cmf = coupled_matrices.CoupledMatrixFactorization((weights, (A2, B_is, C)))
    nn_cmf = coupled_matrices.CoupledMatrixFactorization((weights, (A, B_is, C)))
    nn = NonNegativity()
    aux = nn.init_aux(nn_cmf.to_matrices(), rank, 0, rng)
    dual = nn.init_dual(nn_cmf.to_matrices(), rank, 0, rng)
    out = cmf_aoadmm.admm_update_A(
        matrices=nn_cmf.to_matrices(),
        reg=[nn],
        cmf=modified_cmf,
        A_aux_list=[aux],
        A_dual_list=[dual],
        l2_penalty=0,
        inner_n_iter_max=1_000,
        inner_tol=-1,
        feasibility_penalty_scale=feasibility_penalty_scale,
        constant_feasibility_penalty=constant_feasibility_penalty,
        svd_fun=get_svd("truncated_svd"),
    )
    out_cmf, out_A_auxes, out_A_duals = out
    out_A_normalized = normalize(out_cmf[1][0])
    A_normalized = normalize(A)
    assert_array_almost_equal(out_A_normalized, A_normalized, decimal=3)

    # Check that the feasibility gap is low
    assert_array_almost_equal(out_cmf[1][0], out_A_auxes[0])

    # TODO: Test for l2_penalty > 0
    # # For l2_penalty, compute linear system and solve using SVD to obtain regularised components.
    # This will work with NN constraints too


@pytest.mark.parametrize(
    "feasibility_penalty_scale,constant_feasibility_penalty", all_combinations([0.5, 1, 10], [True, False])
)
def test_admm_update_B(rng, random_ragged_cmf, feasibility_penalty_scale, constant_feasibility_penalty):
    cmf, shapes, rank = random_ragged_cmf
    weights, (A, B_is, C) = cmf

    # Expand the C-matrix to have enough measurements per element of B (this makes the system easier to solve)
    new_K = 10 * (rank + max(shape[0] for shape in shapes) + A.shape[0])
    C = rng.standard_normal((new_K, rank))
    shapes = [(shape[0], new_K) for shape in shapes]
    cmf = coupled_matrices.CoupledMatrixFactorization((weights, (A, B_is, C)))

    # Test that ADMM update with no constraints finds the correct B_i-matrices
    # when A and C is correct but B_is is incorrect
    B_is2 = [rng.uniform(size=tl.shape(B_i)) for B_i in B_is]
    modified_cmf = coupled_matrices.CoupledMatrixFactorization((weights, (A, B_is2, C)))

    out = cmf_aoadmm.admm_update_B(
        matrices=cmf.to_matrices(),
        reg=[],
        cmf=modified_cmf,
        B_is_aux_list=[],
        B_is_dual_list=[],
        l2_penalty=0,
        inner_n_iter_max=1000,
        inner_tol=-1,
        feasibility_penalty_scale=feasibility_penalty_scale,
        constant_feasibility_penalty=constant_feasibility_penalty,
        svd_fun=get_svd("truncated_svd"),
    )
    out_cmf, out_B_auxes, out_B_duals = out
    out_B_is = out_cmf[1][1]
    out_B_is_normalized = [normalize(B_i) for B_i in out_B_is]

    B_is_normalized = [normalize(B_i) for B_i in B_is]

    for B_i, out_B_i in zip(B_is_normalized, out_B_is_normalized):
        assert_array_almost_equal(B_i, out_B_i, decimal=3)

    # WITH NON NEGATIVITY
    # Test that ADMM update with NN constraints finds the correct B_i-matrices
    # when A and C is correct but B_is is incorrect
    A = tl.clip(A, 0.1, None)  # Increasing A improves the conditioning of the system
    B_is = [tl.clip(B_i, 0, None) for B_i in B_is]
    C = tl.clip(C, 0, None)

    nn_cmf = coupled_matrices.CoupledMatrixFactorization((weights, (A, B_is, C)))
    modified_cmf = coupled_matrices.CoupledMatrixFactorization((weights, (A, B_is2, C)))
    nn = NonNegativity()
    aux = nn.init_aux(nn_cmf.to_matrices(), rank, 1, rng)
    dual = nn.init_dual(nn_cmf.to_matrices(), rank, 1, rng)
    out = cmf_aoadmm.admm_update_B(
        matrices=nn_cmf.to_matrices(),
        reg=[nn],
        cmf=modified_cmf,
        B_is_aux_list=[aux],
        B_is_dual_list=[dual],
        l2_penalty=0,
        inner_n_iter_max=5_000,
        inner_tol=-1,
        feasibility_penalty_scale=feasibility_penalty_scale,
        constant_feasibility_penalty=constant_feasibility_penalty,
        svd_fun=get_svd("truncated_svd"),
    )
    out_cmf, out_B_auxes, out_B_duals = out
    out_B_is = out_cmf[1][1]
    out_B_is_normalized = [normalize(B_i) for B_i in out_B_is]
    B_is_normalized = [normalize(B_i) for B_i in B_is]

    for B_i, out_B_i in zip(B_is_normalized, out_B_is_normalized):
        assert_array_almost_equal(B_i, out_B_i, decimal=3)

    # Check that the feasibility gap is low
    for aux_B_i, out_B_i in zip(out_B_auxes[0], out_B_is):
        assert_array_almost_equal(aux_B_i, out_B_i, decimal=3)

    # TODO: Test for l2_penalty > 0
    ## For l2_penalty, compute linear system and solve using SVD to obtain regularised components.
    # This will work with NN constraints too


@pytest.mark.parametrize("feasibility_penalty_scale", [True, False])
def test_admm_update_C(rng, random_ragged_cmf, feasibility_penalty_scale):
    cmf, shapes, rank = random_ragged_cmf
    weights, (A, B_is, C) = cmf

    # Test that ADMM update with no constraints finds the correct C-matrix
    # when Bi-s and A is correct but C is incorrect
    C2 = rng.uniform(size=tl.shape(C))
    modified_cmf = coupled_matrices.CoupledMatrixFactorization((weights, (A, B_is, C2)))

    out = cmf_aoadmm.admm_update_C(
        matrices=cmf.to_matrices(),
        reg=[],
        cmf=modified_cmf,
        C_aux_list=[],
        C_dual_list=[],
        l2_penalty=0,
        inner_n_iter_max=1000,
        inner_tol=-1,
        feasibility_penalty_scale=feasibility_penalty_scale,
        svd_fun=get_svd("truncated_svd"),
    )
    out_C_normalized = normalize(out[0][1][2])
    C_normalized = normalize(C)
    assert_array_almost_equal(out_C_normalized, C_normalized)

    # Test that ADMM update with NN constraints finds the correct C-matrix
    # when Bi-s and A is correct but C is incorrect
    A = tl.clip(A, 0, None)
    B_is = [tl.clip(B_i, 0, None) for B_i in B_is]
    C = tl.clip(C, 0, None)
    nn_cmf = coupled_matrices.CoupledMatrixFactorization((weights, (A, B_is, C)))

    C2 = rng.uniform(size=tl.shape(C))
    modified_cmf = coupled_matrices.CoupledMatrixFactorization((weights, (A, B_is, C2)))

    nn = NonNegativity()
    aux = nn.init_aux(nn_cmf.to_matrices(), rank, 2, rng)
    dual = nn.init_dual(nn_cmf.to_matrices(), rank, 2, rng)
    out = cmf_aoadmm.admm_update_C(
        matrices=nn_cmf.to_matrices(),
        reg=[nn],
        cmf=modified_cmf,
        C_aux_list=[aux],
        C_dual_list=[dual],
        l2_penalty=0,
        inner_n_iter_max=1_000,
        inner_tol=-1,
        feasibility_penalty_scale=feasibility_penalty_scale,
        svd_fun=get_svd("truncated_svd"),
    )
    out_cmf, out_C_auxes, out_C_duals = out
    out_C_normalized = out_cmf[1][2] / tl.sqrt(tl.sum(out_cmf[1][2] ** 2, axis=0, keepdims=True))
    C_normalized = C / tl.sqrt(tl.sum(C ** 2, axis=0, keepdims=True))
    assert_array_almost_equal(out_C_normalized, C_normalized, decimal=3)

    # Check that the feasibility gap is low
    assert_array_almost_equal(out_cmf[1][2], out_C_auxes[0])

    # TODO: Test for l2_penalty > 0
    ## For l2_penalty, compute linear system and solve using SVD to obtain regularised components.
    # This will work with NN constraints too


def test_cmf_aoadmm(rng, random_ragged_cmf):
    cmf, shapes, rank = random_ragged_cmf
    weights, (A, B_is, C) = cmf

    # Ensure that the components are non-negative
    A = tl.clip(A, 0, None)
    B_is = [tl.clip(B_i, 0, None) for B_i in B_is]
    C = tl.clip(C, 0, None)
    nn_cmf = coupled_matrices.CoupledMatrixFactorization((weights, (A, B_is, C)))

    # Construct matrices and compute their norm
    matrices = nn_cmf.to_matrices()
    norm_matrices = tl.sqrt(sum(tl.sum(matrix ** 2) for matrix in matrices))

    # Decompose matrices with cmf_aoadmm with no constraints
    out_cmf, aux, dual, rec_errors, feasibility_gaps = cmf_aoadmm.cmf_aoadmm(
        matrices, rank, n_iter_max=10_000, return_errors=True
    )

    # Check that reconstruction error is low
    assert rec_errors[-1] < 1e-02

    # Add non-negativity constraints on all modes
    out_cmf, aux, dual, rec_errors, feasibility_gaps = cmf_aoadmm.cmf_aoadmm(
        matrices, rank, n_iter_max=10_000, return_errors=True, non_negative=[True, True, True]
    )

    # Check that reconstruction error is low
    assert rec_errors[-1] < 1e-02

    # Check that we get errors out when we ask for errors. Even if convergence checking is disabled and verbose=False
    out = cmf_aoadmm.cmf_aoadmm(matrices, rank, return_errors=True, tol=None, absolute_tol=None, verbose=False)
    assert len(out) == 5

    # Check that we don't get errors out if we don't ask for it
    out = cmf_aoadmm.cmf_aoadmm(matrices, rank, return_errors=False)
    assert len(out) == 3

    # Check that we can add non-negativity constraints with list of regs.
    regs = [[NonNegativity()], [NonNegativity()], [NonNegativity()]]
    out_cmf, aux, dual, rec_errors, feasibility_gaps = cmf_aoadmm.cmf_aoadmm(
        matrices, rank, n_iter_max=10_000, return_errors=True, regs=regs
    )
    # Check that final reconstruction error is the same as when we compute it with the returned decomposition and auxes
    assert cmf_aoadmm._cmf_reconstruction_error(matrices, out_cmf) / norm_matrices == pytest.approx(rec_errors[-1])

    # Check that feasibility gaps are the same as when we compute it with the returned decomposition and auxes
    A_gap_list, B_gap_list, C_gap_list = cmf_aoadmm.compute_feasibility_gaps(out_cmf, regs, *aux)
    for A_gap, out_A_gap in zip(A_gap_list, feasibility_gaps[-1][0]):
        assert A_gap == pytest.approx(out_A_gap)
    for B_gap, out_B_gap in zip(B_gap_list, feasibility_gaps[-1][1]):
        assert B_gap == pytest.approx(out_B_gap)
    for C_gap, out_C_gap in zip(C_gap_list, feasibility_gaps[-1][2]):
        assert C_gap == pytest.approx(out_C_gap)
    # TODO: Test that the code fails gracefully with list of regs not list of list of regs


def test_cmf_aoadmm_verbose(rng, random_ragged_cmf, capfd):
    cmf, shapes, rank = random_ragged_cmf
    matrices = cmf.to_matrices()

    out_cmf, aux, dual, rec_errors, feasibility_gaps = cmf_aoadmm.cmf_aoadmm(
        matrices, rank, n_iter_max=1_000, return_errors=True, verbose=False
    )
    out, err = capfd.readouterr()
    assert len(out) == 0

    out_cmf, aux, dual, rec_errors, feasibility_gaps = cmf_aoadmm.cmf_aoadmm(
        matrices, rank, n_iter_max=1_000, return_errors=True, verbose=True
    )
    out, err = capfd.readouterr()
    assert len(out) > 0


def test_parafac2_makes_nn_cmf_unique(rng):
    rank = 2
    A = rng.uniform(0.1, 1.1, size=(15, rank))
    B_0 = rng.uniform(0, 1, size=(10, rank))
    B_is = [tl.tensor(np.roll(B_0, i, axis=1)) for i in range(15)]
    C = rng.uniform(0, 1, size=(20, rank))
    weights = None

    cmf = CoupledMatrixFactorization((weights, (A, B_is, C)))
    matrices = cmf.to_matrices()

    rec_errors = [float("inf")]
    for init in range(5):
        out = cmf_aoadmm.cmf_aoadmm(
            matrices, rank, n_iter_max=1_000, return_errors=True, non_negative=[True, True, True], parafac2=True
        )

        if out[3][-1] < rec_errors[-1]:
            out_cmf, aux, dual, rec_errors, feasibility_gaps = out

    # Check that reconstruction error is low and that the correct factors are recovered
    assert rec_errors[-1] < 1e-04
    assert congruence_coefficient(A, out_cmf[1][0], absolute_value=True)[0] > 0.99
    for B_i, out_B_i in zip(B_is, out_cmf[1][1]):
        assert congruence_coefficient(B_i, out_B_i, absolute_value=True)[0] > 0.99
    assert congruence_coefficient(C, out_cmf[1][2], absolute_value=True)[0] > 0.99


def test_parafac2_aoadmm(rng, random_ragged_cmf):
    parafac2_argspecs = inspect.getfullargspec(matcouply.cmf_aoadmm.parafac2_aoadmm)
    placeholder_args = {arg: f"PLACEHOLDERARG_{i}" for i, arg in enumerate(parafac2_argspecs.args)}
    cmf_aoadmm_args = copy(placeholder_args)
    cmf_aoadmm_args["parafac2"] = True
    with patch("matcouply.cmf_aoadmm.cmf_aoadmm") as mock:
        cmf_aoadmm.parafac2_aoadmm(**placeholder_args)
        mock.assert_called_once()
        mock.assert_called_once_with(**cmf_aoadmm_args)
