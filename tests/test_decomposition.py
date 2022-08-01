import inspect
from copy import copy
from unittest.mock import patch

import numpy as np
import pytest
import tensorly as tl
from tensorly.metrics.factors import congruence_coefficient
from tensorly.testing import assert_array_equal

import matcouply
from matcouply import coupled_matrices, decomposition, penalties
from matcouply._utils import get_svd
from matcouply.coupled_matrices import CoupledMatrixFactorization
from matcouply.penalties import BoxConstraint, L2Ball, NonNegativity
from matcouply.testing import assert_allclose

from .utils import RTOL_SCALE, all_combinations


def normalize(X):
    ssX = tl.sum(X ** 2, 0)
    ssX = tl.reshape(ssX, (1, *tl.shape(ssX)))
    return X / tl.sqrt(ssX)


@pytest.mark.parametrize(
    "rank,init",
    all_combinations(
        [1, 2, 5],
        ["random", "svd", "threshold_svd", "parafac2_als", "parafac_als", "cp_als", "parafac_hals", "cp_hals"],
    ),
)
def test_initialize_cmf(rng, rank, init):
    shapes = ((5, 10), (10, 10), (15, 10))
    matrices = [tl.tensor(rng.random_sample(shape)) for shape in shapes]

    svd_fun = get_svd("truncated_svd")
    cmf = decomposition.initialize_cmf(matrices, rank, init, svd_fun, random_state=None, init_params=None)
    init_matrices = coupled_matrices.cmf_to_matrices(cmf)
    for matrix, init_matrix in zip(matrices, init_matrices):
        assert matrix.shape == init_matrix.shape


def test_initialize_cmf_init_with_cmf(rng, random_ragged_cmf):
    cmf, shapes, rank = random_ragged_cmf
    weights, (A, B_is, C) = cmf

    out_weights, (out_A, out_B_is, out_C) = decomposition.cmf_aoadmm(
        cmf.to_matrices(), rank, init=cmf, n_iter_max=0, return_errors=False, return_admm_vars=False
    )
    assert out_weights is None
    assert_allclose(out_A, weights * A)
    for out_B_i, B_i in zip(out_B_is, B_is):
        assert_allclose(out_B_i, B_i)
    assert_allclose(out_C, C)

    with pytest.raises(ValueError):
        invalid_init = None, (A, B_is, tl.zeros((tl.shape(C)[0], rank + 1)))
        decomposition.cmf_aoadmm(
            cmf.to_matrices(), rank, init=invalid_init, n_iter_max=0, return_errors=False, return_admm_vars=False
        )


def test_initialize_cmf_invalid_init(rng):
    shapes = ((5, 10), (10, 10), (15, 10))
    rank = 3
    matrices = [tl.tensor(rng.random_sample(shape)) for shape in shapes]

    svd_fun = get_svd("truncated_svd")
    init = "INVALID INITIALIZATION"
    with pytest.raises(ValueError):
        decomposition.initialize_cmf(matrices, rank, init, svd_fun, random_state=None, init_params=None)


@pytest.mark.parametrize(
    "rank", [1, 2, 5],
)
def test_initialize_aux(rng, rank):
    shapes = ((5, 10), (10, 10), (15, 10))
    matrices = [rng.random_sample(shape) for shape in shapes]

    reg = [[NonNegativity(), NonNegativity(), NonNegativity()], [NonNegativity()], []]
    A_aux_list, B_aux_list, C_aux_list = decomposition.initialize_aux(matrices, rank, reg, rng)
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
    matrices = [rng.random_sample(shape) for shape in shapes]

    reg = [[NonNegativity(), NonNegativity(), NonNegativity()], [NonNegativity()], []]
    A_dual_list, B_dual_list, C_dual_list = decomposition.initialize_dual(matrices, rank, reg, rng)
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
    noise = [tl.tensor(rng.standard_normal(size=shape)) for shape in shapes]
    noisy_matrices = [matrix + n for matrix, n in zip(matrices, noise)]
    noise_norm = tl.sqrt(sum(tl.sum(n ** 2) for n in noise))

    # Check that the error is equal to the noise magnitude
    error = decomposition._cmf_reconstruction_error(noisy_matrices, cmf)
    assert error == pytest.approx(noise_norm)


def test_listify(rng):
    param_name = "testparameter"

    # Check that you can send in iterable of length 3
    three_elements_list = [1, 2, 3]
    out = decomposition._listify(three_elements_list, param_name)
    assert len(out) == 3
    assert out[0] == 1
    assert out[1] == 2
    assert out[2] == 3

    # Check that when given an iterable with missnig modes, it returns a list with None for missing modes
    just_two_elements_dict = {0: 1, 2: 2}
    out = decomposition._listify(just_two_elements_dict, param_name)
    assert len(out) == 3
    assert out[0] == 1
    assert out[1] is None
    assert out[2] == 2

    just_one_elements_dict = {1: 1}
    out = decomposition._listify(just_one_elements_dict, param_name)
    assert len(out) == 3
    assert out[0] is None
    assert out[1] == 1
    assert out[2] is None

    # Check that you can send in a non-iterable
    out = decomposition._listify(1, param_name)  # int
    assert len(out) == 3
    assert out[0] == 1
    assert out[1] == 1
    assert out[2] == 1
    out = decomposition._listify(0.33, param_name)  # float
    assert len(out) == 3
    assert out[0] == 0.33
    assert out[1] == 0.33
    assert out[2] == 0.33

    # Check that you cannot send in an non-dictonary iterable of length other than 3
    just_two_elements_list = [1, 2]
    with pytest.raises(ValueError):
        out = decomposition._listify(just_two_elements_list, param_name)

    four_elements_list = [1, 2, 3, 4]
    with pytest.raises(ValueError):
        out = decomposition._listify(four_elements_list, param_name)


def test_listify_is_called_on_l2(rng, random_ragged_cmf):
    cmf, shapes, rank = random_ragged_cmf
    l2_penalty = None
    with patch("matcouply.decomposition._listify", return_value=[l2_penalty] * 3) as mock:
        decomposition.cmf_aoadmm(cmf.to_matrices(), rank, n_iter_max=2, l2_penalty=l2_penalty)
        mock.assert_called()
        mock.assert_any_call(l2_penalty, "l2_penalty")


def test_compute_feasibility_gaps(rng, random_ragged_cmf):
    cmf, shapes, rank = random_ragged_cmf
    weights, (A, B_is, C) = cmf

    # Create list with random noise
    A_noise_list = [tl.tensor(rng.standard_normal(A.shape)) for _ in range(4)]
    C_noise_list = [tl.tensor(rng.standard_normal(C.shape)) for _ in range(3)]
    B_is_noise_list = [[tl.tensor(rng.standard_normal(B_i.shape)) for B_i in B_is] for _ in range(2)]

    A_norm = tl.norm(A)
    B_norm = tl.norm(tl.concatenate(B_is))
    C_norm = tl.norm(C)

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
    A_gap_list, B_gap_list, C_gap_list = decomposition.compute_feasibility_gaps(
        cmf, regs, A_aux_list, B_aux_list, C_aux_list
    )

    # Check that feasibility gap is correct
    for A_noise, A_gap in zip(A_noise_list, A_gap_list):
        assert tl.norm(A_noise) / A_norm == pytest.approx(A_gap)
    for C_noise, C_gap in zip(C_noise_list, C_gap_list):
        assert tl.norm(C_noise) / C_norm == pytest.approx(C_gap)
    B_concatenated_noise_list = [tl.concatenate(B_is_noise) for B_is_noise in B_is_noise_list]
    for B_noise, B_gap in zip(B_concatenated_noise_list, B_gap_list):
        assert tl.norm(B_noise) / B_norm == pytest.approx(B_gap)


def test_parse_all_penalties():
    # Patch get backend since unimodality raises runtime error with non-numpy backend
    with patch("matcouply.decomposition.tensorly.get_backend", return_value="numpy"):
        # Check that regularization is added
        regs = decomposition._parse_all_penalties(
            non_negative=1,
            lower_bound=2,
            upper_bound=3,
            l2_norm_bound=4,
            unimodal=5,
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
        regs = decomposition._parse_all_penalties(
            non_negative=1,
            lower_bound=2,
            upper_bound=3,
            l2_norm_bound=4,
            unimodal=5,
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
    regs = decomposition._parse_all_penalties(
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
    regs = decomposition._parse_all_penalties(
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
    regs = decomposition._parse_all_penalties(
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

    regs = decomposition._parse_all_penalties(
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

    regs = decomposition._parse_all_penalties(
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
    regs = decomposition._parse_all_penalties(
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
    # Patch get backend since unimodality raises runtime error with non-numpy backend
    with patch("matcouply.decomposition.tensorly.get_backend", return_value="numpy"):
        # Check nothing is printed without verbose
        _ = decomposition._parse_all_penalties(
            non_negative=[None, None, None],
            lower_bound=1,
            upper_bound=2,
            l2_norm_bound=3,
            unimodal=4,
            parafac2=True,
            l1_penalty=5,
            tv_penalty=6,
            generalized_l2_penalty=[None, tl.eye(10), None],
            svd="truncated_svd",
            dual_init="random_uniform",
            aux_init="random_uniform",
            verbose=False,
            regs=None,
        )
        out, err = capfd.readouterr()
        assert len(out) == 0

        # Check that text is printed out when verbose is True
        _ = decomposition._parse_all_penalties(
            non_negative=[None, None, None],
            lower_bound=1,
            upper_bound=2,
            l2_norm_bound=3,
            unimodal=4,
            parafac2=True,
            l1_penalty=5,
            tv_penalty=6,
            generalized_l2_penalty=[None, tl.eye(10), None],
            svd="truncated_svd",
            dual_init="random_uniform",
            aux_init="random_uniform",
            verbose=True,
            regs=None,
        )
        out, err = capfd.readouterr()
        assert len(out) > 0

        # Check excact output for a small testcase with one penalty
        _ = decomposition._parse_all_penalties(
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
            "All regularization penalties (including regs list):\n"
            + "* Mode 0:"
            + "\n"
            + "   - (no regularization added)\n"
            + "* Mode 1:"
            + "\n"
            + "   - (no regularization added)\n"
            + "* Mode 2:"
            + "\n"
            + "   - <'matcouply.penalties.NonNegativity' with aux_init='random_uniform', dual_init='random_uniform')>\n"
        )
        assert out == message


@pytest.mark.parametrize(
    "dual_init, aux_init",
    all_combinations(["random_uniform", "random_standard_normal"], ["random_uniform", "random_standard_normal"]),
)
def test_parse_mode_penalties(dual_init, aux_init):
    # Check that dual and aux init is set correctly
    out = decomposition._parse_mode_penalties(
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
        out = decomposition._parse_mode_penalties(
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
    out = decomposition._parse_mode_penalties(
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

    # Check that non-negativity gives length one
    out = decomposition._parse_mode_penalties(
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
    out = decomposition._parse_mode_penalties(
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

    out = decomposition._parse_mode_penalties(
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
    out = decomposition._parse_mode_penalties(
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
    out = decomposition._parse_mode_penalties(
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
    out = decomposition._parse_mode_penalties(
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
    out = decomposition._parse_mode_penalties(
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
    out = decomposition._parse_mode_penalties(
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
    out = decomposition._parse_mode_penalties(
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
    out = decomposition._parse_mode_penalties(
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
    out = decomposition._parse_mode_penalties(
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
    out = decomposition._parse_mode_penalties(
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
    # L1 + upper bound
    out = decomposition._parse_mode_penalties(
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
    out = decomposition._parse_mode_penalties(
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
    # TV + Parafac2
    out = decomposition._parse_mode_penalties(
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
    out = decomposition._parse_mode_penalties(
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
    out = decomposition._parse_mode_penalties(
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
    out = decomposition._parse_mode_penalties(
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
    out = decomposition._parse_mode_penalties(
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

    # Unimodality
    # Patch get_backend since unimodality constraints raises runtime error on non-numpy backend
    with patch("matcouply.decomposition.tensorly.get_backend", return_value="numpy"):
        out = decomposition._parse_mode_penalties(
            non_negative=False,
            lower_bound=None,
            upper_bound=None,
            l2_norm_bound=None,
            unimodal=True,
            parafac2=None,
            l1_penalty=None,
            tv_penalty=None,
            generalized_l2_penalty=None,
            svd="truncated_svd",
            dual_init=dual_init,
            aux_init=aux_init,
        )
    assert len(out) == 1
    assert isinstance(out[0], penalties.Unimodality)
    assert not out[0].non_negativity

    # Unimodality + NN
    # Patch get_backend since unimodality constraints raises runtime error on non-numpy backend
    with patch("matcouply.decomposition.tensorly.get_backend", return_value="numpy"):
        out = decomposition._parse_mode_penalties(
            non_negative=True,
            lower_bound=None,
            upper_bound=None,
            l2_norm_bound=None,
            unimodal=True,
            parafac2=None,
            l1_penalty=None,
            tv_penalty=None,
            generalized_l2_penalty=None,
            svd="truncated_svd",
            dual_init=dual_init,
            aux_init=aux_init,
        )
    assert len(out) == 1
    assert isinstance(out[0], penalties.Unimodality)
    assert out[0].non_negativity

    # Unimodality + NN + TV
    # Patch get_backend since unimodality constraints raises runtime error on non-numpy backend
    with patch("matcouply.decomposition.tensorly.get_backend", return_value="numpy"):
        out = decomposition._parse_mode_penalties(
            non_negative=True,
            lower_bound=None,
            upper_bound=None,
            l2_norm_bound=None,
            unimodal=True,
            parafac2=None,
            l1_penalty=None,
            tv_penalty=1,
            generalized_l2_penalty=None,
            svd="truncated_svd",
            dual_init=dual_init,
            aux_init=aux_init,
        )
    assert len(out) == 2
    if isinstance(out[0], penalties.Unimodality):
        u, tv = out
    else:
        tv, u = out
    assert isinstance(tv, penalties.TotalVariationPenalty)
    assert isinstance(u, penalties.Unimodality)
    assert out[0].non_negativity


@pytest.mark.parametrize(
    "feasibility_penalty_scale,constant_feasibility_penalty", all_combinations([0.5, 1, 2], [True, False])
)
def test_admm_update_A(rng, random_ragged_cmf, feasibility_penalty_scale, constant_feasibility_penalty):
    cmf, shapes, rank = random_ragged_cmf
    weights, (A, B_is, C) = cmf

    # Test that ADMM update with no constraints finds the correct A-matrix
    # when Bi-s and C is correct but A is incorrect
    A2 = tl.tensor(rng.uniform(size=tl.shape(A)))
    modified_cmf = coupled_matrices.CoupledMatrixFactorization((weights, (A2, B_is, C)))

    out = decomposition.admm_update_A(
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
    assert_allclose(out_A_normalized, A_normalized, rtol=1e-6 * RTOL_SCALE)

    # Test that ADMM update with NN constraints finds the correct A-matrix
    # when Bi-s and C is correct but A is incorrect
    nn_A = tl.clip(A, 0, float("inf"))
    nn_B_is = [tl.clip(B_i, 0, float("inf")) for B_i in B_is]
    nn_C = tl.clip(C, 0, float("inf"))

    nn_A2 = tl.tensor(rng.uniform(size=tl.shape(nn_A)))
    nn_modified_cmf = coupled_matrices.CoupledMatrixFactorization((weights, (nn_A2, nn_B_is, nn_C)))
    nn_cmf = coupled_matrices.CoupledMatrixFactorization((weights, (nn_A, nn_B_is, nn_C)))
    nn = NonNegativity()
    nn_aux = nn.init_aux(nn_cmf.to_matrices(), rank, 0, rng)
    nn_dual = nn.init_dual(nn_cmf.to_matrices(), rank, 0, rng)
    out = decomposition.admm_update_A(
        matrices=nn_cmf.to_matrices(),
        reg=[nn],
        cmf=nn_modified_cmf,
        A_aux_list=[nn_aux],
        A_dual_list=[nn_dual],
        l2_penalty=0,
        inner_n_iter_max=1_000,
        inner_tol=-1,
        feasibility_penalty_scale=feasibility_penalty_scale,
        constant_feasibility_penalty=constant_feasibility_penalty,
        svd_fun=get_svd("truncated_svd"),
    )
    out_nn_cmf, out_nn_A_auxes, out_nn_A_duals = out
    out_nn_A_normalized = normalize(out_nn_cmf[1][0])
    nn_A_normalized = normalize(nn_A)
    assert_allclose(out_nn_A_normalized, nn_A_normalized, rtol=1e-5 * RTOL_SCALE)

    # Check that the feasibility gap is low
    assert_allclose(out_nn_cmf[1][0], out_nn_A_auxes[0])

    # Test for l2_penalty > 0 by constructing and solving the regularized normal equations
    X = cmf.to_matrices()
    out = decomposition.admm_update_A(
        matrices=X,
        reg=[],
        cmf=modified_cmf,
        A_aux_list=[],
        A_dual_list=[],
        l2_penalty=1,
        inner_n_iter_max=1000,
        inner_tol=-1,
        feasibility_penalty_scale=feasibility_penalty_scale,
        constant_feasibility_penalty=constant_feasibility_penalty,
        svd_fun=get_svd("truncated_svd"),
    )
    out_cmf, out_A_auxes, out_A_duals = out
    out_A = out_cmf[1][0]
    for a_i, X_i, B_i in zip(out_A, X, B_is):
        lhs = tl.dot(tl.transpose(B_i), B_i) * tl.dot(tl.transpose(C), C) + tl.eye(rank)
        rhs = tl.diag(tl.dot(tl.dot(tl.transpose(B_i), X_i), C))
        assert_allclose(a_i, tl.solve(lhs, rhs), rtol=1e-6 * RTOL_SCALE)


@pytest.mark.parametrize(
    "feasibility_penalty_scale,constant_feasibility_penalty", all_combinations([0.5, 1, 2], [True, False])
)
def test_admm_update_B(rng, random_ragged_cmf, feasibility_penalty_scale, constant_feasibility_penalty):
    cmf, shapes, rank = random_ragged_cmf
    weights, (A, B_is, C) = cmf

    # Expand the C-matrix to have enough measurements per element of B (this makes the system easier to solve)
    new_K = 10 * (rank + max(shape[0] for shape in shapes) + A.shape[0])
    C = tl.tensor(rng.standard_normal((new_K, rank)))
    shapes = [(shape[0], new_K) for shape in shapes]
    cmf = coupled_matrices.CoupledMatrixFactorization((weights, (A, B_is, C)))

    # Test that ADMM update with no constraints finds the correct B_i-matrices
    # when A and C is correct but B_is is incorrect
    B_is2 = [tl.tensor(rng.uniform(size=tl.shape(B_i))) for B_i in B_is]
    modified_cmf = coupled_matrices.CoupledMatrixFactorization((weights, (A, B_is2, C)))

    out = decomposition.admm_update_B(
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
        assert_allclose(B_i, out_B_i, rtol=1e-5 * RTOL_SCALE)

    # WITH NON NEGATIVITY
    # Test that ADMM update with NN constraints finds the correct B_i-matrices
    # when A and C is correct but B_is is incorrect
    nn_A = tl.clip(A, 0.1, float("inf"))  # Increasing A improves the conditioning of the system
    nn_B_is = [tl.clip(B_i, 0, float("inf")) for B_i in B_is]
    nn_C = tl.clip(C, 0, float("inf"))

    nn_cmf = coupled_matrices.CoupledMatrixFactorization((weights, (nn_A, nn_B_is, nn_C)))
    nn_modified_cmf = coupled_matrices.CoupledMatrixFactorization((weights, (nn_A, B_is2, nn_C)))
    nn = NonNegativity()
    nn_aux = nn.init_aux(nn_cmf.to_matrices(), rank, 1, rng)
    nn_dual = nn.init_dual(nn_cmf.to_matrices(), rank, 1, rng)
    out = decomposition.admm_update_B(
        matrices=nn_cmf.to_matrices(),
        reg=[nn],
        cmf=nn_modified_cmf,
        B_is_aux_list=[nn_aux],
        B_is_dual_list=[nn_dual],
        l2_penalty=0,
        inner_n_iter_max=5_000,
        inner_tol=-1,
        feasibility_penalty_scale=feasibility_penalty_scale,
        constant_feasibility_penalty=constant_feasibility_penalty,
        svd_fun=get_svd("truncated_svd"),
    )
    out_nn_cmf, out_nn_B_auxes, out_nn_B_duals = out
    out_nn_B_is = out_nn_cmf[1][1]
    out_nn_B_is_normalized = [normalize(B_i) for B_i in out_nn_B_is]
    nn_B_is_normalized = [normalize(B_i) for B_i in nn_B_is]

    for B_i, out_B_i in zip(nn_B_is_normalized, out_nn_B_is_normalized):
        assert_allclose(B_i, out_B_i, rtol=1e-5 * RTOL_SCALE)

    # Check that the feasibility gap is low
    for aux_B_i, out_B_i in zip(out_nn_B_auxes[0], out_nn_B_is):
        assert_allclose(aux_B_i, out_B_i, rtol=1e-6 * RTOL_SCALE)

    # Test for l2_penalty > 0 by constructing and solving the regularized normal equations
    X = cmf.to_matrices()
    out = decomposition.admm_update_B(
        matrices=X,
        reg=[],
        cmf=modified_cmf,
        B_is_aux_list=[],
        B_is_dual_list=[],
        l2_penalty=1,
        inner_n_iter_max=1000,
        inner_tol=-1,
        feasibility_penalty_scale=feasibility_penalty_scale,
        constant_feasibility_penalty=constant_feasibility_penalty,
        svd_fun=get_svd("truncated_svd"),
    )
    out_cmf, out_B_auxes, out_B_duals = out
    out_B_is = out_cmf[1][1]
    out_B_is_normalized = [normalize(B_i) for B_i in out_B_is]

    for a_i, X_i, B_i in zip(A, X, out_B_is):
        lhs = tl.dot(tl.transpose(a_i * C), (a_i * C)) + tl.eye(rank)
        rhs = tl.transpose(tl.dot(X_i, a_i * C))
        assert_allclose(B_i, tl.transpose(tl.solve(lhs, rhs)), rtol=1e-5 * RTOL_SCALE)


@pytest.mark.parametrize("feasibility_penalty_scale", [0.5, 1, 2])
def test_admm_update_C(rng, random_ragged_cmf, feasibility_penalty_scale):
    cmf, shapes, rank = random_ragged_cmf
    weights, (A, B_is, C) = cmf

    # Test that ADMM update with no constraints finds the correct C-matrix
    # when Bi-s and A is correct but C is incorrect
    C2 = tl.tensor(rng.uniform(size=tl.shape(C)))
    modified_cmf = coupled_matrices.CoupledMatrixFactorization((weights, (A, B_is, C2)))

    out = decomposition.admm_update_C(
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
    assert_allclose(out_C_normalized, C_normalized, rtol=1e-6 * RTOL_SCALE)

    # Test that ADMM update with NN constraints finds the correct C-matrix
    # when Bi-s and A is correct but C is incorrect
    nn_A = tl.clip(A, 0, float("inf"))
    nn_B_is = [tl.clip(B_i, 0, float("inf")) for B_i in B_is]
    nn_C = tl.clip(C, 0, float("inf"))
    nn_cmf = coupled_matrices.CoupledMatrixFactorization((weights, (nn_A, nn_B_is, nn_C)))

    nn_C2 = tl.tensor(rng.uniform(size=tl.shape(nn_C)))
    nn_modified_cmf = coupled_matrices.CoupledMatrixFactorization((weights, (nn_A, nn_B_is, nn_C2)))

    nn = NonNegativity()
    nn_aux = nn.init_aux(nn_cmf.to_matrices(), rank, 2, rng)
    nn_dual = nn.init_dual(nn_cmf.to_matrices(), rank, 2, rng)
    out = decomposition.admm_update_C(
        matrices=nn_cmf.to_matrices(),
        reg=[nn],
        cmf=nn_modified_cmf,
        C_aux_list=[nn_aux],
        C_dual_list=[nn_dual],
        l2_penalty=0,
        inner_n_iter_max=1_000,
        inner_tol=-1,
        feasibility_penalty_scale=feasibility_penalty_scale,
        svd_fun=get_svd("truncated_svd"),
    )
    nn_out_cmf, nn_out_C_auxes, nn_out_C_duals = out
    out_nn_C_normalized = normalize(nn_out_cmf[1][2])
    nn_C_normalized = normalize(nn_C)
    assert_allclose(out_nn_C_normalized, nn_C_normalized, rtol=1e-5 * RTOL_SCALE)

    # Check that the feasibility gap is low
    assert_allclose(nn_out_cmf[1][2], nn_out_C_auxes[0])

    # Test for l2_penalty > 0 by constructing and solving the regularized normal equations
    X = cmf.to_matrices()
    out = decomposition.admm_update_C(
        matrices=X,
        reg=[],
        cmf=modified_cmf,
        C_aux_list=[],
        C_dual_list=[],
        l2_penalty=1,
        inner_n_iter_max=1000,
        inner_tol=-1,
        feasibility_penalty_scale=feasibility_penalty_scale,
        svd_fun=get_svd("truncated_svd"),
    )
    out_C = out[0][1][2]
    lhs = tl.eye(rank)
    rhs = 0
    for a_i, X_i, B_i in zip(A, X, B_is):
        lhs += tl.dot(tl.transpose(a_i * B_i), a_i * B_i)
        rhs += tl.dot(tl.transpose(a_i * B_i), X_i)
    assert_allclose(out_C, tl.transpose(tl.solve(lhs, rhs)), rtol=1e-6 * RTOL_SCALE)


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
    out_cmf, (aux, dual), diagnostics = decomposition.cmf_aoadmm(
        matrices, rank, n_iter_max=5_000, return_errors=True, return_admm_vars=True,
    )

    # Check that reconstruction error is low
    assert diagnostics.rec_errors[-1] < 1e-02

    # Add non-negativity constraints on all modes
    out_cmf, (aux, dual), diagnostics = decomposition.cmf_aoadmm(
        matrices, rank, n_iter_max=5_000, return_errors=True, return_admm_vars=True, non_negative=[True, True, True]
    )

    # Check that reconstruction error is low
    assert diagnostics.rec_errors[-1] < 1e-02

    # Check that we get errors out when we ask for errors. Even if convergence checking is disabled and verbose=False
    out = decomposition.cmf_aoadmm(
        matrices, rank, return_errors=True, return_admm_vars=False, tol=None, absolute_tol=None, verbose=False
    )
    assert len(out) == 2
    assert isinstance(out[0], CoupledMatrixFactorization)
    assert isinstance(out[1], decomposition.DiagnosticMetrics)

    # Check that we get errors and ADMM-vars when we ask for errors.
    # Even if convergence checking is disabled and verbose=False
    out = decomposition.cmf_aoadmm(
        matrices, rank, return_errors=True, return_admm_vars=True, tol=None, absolute_tol=None, verbose=False
    )
    assert len(out) == 3
    assert isinstance(out[0], CoupledMatrixFactorization)
    assert isinstance(out[1], decomposition.AdmmVars)
    assert isinstance(out[2], decomposition.DiagnosticMetrics)

    # Check that we don't get errors but do get ADMM-vars when we ask for it.
    # Even if convergence checking is disabled and verbose=False
    out = decomposition.cmf_aoadmm(
        matrices, rank, return_errors=False, return_admm_vars=True, tol=None, absolute_tol=None, verbose=False
    )
    assert len(out) == 2
    assert isinstance(out[0], CoupledMatrixFactorization)
    assert isinstance(out[1], decomposition.AdmmVars)

    # Check that we don't get errors out if we don't ask for it
    out = decomposition.cmf_aoadmm(matrices, rank, return_errors=False, return_admm_vars=False)
    assert len(out) == 2
    assert isinstance(out, CoupledMatrixFactorization)

    # Check that we can add non-negativity constraints with list of regs.
    regs = [[NonNegativity()], [NonNegativity()], [NonNegativity()]]
    out_cmf, (aux, dual), diagnostics = decomposition.cmf_aoadmm(
        matrices, rank, n_iter_max=10, return_errors=True, return_admm_vars=True, regs=regs
    )
    # Check that final reconstruction error is the same as when we compute it with the returned decomposition and auxes
    assert decomposition._cmf_reconstruction_error(matrices, out_cmf) / norm_matrices == pytest.approx(
        diagnostics.rec_errors[-1]
    )

    # Check that feasibility gaps are the same as when we compute it with the returned decomposition and auxes
    A_gap_list, B_gap_list, C_gap_list = decomposition.compute_feasibility_gaps(out_cmf, regs, *aux)
    for A_gap, out_A_gap in zip(A_gap_list, diagnostics.feasibility_gaps[-1][0]):
        assert A_gap == pytest.approx(out_A_gap)
    for B_gap, out_B_gap in zip(B_gap_list, diagnostics.feasibility_gaps[-1][1]):
        assert B_gap == pytest.approx(out_B_gap)
    for C_gap, out_C_gap in zip(C_gap_list, diagnostics.feasibility_gaps[-1][2]):
        assert C_gap == pytest.approx(out_C_gap)

    # Test that the code fails gracefully with list of regs not list of list of regs
    list_of_regs = [0.1, 0.1, 0.1]
    with pytest.raises(TypeError):
        out_cmf, (aux, dual), diagnostics = decomposition.cmf_aoadmm(
            matrices, rank, n_iter_max=1, return_errors=True, return_admm_vars=True, regs=list_of_regs
        )

    # Test that the code fails gracefully with list of lists containting something other than ADMMPenalty
    list_of_regs = [[1], [], []]
    with pytest.raises(TypeError):
        out_cmf, (aux, dual), diagnostics = decomposition.cmf_aoadmm(
            matrices, rank, n_iter_max=1, return_errors=True, return_admm_vars=True, regs=list_of_regs
        )

    # Test that we get correct amount of auxes and duals with only one constraint on B
    regs = [[NonNegativity()], [NonNegativity(), L2Ball(1)], [NonNegativity(), L2Ball(1), BoxConstraint(0, 1)]]
    out_cmf, (aux, dual), diagnostics = decomposition.cmf_aoadmm(
        matrices, rank, n_iter_max=10, return_errors=True, return_admm_vars=True, regs=regs
    )
    assert len(aux) == 3
    assert len(dual) == 3
    assert len(aux[0]) == 1
    assert len(dual[0]) == 1
    assert len(aux[1]) == 2
    assert len(dual[1]) == 2
    assert len(aux[2]) == 3
    assert len(dual[2]) == 3

    # Test that feasibility_tol can be None
    regs = [[], [NonNegativity()], []]
    out_cmf, diagnostics = decomposition.cmf_aoadmm(
        matrices, rank, n_iter_max=10, return_errors=True, regs=regs, feasibility_tol=None
    )

    # Test that feasibility gap is computed even if tol = None and absolute_tol = None
    regs = [[], [NonNegativity()], []]
    out_cmf, diagnostics = decomposition.cmf_aoadmm(
        matrices, rank, n_iter_max=10, return_errors=True, regs=regs, tol=None, absolute_tol=None
    )
    assert diagnostics.satisfied_feasibility_condition is not None


def test_cmf_aoadmm_verbose(rng, random_ragged_cmf, capfd):
    cmf, shapes, rank = random_ragged_cmf
    matrices = cmf.to_matrices()

    # Check that verbose = False results in no print
    decomposition.cmf_aoadmm(matrices, rank, n_iter_max=10, return_errors=True, return_admm_vars=True, verbose=False)
    out, err = capfd.readouterr()
    assert len(out) == 0

    # Check that verbose = True results in print when return_errors = True
    decomposition.cmf_aoadmm(matrices, rank, n_iter_max=10, return_errors=True, return_admm_vars=True, verbose=True)
    out, err = capfd.readouterr()
    assert len(out) > 0

    # Check that verbose = True results in print when return_errors = False and
    decomposition.cmf_aoadmm(
        matrices,
        rank,
        n_iter_max=10,
        return_errors=False,
        return_admm_vars=True,
        verbose=True,
        feasibility_tol=-1,
        non_negative=True,
    )
    out, err = capfd.readouterr()
    assert len(out) > 0


def test_parafac2_makes_nn_cmf_unique(rng):
    rank = 2
    A = tl.tensor(rng.uniform(0.1, 1.1, size=(10, rank)))
    B_0 = rng.uniform(0, 1, size=(7, rank))
    B_is = [tl.tensor(np.roll(B_0, i, axis=1)) for i in range(10)]
    C = tl.tensor(rng.uniform(0, 1, size=(10, rank)))
    weights = None

    cmf = CoupledMatrixFactorization((weights, (A, B_is, C)))
    matrices = cmf.to_matrices()

    regularized_loss = [float("inf")]
    for init in range(5):
        out, diagnostics = decomposition.cmf_aoadmm(
            matrices, rank, n_iter_max=1_000, return_errors=True, non_negative=[True, True, True], parafac2=True,
        )

        if diagnostics.regularized_loss[-1] < regularized_loss[-1] and diagnostics.satisfied_feasibility_condition:
            out_cmf = out
            regularized_loss = diagnostics.regularized_loss

    # Check that reconstruction error is low and that the correct factors are recovered
    assert regularized_loss[-1] < 1e-5

    # Low congruence coefficient tolerance to account for single precision (pytorch) being less accurate
    assert congruence_coefficient(A, out_cmf[1][0], absolute_value=True)[0] > 0.95
    for B_i, out_B_i in zip(B_is, out_cmf[1][1]):
        assert congruence_coefficient(B_i, out_B_i, absolute_value=True)[0] > 0.95
    assert congruence_coefficient(C, out_cmf[1][2], absolute_value=True)[0] > 0.95


def test_cmf_aoadmm_not_updating_A_works(rng, random_rank5_ragged_cmf):
    cmf, shapes, rank = random_rank5_ragged_cmf
    weights, (A, B_is, C) = cmf
    wrong_A = tl.tensor(rng.random_sample(tl.shape(A)))
    wrong_A_copy = tl.copy(wrong_A)
    B_is_copy = [tl.copy(B_i) for B_i in B_is]
    C_copy = tl.copy(C)

    # Construct matrices and compute their norm
    matrices = cmf.to_matrices()

    # Decompose matrices with cmf_aoadmm with no constraints
    out_cmf = decomposition.cmf_aoadmm(
        matrices, rank, n_iter_max=5, update_A=False, init=(None, (wrong_A_copy, B_is_copy, C_copy)),
    )

    out_weights, (out_A, out_B_is, out_C) = out_cmf
    assert_allclose(wrong_A, out_A)
    assert not np.all(tl.to_numpy(out_C) == tl.to_numpy(C))
    for B_i, out_B_i in zip(B_is, out_B_is):
        assert not np.all(tl.to_numpy(B_i) == tl.to_numpy(out_B_i))


def test_cmf_aoadmm_not_updating_C_works(rng, random_rank5_ragged_cmf):
    cmf, shapes, rank = random_rank5_ragged_cmf
    weights, (A, B_is, C) = cmf
    A_copy = tl.copy(A)
    B_is_copy = [tl.copy(B_i) for B_i in B_is]
    wrong_C = tl.tensor(rng.random_sample(tl.shape(C)))
    wrong_C_copy = tl.copy(wrong_C)

    # Construct matrices and compute their norm
    matrices = cmf.to_matrices()

    # Decompose matrices with cmf_aoadmm with no constraints
    out_cmf = decomposition.cmf_aoadmm(
        matrices, rank, n_iter_max=5, update_C=False, init=(None, (A_copy, B_is_copy, wrong_C_copy)),
    )

    out_weights, (out_A, out_B_is, out_C) = out_cmf
    assert_allclose(wrong_C, out_C)
    assert not np.all(tl.to_numpy(out_A) == tl.to_numpy(A))
    for B_i, out_B_i in zip(B_is, out_B_is):
        assert not np.all(tl.to_numpy(B_i) == tl.to_numpy(out_B_i))


def test_cmf_aoadmm_not_updating_B_is_works(rng, random_rank5_ragged_cmf):
    cmf, shapes, rank = random_rank5_ragged_cmf
    weights, (A, B_is, C) = cmf
    A_copy = tl.copy(A)
    wrong_B_is = [tl.tensor(rng.standard_normal(size=tl.shape(B_i))) for B_i in B_is]
    wrong_B_is_copy = [tl.copy(B_i) for B_i in wrong_B_is]
    C_copy = tl.copy(C)
    # Construct matrices and compute their norm
    matrices = cmf.to_matrices()

    # Decompose matrices with cmf_aoadmm with no constraints
    out_cmf = decomposition.cmf_aoadmm(
        matrices, rank, n_iter_max=5, update_B_is=False, init=(None, (A_copy, wrong_B_is_copy, C_copy)),
    )

    out_weights, (out_A, out_B_is, out_C) = out_cmf
    assert not np.all(tl.to_numpy(out_C) == tl.to_numpy(C))
    assert not np.all(tl.to_numpy(out_A) == tl.to_numpy(A))
    for B_i, out_B_i in zip(wrong_B_is, out_B_is):
        assert_allclose(B_i, out_B_i)


def test_parafac2_aoadmm(rng, random_ragged_cmf):
    parafac2_argspecs = inspect.getfullargspec(matcouply.decomposition.parafac2_aoadmm)
    placeholder_args = {arg: f"PLACEHOLDERARG_{i}" for i, arg in enumerate(parafac2_argspecs.args)}
    cmf_aoadmm_args = copy(placeholder_args)
    cmf_aoadmm_args["parafac2"] = True
    with patch("matcouply.decomposition.cmf_aoadmm") as mock:
        decomposition.parafac2_aoadmm(**placeholder_args)
        mock.assert_called_once()
        mock.assert_called_once_with(**cmf_aoadmm_args)


def test_compute_l2_penalty(rng, random_ragged_cmf):
    cmf, shapes, rank = random_ragged_cmf
    weights, (A, B_is, C) = cmf

    SS_A = tl.sum(A ** 2)
    SS_B = sum(tl.sum(B_i ** 2) for B_i in B_is)
    SS_C = tl.sum(C ** 2)

    assert decomposition._compute_l2_penalty(cmf, [0, 0, 0]) == 0
    assert decomposition._compute_l2_penalty(cmf, [1, 0, 0]) == pytest.approx(0.5 * SS_A)
    assert decomposition._compute_l2_penalty(cmf, [0, 1, 0]) == pytest.approx(0.5 * SS_B)
    assert decomposition._compute_l2_penalty(cmf, [0, 0, 1]) == pytest.approx(0.5 * SS_C)
    assert decomposition._compute_l2_penalty(cmf, [1, 1, 1]) == pytest.approx(0.5 * (SS_A + SS_B + SS_C))
    assert decomposition._compute_l2_penalty(cmf, [1, 2, 3]) == pytest.approx(0.5 * (SS_A + 2 * SS_B + 3 * SS_C))


def test_l2_penalty_is_included(rng, random_ragged_cmf):
    cmf, shapes, rank = random_ragged_cmf
    matrices = cmf.to_matrices()

    # Decompose matrices with cmf_aoadmm with no constraints
    out_cmf, diagnostics = decomposition.cmf_aoadmm(
        matrices, rank, n_iter_max=5, return_errors=True, update_B_is=False,
    )

    rel_sse = diagnostics.rec_errors[-1] ** 2
    assert diagnostics.regularized_loss[-1] == pytest.approx(0.5 * rel_sse)

    out_cmf, diagnostics = decomposition.cmf_aoadmm(
        matrices, rank, n_iter_max=5, l2_penalty=1, return_errors=True, update_B_is=False,
    )

    out_weights, (out_A, out_B_is, out_C) = out_cmf
    rel_sse = diagnostics.rec_errors[-1] ** 2
    SS_A = tl.sum(out_A ** 2)
    SS_B = sum(tl.sum(out_B_i ** 2) for out_B_i in out_B_is)
    SS_C = tl.sum(out_C ** 2)
    assert diagnostics.regularized_loss[-1] == pytest.approx(0.5 * rel_sse + 0.5 * (SS_A + SS_B + SS_C))

    out_cmf, diagnostics = decomposition.cmf_aoadmm(
        matrices, rank, n_iter_max=5, l2_penalty=[1, 2, 3], return_errors=True, update_B_is=False,
    )

    out_weights, (out_A, out_B_is, out_C) = out_cmf
    rel_sse = diagnostics.rec_errors[-1] ** 2
    SS_A = tl.sum(out_A ** 2)
    SS_B = sum(tl.sum(out_B_i ** 2) for out_B_i in out_B_is)
    SS_C = tl.sum(out_C ** 2)
    assert diagnostics.regularized_loss[-1] == pytest.approx(0.5 * rel_sse + 0.5 * (1 * SS_A + 2 * SS_B + 3 * SS_C))


def test_cmf_aoadmm_stopping_information(random_ragged_cmf):
    cmf, shapes, rank = random_ragged_cmf
    matrices = cmf.to_matrices()
    n_iter_max = 10

    # Check that we get correct output when none of the conditions are met
    out_cmf, diagnostics = decomposition.cmf_aoadmm(
        matrices,
        rank,
        n_iter_max=n_iter_max,
        return_errors=True,
        verbose=False,
        tol=-float("inf"),
        absolute_tol=-float("inf"),
        feasibility_tol=-float("inf"),
        non_negative=True,
    )

    assert not diagnostics.satisfied_stopping_condition
    assert not diagnostics.satisfied_feasibility_condition
    assert diagnostics.message == "MAXIMUM NUMBER OF ITERATIONS REACHED"
    assert len(diagnostics.regularized_loss) == n_iter_max + 1
    assert len(diagnostics.rec_errors) == n_iter_max + 1
    assert len(diagnostics.feasibility_gaps) == n_iter_max + 1

    # Check that we get correct output when only the feasibility conditions are met
    out_cmf, diagnostics = decomposition.cmf_aoadmm(
        matrices,
        rank,
        n_iter_max=n_iter_max,
        return_errors=True,
        verbose=False,
        tol=-float("inf"),
        absolute_tol=-float("inf"),
        feasibility_tol=float("inf"),
        non_negative=True,
    )

    assert not diagnostics.satisfied_stopping_condition
    assert diagnostics.satisfied_feasibility_condition
    assert diagnostics.message == "MAXIMUM NUMBER OF ITERATIONS REACHED"
    assert len(diagnostics.regularized_loss) == n_iter_max + 1
    assert len(diagnostics.rec_errors) == n_iter_max + 1
    assert len(diagnostics.feasibility_gaps) == n_iter_max + 1

    # Check that we get correct output when only the feasibility conditions and the relative loss tolerance are met
    out_cmf, diagnostics = decomposition.cmf_aoadmm(
        matrices,
        rank,
        n_iter_max=n_iter_max,
        return_errors=True,
        verbose=False,
        tol=float("inf"),
        absolute_tol=-float("inf"),
        feasibility_tol=float("inf"),
        non_negative=True,
    )

    assert diagnostics.satisfied_stopping_condition
    assert diagnostics.satisfied_feasibility_condition
    assert diagnostics.message == "FEASIBILITY GAP CRITERION AND RELATIVE LOSS CRITERION SATISFIED"
    assert len(diagnostics.regularized_loss) == 2
    assert len(diagnostics.rec_errors) == 2
    assert len(diagnostics.feasibility_gaps) == 2

    # Check that we get correct output when only the feasibility conditions and the absolute loss tolerance are met
    out_cmf, diagnostics = decomposition.cmf_aoadmm(
        matrices,
        rank,
        n_iter_max=n_iter_max,
        return_errors=True,
        verbose=False,
        tol=-float("inf"),
        absolute_tol=float("inf"),
        feasibility_tol=float("inf"),
        non_negative=True,
    )

    assert diagnostics.satisfied_stopping_condition
    assert diagnostics.satisfied_feasibility_condition
    assert diagnostics.message == "FEASIBILITY GAP CRITERION AND ABSOLUTE LOSS CRITERION SATISFIED"
    assert len(diagnostics.regularized_loss) == 2
    assert len(diagnostics.rec_errors) == 2
    assert len(diagnostics.feasibility_gaps) == 2


def test_first_loss_value_is_correct(random_ragged_cmf):
    cmf, shapes, rank = random_ragged_cmf
    matrices = cmf.to_matrices()
    n_iter_max = 0

    weights, (A, B_is, C) = cmf
    l1_A = tl.sum(tl.abs(weights * A))
    l1_B = sum(tl.sum(tl.abs(B_i)) for B_i in B_is)
    l1_C = tl.sum(tl.abs(C))
    l1_reg_penalty = l1_A + l1_B + l1_C
    l2_A = tl.sum((weights * A) ** 2)
    l2_B = sum(tl.sum(B_i ** 2) for B_i in B_is)
    l2_C = tl.sum(C ** 2)
    l2_reg_penalty = l2_A + l2_B + l2_C
    rec_error = 0
    initial_loss = 0.5 * rec_error ** 2 + 0.5 * 0.1 * l2_reg_penalty + 0.2 * l1_reg_penalty

    # Check that we get correct output when none of the conditions are met
    out_cmf, diagnostics = decomposition.cmf_aoadmm(
        matrices,
        rank,
        init=cmf,
        n_iter_max=n_iter_max,
        return_errors=True,
        verbose=False,
        l1_penalty=0.2,
        l2_penalty=0.1,
    )
    loss = diagnostics.regularized_loss
    assert len(loss) == 1
    assert loss[0] == pytest.approx(initial_loss)


@pytest.mark.parametrize(
    "n_iter_max", [-1, 0],
)
def test_cmf_aoadmm_works_with_zero_iteration(random_ragged_cmf, n_iter_max):
    cmf, shapes, rank = random_ragged_cmf
    matrices = cmf.to_matrices()
    out_cmf, admm_vars, diagnostics = decomposition.cmf_aoadmm(
        matrices, rank, init=cmf, n_iter_max=n_iter_max, return_errors=True, return_admm_vars=True
    )
    assert len(diagnostics.regularized_loss) == 1
    assert len(diagnostics.rec_errors) == 1
    assert len(diagnostics.feasibility_gaps) == 1
    assert diagnostics.n_iter == 0


def test_returns_feasibility_info_with_no_tol(random_ragged_cmf):
    cmf, shapes, rank = random_ragged_cmf
    matrices = cmf.to_matrices()
    out_cmf, diagnostics = decomposition.cmf_aoadmm(
        matrices, rank, n_iter_max=10, return_errors=True, tol=None, absolute_tol=None, non_negative=True
    )
    assert diagnostics.satisfied_feasibility_condition is not None
    assert len(diagnostics.feasibility_gaps) == 11
