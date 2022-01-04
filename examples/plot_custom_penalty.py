"""
Example with custom penalty class for unimodality for all but one component
---------------------------------------------------------------------------

In this example, we first demonstrate how to specify exactly how the penalties are imposed in the AO-ADMM fitting
procedure. Then, we create a custom penalty that imposes non-negativity on all component vectors and unimodality on all
but one of the component vectors.
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import tensorly as tl
from component_vis.factor_tools import factor_match_score

import matcouply.decomposition as decomposition
from matcouply.coupled_matrices import CoupledMatrixFactorization

###############################################################################
# Setup
# ^^^^^

I, J, K = 10, 40, 20
rank = 3
noise_level = 0.2
rng = np.random.default_rng(0)


def normalize(x):
    return x / tl.sqrt(tl.sum(x ** 2, axis=0, keepdims=True))


###############################################################################
# Generate simulated data that follows the PARAFAC2 constraint
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# We start by generating some components, for the :math:`\mathbf{A}` and :math:`\mathbf{C}` components, we use uniformly
# distributed component vector elements. For the :math:`\mathbf{B}_i`-components, we create two unimodal vectors and one
# component vector with uniformly distributed elements, and shift these vectors for each :math:`i`.

# Random uniform components
A = rng.uniform(size=(I, rank)) + 0.1  # Add 0.1 to ensure that there is signal for all components for all slices
A = tl.tensor(A)

B_0 = tl.zeros((J, rank))
# Simulating unimodal components
t = np.linspace(-10, 10, J)
for r in range(rank - 1):
    sigma = rng.uniform(0.5, 1.5)
    mu = rng.uniform(-10, 0)
    B_0[:, r] = stats.norm.pdf(t, loc=mu, scale=sigma)
# The final component is random uniform, not unimodal
B_0[:, rank - 1] = rng.uniform(size=(J,))

# Shift the components for each slice
B_is = [np.roll(B_0, i, axis=0) for i in range(I)]
B_is = [tl.tensor(B_i) for B_i in B_is]


# Random uniform components
C = rng.uniform(size=(K, rank))
C = tl.tensor(C)

###############################################################################
# Plot the simulated components
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

fig, axes = plt.subplots(2, 3, tight_layout=True)

axes[0, 0].plot(normalize(A))
axes[0, 0].set_title("$\\mathbf{A}$")

axes[0, 1].plot(normalize(C))
axes[0, 1].set_title("$\\mathbf{C}$")

axes[0, 2].axis("off")

axes[1, 0].plot(normalize(B_is[0]))
axes[1, 0].set_title("$\\mathbf{B}_0$")

axes[1, 1].plot(normalize(B_is[I // 2]))
axes[1, 1].set_title(f"$\\mathbf{{B}}_{{{I//2}}}$")

axes[1, 2].plot(normalize(B_is[-1]))
axes[1, 2].set_title(f"$\\mathbf{{B}}_{{{I-1}}}$")
fig.legend(["Component 0", "Component 1", "Component 2"], bbox_to_anchor=(0.95, 0.75), loc="center right")
fig.suptitle("Simulated components")

plt.show()

###############################################################################
# For the :math:`\mathbf{B}_i`-s, we see that component 0 and 1 are unimodal, while component 2 is not.

###############################################################################
# Create the coupled matrix factorization, simulated data matrices and add noise
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

cmf = CoupledMatrixFactorization((None, (A, B_is, C)))
matrices = cmf.to_matrices()
noise = [tl.tensor(rng.uniform(size=M.shape)) for M in matrices]
noisy_matrices = [M + N * noise_level * tl.norm(M) / tl.norm(N) for M, N in zip(matrices, noise)]


###############################################################################
# Use the ``regs`` parameter to input regularisation classes
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Matcouply automatically parses the constraints from the ``parafac2_aoadmm`` and ``cmf_aoadmm`` funciton
# arguments. However, sometimes, you may want full control over how a penalty is implemented. In that case,
# the ``regs``-argument is useful. This argument makes it possible to specify exactly which penalty instances
# to use.
#
# Since the components are non-negative, it makes sense to fit a non-negative PARAFAC2 model, however,
# we also know that two of the :math:`\mathbf{B}_i`-component vectors are unimodal, so we first try with
# a fully unimodal decomposition.

from matcouply.penalties import NonNegativity, Unimodality

lowest_error = float("inf")
for init in range(4):
    print("Init:", init)
    out, diagnostics = decomposition.parafac2_aoadmm(
        noisy_matrices,
        rank,
        n_iter_max=1000,
        regs=[[NonNegativity()], [Unimodality(non_negativity=True)], [NonNegativity()]],
        return_errors=True,
        random_state=init,
    )
    if diagnostics.regularised_loss[-1] < lowest_error and len(diagnostics.rec_errors) < 1000:
        out_cmf = out
        rec_errors, feasibility_gaps, regularised_loss = diagnostics
        lowest_error = rec_errors[-1]

print("=" * 50)
print(f"Final reconstruction error: {lowest_error:.3f}")
print(f"Feasibility gap for A: {feasibility_gaps[-1][0]}")
print(f"Feasibility gap for B_is: {feasibility_gaps[-1][1]}")
print(f"Feasibility gap for C: {feasibility_gaps[-1][2]}")

###############################################################################
# Compute factor match score to measure the accuracy of the recovered components
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


def get_stacked_CP_tensor(cmf):
    weights, factors = cmf
    A, B_is, C = factors

    stacked_cp_tensor = (weights, (A, np.concatenate(B_is, axis=0), C))
    return stacked_cp_tensor


fms, permutation = factor_match_score(
    get_stacked_CP_tensor(cmf), get_stacked_CP_tensor(out_cmf), consider_weights=False, return_permutation=True
)
print(f"Factor match score: {fms}")

###############################################################################
# Plot the recovered components
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

out_weights, (out_A, out_B_is, out_C) = out_cmf
out_A = out_A[:, permutation]
out_B_is = [out_B_i[:, permutation] for out_B_i in out_B_is]
out_C = out_C[:, permutation]

fig, axes = plt.subplots(2, 3, tight_layout=True)

axes[0, 0].plot(normalize(out_A))
axes[0, 0].set_title("$\\mathbf{A}$")

axes[0, 1].plot(normalize(out_C))
axes[0, 1].set_title("$\\mathbf{C}$")

axes[0, 2].axis("off")

axes[1, 0].plot(normalize(out_B_is[0]))
axes[1, 0].set_title("$\\mathbf{B}_0$")

axes[1, 1].plot(normalize(out_B_is[I // 2]))
axes[1, 1].set_title(f"$\\mathbf{{B}}_{{{I//2}}}$")

axes[1, 2].plot(normalize(out_B_is[-1]))
axes[1, 2].set_title(f"$\\mathbf{{B}}_{{{I-1}}}$")
fig.legend(["Component 0", "Component 1", "Component 2"], bbox_to_anchor=(0.95, 0.75), loc="center right")

fig.suptitle(r"Unimodality on the $\mathbf{B}_i$-components")
plt.show()

###############################################################################
# We see that the :math:`\mathbf{C}`-component vectors all follow the same pattern and that the the
# :math:`\mathbf{A}`-component vectors all follow a similar pattern. This is not the case with the real,
# uncorrelated random, components. The :math:`\mathbf{B}_i`-component vectors also follow a strange pattern
# with peaks jumping forwards and backwards, which we know are not the case with the real components either.
#
# However, this strange behaviour is not too surprising, considering that there are only two uniomdal component
# vectors in the data. So this model that assumes all unimodal components might be too restrictive.


###############################################################################
# Create a custom penalty class for unimodality in all but one class
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Now, we want to make a custom penalty that imposes unimodality on the first :math:`R-1` component
# vectors. Since unimodality are imposed column-wise, we know that this constraint is a matrix penalty
# (as opposed to a row-vector penalty or a multi-matrix penalty), so we import the ``MatrixPenalty``-superclass
# from ``matcouply.penalties``. We also know that unimodality is a hard constraint, so we import
# the ``HardConstraintMixin``-class, which provides a ``penalty``-method that always returns 0 and has an informative
# docstring.

from matcouply._doc_utils import (
    copy_ancestor_docstring,  # Helper decorator that makes it possible for ADMMPenalties to inherit a docstring
)
from matcouply._unimodal_regression import unimodal_regression  # The unimodal regression implementation
from matcouply.penalties import HardConstraintMixin, MatrixPenalty


class CustomUnimodality(HardConstraintMixin, MatrixPenalty):
    def __init__(self, non_negativity=False, aux_init="random_uniform", dual_init="random_uniform"):
        super().__init__(aux_init, dual_init)
        self.non_negativity = non_negativity

    @copy_ancestor_docstring
    def factor_matrix_update(self, factor_matrix, feasibility_penalty, aux):
        new_factor_matrix = tl.copy(factor_matrix)
        new_factor_matrix[:, :-1] = unimodal_regression(factor_matrix[:, :-1], non_negativity=self.non_negativity)
        if self.non_negativity:
            new_factor_matrix = tl.clip(new_factor_matrix, 0)
        return new_factor_matrix


###############################################################################
# Fit a non-negative PARAFAC2 model using the custom penalty class on the B mode
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Now, we can fit a new model with the custom unimodality class

lowest_error = float("inf")
for init in range(4):
    print("Init:", init)
    out, diagnostics = decomposition.parafac2_aoadmm(
        noisy_matrices,
        rank,
        n_iter_max=1000,
        regs=[[NonNegativity()], [CustomUnimodality(non_negativity=True)], [NonNegativity()]],
        return_errors=True,
        random_state=init,
    )
    if diagnostics.regularised_loss[-1] < lowest_error and len(diagnostics.rec_errors) < 1000:
        out_cmf = out
        rec_errors, feasibility_gaps, regularised_loss = diagnostics
        lowest_error = rec_errors[-1]

print("=" * 50)
print(f"Final reconstruction error: {lowest_error:.3f}")
print(f"Feasibility gap for A: {feasibility_gaps[-1][0]}")
print(f"Feasibility gap for B_is: {feasibility_gaps[-1][1]}")
print(f"Feasibility gap for C: {feasibility_gaps[-1][2]}")

###############################################################################
# Compute factor match score to measure the accuracy of the recovered components
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

fms, permutation = factor_match_score(
    get_stacked_CP_tensor(cmf), get_stacked_CP_tensor(out_cmf), consider_weights=False, return_permutation=True
)
print(f"Factor match score: {fms}")

###############################################################################
# We see that the factor match score is much better now compared to before!

###############################################################################
# Plot the recovered components
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

out_weights, (out_A, out_B_is, out_C) = out_cmf
out_A = out_A[:, permutation]
out_B_is = [out_B_i[:, permutation] for out_B_i in out_B_is]
out_C = out_C[:, permutation]

fig, axes = plt.subplots(2, 3, tight_layout=True)

axes[0, 0].plot(normalize(out_A))
axes[0, 0].set_title("$\\mathbf{A}$")

axes[0, 1].plot(normalize(out_C))
axes[0, 1].set_title("$\\mathbf{C}$")

axes[0, 2].axis("off")

axes[1, 0].plot(normalize(out_B_is[0]))
axes[1, 0].set_title("$\\mathbf{B}_0$")

axes[1, 1].plot(normalize(out_B_is[I // 2]))
axes[1, 1].set_title(f"$\\mathbf{{B}}_{{{I//2}}}$")

axes[1, 2].plot(normalize(out_B_is[-1]))
axes[1, 2].set_title(f"$\\mathbf{{B}}_{{{I-1}}}$")
fig.legend(["Component 0", "Component 1", "Component 2"], bbox_to_anchor=(0.95, 0.75), loc="center right")
fig.suptitle(r"Custom uniomdality on the $\mathbf{B}_i$-components")
plt.show()

###############################################################################
# We see that the model finds much more sensible component vectors. The :math:`\mathbf{A}`- and
# :math:`\mathbf{C}`-component vectors no longer seem correlated, and the peaks of the :math:`\mathbf{B}_i`-component
# vectors no longer jump around.

