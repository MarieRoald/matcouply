"""
Simple example with simulated non-negative components.
------------------------------------------------------
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorly as tl
from tlviz.factor_tools import factor_match_score

import matcouply.decomposition as decomposition
from matcouply.coupled_matrices import CoupledMatrixFactorization

###############################################################################
# Setup
# ^^^^^

I, J, K = 10, 15, 20
rank = 3
noise_level = 0.2
rng = np.random.default_rng(0)


def truncated_normal(size):
    x = rng.standard_normal(size=size)
    x[x < 0] = 0
    return tl.tensor(x)


def normalize(x):
    return x / tl.sqrt(tl.sum(x**2, axis=0, keepdims=True))


###############################################################################
# Generate simulated data that follows the PARAFAC2 constraint
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
A = rng.uniform(size=(I, rank)) + 0.1  # Add 0.1 to ensure that there is signal for all components for all slices
A = tl.tensor(A)

B_blueprint = truncated_normal(size=(J, rank))
B_is = [np.roll(B_blueprint, i, axis=0) for i in range(I)]
B_is = [tl.tensor(B_i) for B_i in B_is]

C = truncated_normal(size=(K, rank))
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

plt.show()


###############################################################################
# Create the coupled matrix factorization, simulated data matrices and add noise
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

cmf = CoupledMatrixFactorization((None, (A, B_is, C)))
matrices = cmf.to_matrices()
noise = [tl.tensor(rng.uniform(size=M.shape)) for M in matrices]
noisy_matrices = [M + N * noise_level * tl.norm(M) / tl.norm(N) for M, N in zip(matrices, noise)]

###############################################################################
# Fit a non-negative PARAFAC2 model to the noisy data
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

lowest_error = float("inf")
for init in range(5):
    print("Init:", init)
    out = decomposition.parafac2_aoadmm(
        noisy_matrices, rank, n_iter_max=1000, non_negative=True, return_errors=True, random_state=init
    )
    if out[1].regularized_loss[-1] < lowest_error and out[1].satisfied_stopping_condition:
        out_cmf, diagnostics = out
        lowest_error = diagnostics.rec_errors[-1]

print("=" * 50)
print(f"Final reconstruction error: {lowest_error:.3f}")
print(f"Feasibility gap for A: {diagnostics.feasibility_gaps[-1][0]}")
print(f"Feasibility gap for B_is: {diagnostics.feasibility_gaps[-1][1]}")
print(f"Feasibility gap for C: {diagnostics.feasibility_gaps[-1][2]}")

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
# Plot the loss logg
# ^^^^^^^^^^^^^^^^^^

fig, ax = plt.subplots(tight_layout=True)
ax.semilogy(diagnostics.rec_errors)
plt.xlabel("Iteration")
plt.ylabel("Relative normed error (2-norm)")
plt.show()

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

plt.show()
