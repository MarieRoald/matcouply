"""
Examining effect of adding more components
------------------------------------------
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorly as tl

import matcouply.cmf_aoadmm as cmf_aoadmm
from matcouply.coupled_matrices import CoupledMatrixFactorization

###############################################################################
# Setup
# ^^^^^
I, J, K = 10, 20, 30
rank = 4
noise_level = 0.1
rng = np.random.default_rng(0)


def truncated_normal(size):
    x = rng.standard_normal(size=size)
    x[x < 0] = 0
    return tl.tensor(x)


def normalize(x):
    return x / tl.sqrt(tl.sum(x ** 2, axis=0, keepdims=True))


###############################################################################
# Generate simulated PARFAC2 factor matrices where the true number of components (`rank`) is known
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
A = rng.uniform(size=(I, rank)) + 0.1  # Add 0.1 to ensure that there is signal for all components for all slices
A = tl.tensor(A)

B_blueprint = truncated_normal(size=(J, rank))
B_is = [np.roll(B_blueprint, i, axis=0) for i in range(I)]
B_is = [tl.tensor(B_i) for B_i in B_is]

C = rng.standard_normal(size=(K, rank))
C = tl.tensor(C)

cmf = CoupledMatrixFactorization((None, (A, B_is, C)))

###############################################################################
# Create data marices from the decomposition and add noise
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

matrices = cmf.to_matrices()
noise = [tl.tensor(rng.uniform(size=M.shape)) for M in matrices]
noisy_matrices = [M + N * noise_level * tl.norm(M) / tl.norm(N) for M, N in zip(matrices, noise)]


###############################################################################
# Fit PARAFAC2 models with different number of components to the noisy data
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

fit_scores = []
B_gaps = []
A_gaps = []
for num_components in range(2, 7):
    lowest_error = float("inf")
    for init in range(5):
        cmf, diagnostics = cmf_aoadmm.parafac2_aoadmm(
            noisy_matrices,
            num_components,
            n_iter_max=1000,
            non_negative=[True, False, False],
            return_errors=True,
            random_state=init,
        )
        if diagnostics.regularised_loss[-1] < lowest_error:
            selected_cmf = cmf
            selected_diagnostics = diagnostics
            lowest_error = diagnostics.regularised_loss[-1]

    fit_score = 1 - lowest_error
    fit_scores.append(fit_score)
    B_gaps.append(selected_diagnostics.feasibility_gaps[-1][1][0])
    A_gaps.append(selected_diagnostics.feasibility_gaps[-1][0][0])


###############################################################################
# Create scree plots of fit score and feasability gaps for different number of components
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

fig, axes = plt.subplots(3, 1, tight_layout=True, sharex=True)
axes[0].set_title("Fit score")
axes[0].plot(range(2, 7), fit_scores)
axes[1].set_title("Feasibility gap for A  (NN constraint)")
axes[1].plot(range(2, 7), A_gaps)
axes[2].set_title("Feasibility gap for B_is (PF2 constraint)")
axes[2].plot(range(2, 7), B_gaps)
axes[2].set_xlabel("No. components")
axes[2].set_xticks(range(2, 7))
plt.show()

###############################################################################
# The top plot above shows that adding more components improves the fit in the beginning,
# but then the improvement lessens as we reach the "true" number of components.
# We know that the correct number of components is four for this simulated data,
# but if you work with a real dataset, you don't always know the "true" number.
# So then, examining such a plot can help you choose an appropriate number of components.
# The slope of the line plot decreases gradually, so it can be challenging to precisely
# determine the correct number of components, but you can make out that 4 and 5 are
# good candidates. For real data, the line plot might be even more challenging to read,
# and you may find several candidates that you should then examine further.
# Note that the fit score is just one metric and will not give you the entire picture,
# so you should also examine other metrics and, most importantly, look at what makes
# sense for your data when choosing a suitable model.
#
# Another important metric to consider when evaluating your models is the feasibility gap.
# If the feasibility gap is too large, then the model doesn't satisfy the constraints. Here,
# we see that the A-matrix was completely non-negative for all models, while there was a
# slight feasibility gap for the B_i-matrices. This means that the B_i-matrices only
# approximately satisfied the PARAFAC2 constraint (and this will often be the case). The
# four-component model had the lowest feasibility gap, so it was the model that best followed
# the PARAFAC2 constraint. This could be a clue that four is an appropriate number of components.
# Still, we see that the feasibility gap was on the order of :math:`10^{-5}` for all of the
# models, which means that the approximation is very good for all of them.
