# -*- coding: utf-8 -*-
"""
PARAFAC2 for semiconductor etch analysis
========================================

Component models have been used to detect errors in semiconductor etch processes :cite:p:`wise1999comparison`, where the
datasets have three natural modes: sample, measurement and time. However, the time required for measuring
different samples may vary, which leads to a stack of matrices, one for each sample. This makes PARAFAC2 a
natural choice :cite:p:`wise2001application`, as it naturally handles time profiles of different lengths.

In this example, we repeat some of the analysis from :cite:p:`wise2001application` and show how total variation (TV) regularization
can reduce noise in the components. TV regularization is well suited for reducing noise without overly smoothing
sharp transitions :cite:p:`rudin1992nonlinear`.
"""

###############################################################################
# Setup
# ^^^^^

# sphinx_gallery_thumbnail_number = -1

import matplotlib.pyplot as plt
import numpy as np
from tensorly.decomposition import parafac2

from matcouply.data import get_semiconductor_etch_machine_data
from matcouply.decomposition import parafac2_aoadmm

###############################################################################
# Data loading and preprocessing
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


train_data, train_metadata, test_data, test_metadata = get_semiconductor_etch_machine_data()


###############################################################################
# The dataset contains three experiments (experiments 29, 31 and 33). In :cite:p:`wise2001application`, the authors
# highlight the components obtained for experiment 29, so let's look at the same experiment.


dataset_29 = [key for key in train_data if key.isnumeric() and key.startswith("29")]
matrices = [train_data[key].values for key in dataset_29]

###############################################################################
# Before analysing, we apply the same preprocessing steps as in :cite:p:`wise2001application` — centering and scaling each
# matrix based on the global mean and standard deviation.


stacked = np.concatenate(matrices, axis=0)
mean = stacked.mean(0, keepdims=True)
std = stacked.std(0, keepdims=True)
standardised = [(m - mean) / std for m in matrices]

###############################################################################
# Fit a PARAFAC2 model
# ^^^^^^^^^^^^^^^^^^^^
#
# Let's begin by fitting an unregularized PARAFAC2 model using the alternating least squares algorithm
# :cite:p:`kiers1999parafac2` with the implementation in `TensorLy <http://tensorly.org/>`_ :cite:p:`kossaifi2019tensorly`.
# This algorithm is comparable with the one used in :cite:p:`wise2001application`.
#
# We also impose non-negativity on the :math:`\mathbf{A}`-matrix to handle the special sign indeterminacy of
# PARAFAC2 :cite:p:`harshman1972parafac2`. The :math:`\mathbf{A}`-matrix elements in :cite:p:`wise2001application` are
# also non-negative, so this shouldn't change the components.
#
# Similarly as :cite:`wise2001application`, we extract two components.


pf2, rec_err = parafac2(
    standardised, 2, n_iter_max=10_000, return_errors=True, nn_modes=[0], random_state=0, tol=1e-9, verbose=True
)


###############################################################################
# We examine the results by plotting the relative SSE and its relative change as a function of iteration number

it_num = np.arange(len(rec_err)) + 1
rel_sse = np.array(rec_err) ** 2

fig, axes = plt.subplots(1, 2, figsize=(10, 3), tight_layout=True)
axes[0].plot(it_num, rel_sse)
axes[0].set_ylim(0.67, 0.68)
axes[0].set_xlabel("Iteration number")
axes[0].set_ylabel("Relative SSE")

axes[1].semilogy(it_num[1:], (rel_sse[:-1] - rel_sse[1:]) / rel_sse[:-1])
axes[1].set_xlabel("Iteration number")
axes[1].set_ylabel("Relative change in SSE")
axes[1].set_ylim(1e-9, 1e-6)

plt.show()

###############################################################################
# Next, we look at the components

weights, (A, B, C), P_is = pf2
B_is = [P_i @ B for P_i in P_is]

# We normalise the components to make them easier to compare
A_norm = np.linalg.norm(A, axis=0, keepdims=True)
C_norm = np.linalg.norm(C, axis=0, keepdims=True)
A = A / A_norm
B_is = [B_i * A_norm * C_norm for B_i in B_is]
C = C / C_norm

# We find the permutation so the first component explains most of the variation in the data
B_norm = np.linalg.norm(B, axis=0, keepdims=True)
permutation = np.argsort(weights * A_norm * B_norm * C_norm).squeeze()

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

for i, B_i in enumerate(B_is):
    axes[0].plot(B_i[:, permutation[0]], color="k", alpha=0.3)
    axes[1].plot(B_i[:, permutation[1]], color="k", alpha=0.3)

###############################################################################
# We see that the components are similar to those in :cite:p:`wise2001application`. We can see an overall shape, but
# they are fairly noisy.
#
# .. note::
#
#     In this simple example, we only use one random initialisation. For a more thorough analysis, you should fit
#     several models with different random initialisations and select the model with the lowest SSE
#     :cite:p:`yu2021parafac2`.
#

###############################################################################
# Next we use PARAFAC2 ADMM to apply a TV penalty
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Since the TV-penalty scales with the norm of the factors, we also need to penalise the norm of :math:`\mathbf{A}`
# and :math:`\mathbf{C}` :cite:p:`roald2021admm`. In this case, we use unit ball constraints, constraining the columns of
# :math:`\mathbf{A}` and :math:`\mathbf{C}` to have unit norm.
#
# Similar as before, we add non-negativity on :math:`\mathbf{A}` to resolve the sign indeterminacy.
#
# .. note::
#
#     The proximal operator for the total variation penalty is computed using the C-implementation for the improved
#     version of the direct TV algorithm presented in :cite:p:`condat2013direct`. The C-implementation is CeCILL
#     lisenced and is available `here <https://lcondat.github.io/software.html>`__, and the Python-wrapper,
#     `condat-tv`, is GPL-3 lisenced and is available `here <https://github.com/MarieRoald/condat_tv>`__.
#

cmf, diagnostics = parafac2_aoadmm(
    standardised,
    2,
    n_iter_max=10_000,
    non_negative={0: True},
    l2_norm_bound=[1, None, 1],
    tv_penalty={1: 0.1},
    verbose=100,
    return_errors=True,
    init_params={"nn_modes": [0]},
    constant_feasibility_penalty=True,
    tol=1e-9,
    random_state=0,
)

###############################################################################
# We examine the diagnostic plots
#
# For ALS, the relative SSE and its change was the only interesting metrics. However, with regularized PARAFAC2 and AO-ADMM
# we should also to look at the feasibility gaps and the regularization penalty.
#
# All feasibility gaps and the change in relative SSE should be low.

rel_sse = np.array(diagnostics.rec_errors) ** 2
loss = np.array(diagnostics.regularized_loss)
feasibility_penalty_A = np.array([gapA for gapA, gapB, gapC in diagnostics.feasibility_gaps])
feasibility_penalty_B = np.array([gapB for gapA, gapB, gapC in diagnostics.feasibility_gaps])
feasibility_penalty_C = np.array([gapC for gapA, gapB, gapC in diagnostics.feasibility_gaps])

it_num = np.arange(len(rel_sse))

fig, axes = plt.subplots(2, 3, figsize=(15, 6), tight_layout=True)
axes[0, 0].plot(it_num, rel_sse)
axes[0, 0].set_ylim(0.69, 0.71)
axes[0, 0].set_xlabel("Iteration number")
axes[0, 0].set_ylabel("Relative SSE")

axes[0, 1].plot(it_num, loss)
axes[0, 1].set_xlabel("Iteration number")
axes[0, 1].set_ylabel("Regularized loss")

axes[0, 2].semilogy(it_num[1:], np.abs(loss[:-1] - loss[1:]) / loss[:-1])
axes[0, 2].set_xlabel("Iteration number")
axes[0, 2].set_ylabel("Relative change in regularized loss")

axes[1, 0].semilogy(it_num, feasibility_penalty_A)
axes[1, 0].set_xlabel("Iteration number")
axes[1, 0].set_ylabel("Feasibility gap A")

axes[1, 1].semilogy(it_num, feasibility_penalty_B)
axes[1, 1].set_xlabel("Iteration number")
axes[1, 1].set_ylabel("Feasibility gap B_is")
axes[1, 1].legend(["PARAFAC2", "TV"])

axes[1, 2].semilogy(it_num, feasibility_penalty_C)
axes[1, 2].set_xlabel("Iteration number")
axes[1, 2].set_ylabel("Feasibility gap C")

###############################################################################
# Next, we look at the regularized components

weights, (A, B_is, C) = cmf
# We find the permutation so the first component explains most of the variation in the data
B_norm = np.linalg.norm(B_is[0], axis=0, keepdims=True)  # All B_is have same norm due to PARAFAC2 constraint
permutation = np.argsort(B_norm).squeeze()

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

for i, B_i in enumerate(B_is):
    axes[0].plot(B_i[:, permutation[0]], color="k", alpha=0.3)
    axes[1].plot(B_i[:, permutation[1]], color="k", alpha=0.3)

###############################################################################
# We see that the TV regularization removed much of the noise. We now have piecewise constant components
# with transitions that are easy to identify.

###############################################################################
# Comparing with unregularized PARAFAC2
print("Relative SSE with unregularized PARAFAC2: ", rec_err[-1] ** 2)
print("Relative SSE with TV regularized PARAFAC2:", diagnostics.rec_errors[-1] ** 2)

###############################################################################
# We see that there is only a small change in the relative SSE, but the components are much smoother and
# the transitions are clearer.

###############################################################################
# License
# ^^^^^^^
#
# Since this example uses the `condat_tv`-library, it is lisenced under a GPL-3 license
#
# .. code:: text
#
#                       Version 3, 29 June 2007
#
#     Example demonstrating TV regularized PARAFAC2
#     Copyright (C) 2021 Marie Roald
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <http://www.gnu.org/licenses/>.
