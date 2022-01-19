.. _optimization:

Fitting coupled matrix factorization models
===========================================

Objective function
^^^^^^^^^^^^^^^^^^

To fit a coupled matrix factorization, we solve the following optimization problem

.. math::
    \min_{\mathbf{A}, \{\mathbf{B}^{(i)}\}_{i=1}^I, \mathbf{C}}
    \frac{1}{2} \sum_{i=1}^I \frac{\| \mathbf{B}^{(i)} \mathbf{D}^{(i)} \mathbf{C}^\mathsf{T} - \mathbf{X}^{(i)}\|^2}{\|\mathbf{X}^{(i)}\|^2},

where :math:`\mathbf{A}` is the matrix obtained by stacking the diagonal entries of all
:math:`\mathbf{D}^{(i)}`-matrices. However, as discussed in :doc:`coupled_matrix_factorization`, this problem does not
have a unique solution, and each time we fit a coupled matrix factorization, we may obtain
different factor matrices. As a consequence, we cannot interpret the factor matrices.
To circumvent this problem, it is common to add regularization, forming the following
optimisation problem

.. math::
    \min_{\mathbf{A}, \{\mathbf{B}^{(i)}\}_{i=1}^I, \mathbf{C}}
    \frac{1}{2} \sum_{i=1}^I \frac{\| \mathbf{B}^{(i)} \mathbf{D}^{(i)} \mathbf{C}^\mathsf{T} - \mathbf{X}^{(i)}\|^2}{\|\mathbf{X}^{(i)}\|^2}
    + \sum_{n=1}^{N_\mathbf{A}} g^{(A)}_n(\mathbf{A})
    + \sum_{n=1}^{N_\mathbf{B}} g^{(B)}_n(\{ \mathbf{B}^{(i)} \}_{i=1}^I)
    + \sum_{n=1}^{N_\mathbf{C}} g^{(C)}_n(\mathbf{C}),

where the :math:`g`-functions are regularization penalties, and :math:`N_\mathbf{A}, N_\mathbf{B}`
and :math:`N_\mathbf{C}` are the number of regularization penalties for 
:math:`\mathbf{A}, \{\mathbf{B}^{(i)}\}_{i=1}^I` and :math:`\mathbf{C}`, respectively.

The formulation above also encompasses hard constraints, such as :math:`a_{ir} \geq 0` for
any index :math:`(i, r)`. To obtain such a constraint, we set 

.. math::
    g^{(\mathbf{A})} = \begin{cases}
        0 & \text{if } a_{ir} \geq 0 \text{ for all } a_{ir} \\
        \infty & \text{otherwise}.
    \end{cases}

.. note::

    The data fidelity term (sum of squared error) differs by a factor :math:`1/2`
    compared to that in :cite:p:`roald2021parafac2,roald2021admm`

Optimization
^^^^^^^^^^^^

To solve the regularized least squares problem, we use alternating optimisation (AO) with
the alternating direction method of multipliers (ADMM). AO-ADMM is a block 
coordinate descent scheme, where the factor matrices for each mode is updated in an
alternating fashion. This means that the regularized loss function above is split into
the following three optimization subproblems:

.. math::
    \min_{\mathbf{A}}
    \frac{1}{2} \sum_{i=1}^I \| \mathbf{B}^{(i)} \mathbf{D}^{(i)} \mathbf{C}^\mathsf{T} - \mathbf{X}^{(i)}\|^2
    + \sum_{n=1}^{N_\mathbf{A}} g^{(A)}_n(\mathbf{A}),

.. math::
    \min_{\{\mathbf{B}^{(i)}\}_{i=1}^I}
    \frac{1}{2} \sum_{i=1}^I \| \mathbf{B}^{(i)} \mathbf{D}^{(i)} \mathbf{C}^\mathsf{T} - \mathbf{X}^{(i)}\|^2
    + \sum_{n=1}^{N_\mathbf{B}} g^{(B)}_n(\{ \mathbf{B}^{(i)} \}_{i=1}^I),

.. math::
    \min_{\mathbf{C}}
    \frac{1}{2} \sum_{i=1}^I \| \mathbf{B}^{(i)} \mathbf{D}^{(i)} \mathbf{C}^\mathsf{T} - \mathbf{X}^{(i)}\|^2
    + \sum_{n=1}^{N_\mathbf{C}} g^{(C)}_n(\mathbf{C}),

which we solve approximately, one at a time, using a few iterations of ADMM. We repeat this
process, updating :math:`\mathbf{A}, \{\mathbf{B}^{(i)}\}_{i=1}^I` and :math:`\mathbf{C}` untill
some convergence criteria are satisfied.

The benefit of AO-ADMM is its flexibility in terms of regularization and constraints. We
can impose any regularization penalty or hard constraint so long as we have a way to
evaluate the scaled proximal operator of the penalty function or projection onto the
feasible set of the hard constraint :cite:p:`huang2016flexible`. That is, any regularization
function, :math:`g(\mathbf{v})`, where we can solve the problem

.. math::
    \min_{\mathbf{w}} g(\mathbf{w}) + \frac{\rho}{2}\|\mathbf{w} - \mathbf{v}\|^2,

where :math:`\mathbf{v}` and :math:`\mathbf{w}` are the vectorized input to the regularized
least squares subproblem we solve with ADMM. :math:`\rho` is a parameter that penalises infeasible
solutions (more about that later), we use the name *feasibility penalty parameter* for :math:`\rho`.

The AO-ADMM algorithm is described in detail (in the context of PARAFAC2, a special case of
coupled matrix factorization) in :cite:p:`roald2021admm`.

.. note::
    The role of :math:`\mathbf{A}` and :math:`\mathbf{C}` are switched between this software and
    :cite:p:`roald2021admm`, as this change makes for more straightforward usage.

ADMM and the feasibility gap
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

There is one property of ADMM that is important to be aware of when fitting models with AO-ADMM:
The *feasibility gap*. If the feasibility gap is high (what a high value depends on the application,
but any value above 0.0001 is suspicious and any value above 0.01 is likely high), then the
constraints and regularization we impose may not be satisfied. To explain why, we briefly describe
ADMM (for a thorough introduction, see :cite:p:`boyd2011distributed`, and for an introduction
specifically for coupled matrix factorization, see :cite:p:`roald2021admm`).

ADMM can solve problems on the form

.. math::
    \min_{\mathbf{x}, \mathbf{z}} f(\mathbf{x}) + g(\mathbf{z}) \\
    \text{s.t. } \mathbf{Mx} - \mathbf{Ny} = \mathbf{c},

which is useful for solving regularized least squares problems. If we have only one regularization
penalty, then we can rewrite a regularized least squares problem to the standard form of ADMM:

.. math::
    \min_{\mathbf{x}, \mathbf{z}} \frac{1}{2}\|\mathbf{Tx} - \mathbf{b}\|^2 + g(\mathbf{z}) \\
    \text{s.t. } \mathbf{x} = \mathbf{z}.

ADMM then works by forming the augmented Lagrange dual problem:

.. math::
    \max_{\boldsymbol{\lambda}} \min_{\mathbf{x}, \mathbf{z}} \frac{1}{2}\|\mathbf{Tx} - \mathbf{b}\|^2
              + g(\mathbf{z}) 
              + \frac{\rho}{2}(\mathbf{x} - \mathbf{z})^2
              + \boldsymbol{\lambda}^\mathsf{T} (\mathbf{x} - \mathbf{z}),

where :math:`\rho` is a penalty parameter for infeasible solutions (i.e. solutions where
:math:`\mathbf{x} \neq \mathbf{z}`).

An important property for assessing validity of models fitted with AO-ADMM is therefore the
feasibility gap, given by

.. math::
    \frac{\|\mathbf{x} - \mathbf{z}\|}{\|\mathbf{x}\|}

If this is high, then the solution is infeasible, and the model is likely not valid.

.. note::

    A sufficiently small feasibility gap is part of the stopping criteria, so if the AO-ADMM
    procedure stopped before the maximum number of iterations were reached, then the feasibility
    gaps are sufficiently small.

Penalty-functions
^^^^^^^^^^^^^^^^^

We separate the penalty functions into three categories: row-wise penalties, matrix-wise penalties
and multi-matrix penalties:

* *Multi-matrix* penalties are penalties that penalise behaviour across 
  multiple :math:`\mathbf{B}^{(i)}`-matrices at once (e.g. the PARAFAC2 constraint: :meth:`matcouply.penalties.Parafac2`).
* *Matrix-wise* penalties are penalties full matrices (or columns of full matrices) at once
  (e.g. total variation regularization: :meth:`matcouply.penalties.TotalVariationPenalty`) and can be
  applied either to the :math:`\mathbf{B}^{(i)}`-matrices, or the :math:`\mathbf{C}`-matrix with no.
* Finally, *row-wise* penalties are penalties that single rows (or elements) of a matrix at a time
  (e.g. non-negativity: :meth:`matcouply.penalties.NonNegativity`. These penalties can be applied to
  any factor matrix.

.. note::

    We can also apply matrix-wise penalties on :math:`\mathbf{A}` and special multi-matrix
    penalties that require a constant feasibility penalty for all :math:`\mathbf{B}^{(i)}`-matrices
    by using the `constant_feasibility_penalty=True` argument. There are currently no
    multi-matrix penalties that require a constant feasibility penalty in MatCoupLy. An example
    of such a penalty could be a similarity-based penalty across the different
    :math:`\mathbf{B}^{(i)}`-matrices.

Stopping conditions
^^^^^^^^^^^^^^^^^^^

The AO-ADMM procedure has two kinds of stopping conditions. The ADMM stopping conditions (inner loop), used to
determine if the regularized least squares subproblems have converged and the AO-ADMM stopping conditions
(outer loop) used to determine whether the the full fitting procedure should end.

**Inner loop (ADMM):**

The ADMM stopping conditions is by default disabled, and all inner iterations are ran without checking for
convergence. The reason is that for a large portion of the iterations, the ADMM iterations will not converge,
and checking the stopping conditions may be a bottleneck. If they are set, then the following conditions must
be satisfied

.. math::
    \frac{\|\mathbf{x}^{(t, q)} - \mathbf{z}^{(t, q)}\|}{\|\mathbf{x}^{(t, q)}\|} < \epsilon_{\text{inner}},

.. math::
    \frac{\|\mathbf{x}^{(t, q)} - \mathbf{z}^{(t, q-1)}\|}{\|\mathbf{z}^{(t, q)}\|} < \epsilon_{\text{inner}},

where :math:`\mathbf{x}^{(t, q)}` is the variable whose linear system we solve (i.e. :math:`\mathbf{A}, \{\mathbf{B}^{(i)}\}_{i=1}^I`
or :math:`\mathbf{C}`) and :math:`t` and :math:`q` represent the outer and inner iteration number, respectively.

**Outer loop (AO-ADMM):**

For the outer, AO-ADMM, loop, the stopping conditions are enabled by default and consist of two parts that must
be satisfied. The loss condition and the feasibility conditions.

The loss condition states that either an absolute loss value condition or a relative loss decrease
condition should be satisfied. These conditions are given by:

.. math::
    f(\mathbf{M}^{(t)}) + g(\mathbf{M}^{(t)}) < \epsilon_{\text{abs}},

and

.. math::
    \frac{|f(\mathbf{M}^{(t-1)}) - f(\mathbf{M}^{(t)}) + g(\mathbf{M}^{(t-1)}) - g(\mathbf{M}^{(1)})|}
         {f(\mathbf{M}^{(t-1)}) + g(\mathbf{M}^{(t-1)})}
    < \epsilon_{\text{rel}},

where :math:`f` is the relative sum of squared error and :math:`g` is the sum of all regularization functions.
:math:`\mathbf{M}^{(t)}` represents the full decomposition after :math:`t` outer iterations.

The feasibility conditions must also be satisfied for stopping the AO-ADMM algorithm, and they are on the form

.. math::
    \frac{\|\mathbf{x}^{(t)} - \mathbf{z}^{(t)}\|}{\|\mathbf{x}^{(t)}\|} \leq \epsilon_{\text{feasibility}},

where :math:`\mathbf{x}^{(t)}` represents :math:`\mathbf{A}, \{\mathbf{B}^{(i)}\}_{i=1}^I` or :math:`\mathbf{C}` after :math:`t`
outer iterations and :math:`\mathbf{z}^{(t)}` represents a corresponding auxiliary variable after after :math:`t`
outer iterations. The feasibility conditions must be satisfied for all auxiliary variables for all modes for stopping
the outer loop.
