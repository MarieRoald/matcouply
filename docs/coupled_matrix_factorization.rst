What are coupled matrix factorizations?
======================================= 

MatCoupLy computes coupled matrix factorizations, which are useful for finding patterns in
collections of matrices and third order tensors. A coupled matrix factorization is used to jointly
factorize a collection of matrices, :math:`\{\mathbf{X}^{(i)}\}_{i=1}^I`, with the same number of columns
but possibly different number of rows, on the form

.. math::

    \mathbf{X}^{(i)} \approx \mathbf{B}^{(i)} \mathbf{D}^{(i)} \mathbf{C}^\mathsf{T},

where :math:`\mathbf{C}` is a factor matrix shared for all :math:`\mathbf{X}^{(i)}`-s, and
:math:`\{\mathbf{B}^{(i)}\}_{i=1}^I` is a collection factor matrices, one for each :math:`\mathbf{X}^{(i)}`.
The diagonal :math:`\{\mathbf{D}^{(i)}\}_{i=1}^I`-matrices describes the signal strength of each
component for each :math:`\mathbf{X}^{(i)}`, and their diagonal entries are often collected into
a single factor matrix, :math:`\mathbf{A}`. Below is an illustration of this model:

.. figure:: figures/CMF_multiblock.svg
   :alt: Illustration of a coupled matrix factorization
   :width: 90 %

   Illustration of a coupled matrix factorization where colours represent different components.

Factor weights and the scale indeterminacy of CMF models
--------------------------------------------------------
There are two important scaling indeterminacies with coupled matrix factorization. First, any column in one of the factor
matrices (e.g. :math:`\mathbf{A}`) may be scaled arbitrarilly by a constant factor :math:`s` as long as the corresponding column
of :math:`\mathbf{C}` or *all* :math:`\mathbf{B}^{(i)}` matrices are scaled by :math:`1/s`, without affecting the data represented
by the model. It is therefore, sometimes, customary to normalize the factor matrices and store their norms in a separate
*weight*-vector, :math:`\mathbf{w}`. Then, the data matrices are represented by

.. math::
    \mathbf{X}^{(i)} \approx \mathbf{B}^{(i)} \mathbf{D}^{(i)} \text{diag}(\mathbf{w}) \mathbf{C}^\mathsf{T},

where :math:`\text{diag}(\mathbf{w})` is the diagonal matrix with diagonal entries given by the weights. :func:`matcouply.decomposition.cmf_aoadmm`
and :func:`matcouply.decomposition.parafac2` will not scale the factor matrices this way, since that may affect the penalty
from norm-dependent regularization (e.g. the :class:`matcouply.penalties.GeneralizedL2Penalty`). However, :class:`matcouply.coupled_matrices.CoupledMatrixFactorization`
supports the use of a ``weights`` vector to be consistent with TensorLy.

There are, however, another scale indeterminacy which can affect the extracted components in a more severe way. Any column
in any :math:`\mathbf{B}^{(i)}`-matrix may also be scaled arbitrarilly by a constant :math:`s` if the corresponding entry
in :math:`\mathbf{D}^{(i)}` is scaled by :math:`1/s`. To resolve this indeterminacy, we need to either impose constraints or
fix the values in the :math:`\mathbf{D}^{(i)}`-matrices (e.g. keeping them equal to 1) by passing ``update_A=False`` to
:func:`matcouply.decomposition.cmf_aoadmm`. PARAFAC2, for example, takes care of the scaling indeterminacy, however, the sign of
any column, :math:`\mathbf{b}^{(i)}_r` of :math:`\mathbf{B}^{(i)}` and any entry of :math:`\mathbf{D}^{(i)}` can be flipped if the same flip is imposed on all
columns in :math:`\mathbf{B}^{(i)}` (and entries of :math:`\mathbf{D}^{(i)}`) that are not orthogonal to :math:`\mathbf{b}^{(i)}_r`.
To avoid this sign indeterminacy, you can apply additional constraints to the PARAFAC2 model, e.g. enforcing non-negative :math:`\mathbf{D}^{(i)}` matrices. 
:cite:p:`harshman1972parafac2`

Constraints and uniqueness
--------------------------

Coupled matrix factorization models without any additional constraints are not unique. This means
that their components cannot be directly interpreted. To see this, consider the stacked matrix

.. math::

    \mathbf{X} = \begin{bmatrix}
        \mathbf{X}^{(0)} \\
        \mathbf{X}^{(1)} \\
        \vdots \\
        \mathbf{X}^{(I)} \\
    \end{bmatrix}

A coupled matrix factorization of :math:`\{\mathbf{X}^{(i)}\}_{i=1}^I` can be interpreted as a 
matrix factorization of :math:`\mathbf{X}`, which is known to have several solutions. Therefore,
we need to impose additional constraints to obtain interpretable components.

PARAFAC2
^^^^^^^^

One popular constraint used to obtain uniqueness is the *constant cross product constraint* of the
PARAFAC2 model :cite:p:`harshman1972parafac2,kiers1999parafac2,harshman1996uniqueness` (therefore also called the *PARAFAC2 constraint*). 

.. math::

    {\mathbf{B}^{(i_1)}}^\mathsf{T}{\mathbf{B}^{(i_1)}} = {\mathbf{B}^{(i_2)}}^\mathsf{T}{\mathbf{B}^{(i_2)}}.

Coupled matrix factorization models with this constraint are named PARAFAC2 models, and they are
commonly used in data mining :cite:p:`chew2007cross,gujral2020spade`, chemometrics :cite:p:`amigo2008solving`,
and analysis of electronic health records :cite:p:`afshar2018copa`. 

Non-negativity
^^^^^^^^^^^^^^

Another popular constraint is non-negativity constraints, which are commonly imposed on all parameters of
the model. Non-negativity constraints are commonly used for non-negative data, where we want non-negative
components. While this constraint doesn't necessarily result in a unique model, it does improve the uniqueness
properties of coupled matrix factorization models. Lately, it has also been a focus on adding non-negativity 
constraints to PARAFAC2, which often provides a unique model :cite:p:`cohen2018nonnegative,van2020getting,roald2021admm`.
The added non-negativity constraints improves PARAFAC2 model's numerical properties and it can also make
the components more interpretable :cite:p:`roald2021admm`.

Other constraints and regularization penalties
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

MatCoupLy supports a wide array of possible constraints and regularization penalties. For a full list
of the implemented constraints and penalties, see :doc:`autodoc/penalties`.

.. note::

    If you use penalty based regularization that scales with the norm of one of the parameters, then
    norm-based regularization should be imposed on all modes. This can, for example, be L2 regularization,
    max- and min-bound constraints, L1 penalties or hard L2 norm constraints. See :cite:p:`roald2021admm`
    for more details.
