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
