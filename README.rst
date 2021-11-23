=========
MatCoupLy
=========
*Learning coupled matrix factorizations with Python*

.. image:: https://readthedocs.org/projects/cm-aoadmm/badge/?version=latest
        :target: https://cm-aoadmm.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black

.. image:: https://github.com/MarieRoald/cm_aoadmm/actions/workflows/Tests.yml/badge.svg
    :target: https://github.com/MarieRoald/cm_aoadmm/actions/workflows/Tests.yml
    :alt: Tests

.. image:: https://codecov.io/gh/MarieRoald/cm_aoadmm/branch/main/graph/badge.svg?token=GDCXEF2MGE
    :target: https://codecov.io/gh/MarieRoald/cm_aoadmm
    :alt: Coverage


MatCoupLy is a Python library for learning coupled matrix factorizations with flexible constraints and regularization with Python.


Installation
------------

To install MatCoupLy and all MIT-compatible dependencies from PyPI, you can run

.. code::

        pip install matcouply
        
If you also want to enable total variation regularization, you need to install all components, which comes with a GPL-v3 lisence

.. code::

        pip install matcouply[all]

About
-----

MatCoupLy uses AO-ADMM to fit constrained and regularised coupled matrix factorization (and PARAFAC2) models.
It uses the alternating updates with the alternating direction method of multipliers (AO-ADMM) algorithm,
which is very flexible in terms of constraints [1, 2]


Example
-------

Below is a simulated example, where a set of three random non-negative coupled matrices are generated and
decomposed using a non-negative PARAFAC2 factorization with both L1 and L2 penalty functions.

.. code:: python

        from matcouply import cmf_aoadmm
        from matcouply.random import random_coupled_matrices
        
        simulated_components = random_coupled_matrices(rank=3, shapes=((40, 25), (35, 25), (19, 25)))
        simulated_data = simulated_components.to_matrices()
        
        estimated_cmf, aux_vars, dual_vars = cmf_aoadmm(
            simulated_data,
            rank=3,
            non_negative=True,
            l1_penalty=[0.1, 0.2, 0.1],
            l2_penalty={2: 1},
            parafac2=True
        )
        
        est_weights, (est_A, est_B_is, est_C) = estimated_cmf


References
----------

 * [1]: Roald M, Schenker C, Cohen JE, Acar E. PARAFAC2 AO-ADMM: Constraints in all modes. EUSIPCO (2021).
 * [2]: Roald M, Schenker C, Bro R, Cohen JE, Acar E. An AO-ADMM approach to constraining PARAFAC2 on all modes (2021). arXiv preprint arXiv:2110.01278.