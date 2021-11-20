=========
MatCoupLy
=========
*Learning coupled matrix factorisations with Python*


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


MatCoupLy is a Python library for learning coupled matrix factorisations with flexible constraints and regularisation with Python.

Installation
------------

To install MatCoupLy and all MIT-compatible dependencies from PyPI, you can run

.. code::

        pip install matcouply
        
If you also want to enable total variation regularisation, you need to install all components, which comes with a GPL-v3 lisence

.. code::

        pip install matcouply[all]


Example
-------

Below is a simulated example, where a set of three random non-negative coupled matrices are generated and decomposed using a non-negative PARAFAC2 factorisation with both L1 and L2 penalty functions.

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
