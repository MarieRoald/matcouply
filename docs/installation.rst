Installing MatCoupLy
====================

You can install MatCoupLy and its dependencies by running

.. code:: bash

    pip install matcouply

If you also want to use the GPL-lisenced functionality (currently only the TV penalty), then you also need to install
``condat_tv``, which is under a GPL lisence. To do this, run

.. code:: bash

    pip install matcouply[gpl]

instead. The examples depends on some additional libraries (e.g. ``wordcloud`` and ``plotly``), and to install these
dependencies as well, you can run

.. code:: bash

    pip install matcouply[examples]

or

.. code:: bash

    pip install matcouply[gpl,examples]

to install both the GPL-lisenced functionality and the requirements for the examples. Finally, to install the
latest development branch of MatCoupLy, run

.. code:: bash

    git clone https://github.com/marieroald/matcouply.git
    cd matcouply
    pip install -e .

Alternatively, to install all requirements (including the development requirements), ``pip install -e .[gpl,devel,examples]``.


.. note::

    There is a known problem with installing both ``condat_tv`` (the GPL-lisenced library for TV regularization) and the latest version
    of <Numba `https://numba.pydata.org/`>_, which will lead to an import-time ``ValueError`` when ``matcouply`` is imported. The bug seems
    to be due to Numba pinning the NumPy version. to circumvent this, you can install ``numpy>=1.22.1`` and ``numba==0.53.1``. Alternatively,
    you can have separate virtual environments for Numba and ``condat_tv``.
