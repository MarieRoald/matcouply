Installing MatCoupLy
====================

You can install MatCoupLy and its dependencies by running

.. code:: bash

    pip install matcouply


Optional functionality
----------------------
For loading data, we also need to install the Pandas, tqdm and Requests libraries.
To install MatCoupLy with these additional dependencies, run

.. code:: bash

    pip install matcouply[data]

The ``testing`` module, which contains functionality for automatic unit test generation requires pytest, which you can get by running

.. code:: bash

    pip install matcouply[testing]

The unimodality constraint can use Numba to increase its efficiency with just in time compilation.
However, this requires that compatible versions of NumPy and Numba are installed. To ensure this,
you can install matcouply with

.. code:: bash

    pip install matcouply[numba]

which will install ``numpy >= 1.22.1`` and ``numba == 0.53.1`` (which are compatible).

If you also want to use the GPL-lisenced functionality (currently only the TV penalty), then you also need to install
``condat_tv``, which is under a GPL lisence. To do this, run

.. code:: bash

    pip install matcouply[gpl]

The examples depends on some additional libraries (e.g. ``wordcloud`` and ``plotly``), and to install these
dependencies as well, you can run

.. code:: bash

    pip install matcouply[examples]


To install multiple optional dependencies, list them all in the brackets, separated with a comma. For example

.. code:: bash

    pip install matcouply[gpl,examples]

will install both the GPL-lisenced functionality and the requirements for the examples.

Finally, to install the latest development branch of MatCoupLy, run

.. code:: bash

    git clone https://github.com/marieroald/matcouply.git
    cd matcouply
    pip install -e .

Alternatively, to install all requirements (including the development requirements), ``pip install -e .[gpl,devel,examples,data]``.
