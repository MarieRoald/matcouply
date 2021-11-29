Installing MatCoupLy
====================

You can install MatCoupLy and its dependencies by running

.. code:: bash

    pip install matcouply

If you also want to use the GPL-lisenced functionality (currently only the TV penalty), then you also need to install
``condat_tv``, which is under a GPL lisence. To do this, run

.. code:: bash

    pip install matcouply[all]

instead. The examples depends on some additional libraries (e.g. ``wordcloud`` and ``plotly``), and to install these
dependencies as well, you can run

.. code:: bash

    pip install matcouply[examples]

or

.. code:: bash

    pip install matcouply[all,examples]

to install both the GPL-lisenced functionality and the requirements for the examples. Finally, to install the
latest development branch of MatCoupLy, run

.. code:: bash

    git clone https://github.com/marieroald/matcouply.git
    cd matcouply
    pip install -e .

Alternatively, to install all requirements (including the development requirements), ``pip install -e .[all,devel,examples]``.