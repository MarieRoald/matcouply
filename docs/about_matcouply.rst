About MatCoupLy
===============

:doc:`Coupled matrix factorization<coupled_matrix_factorization>` methods are useful for uncovering shared and varying patterns from related measurements. 
For these patterns to be unique and interpretable, it is often necessary to impose additional constraints. 
A prominent example of such a constrained coupled matrix factorization is PARAFAC2 :cite:p:`harshman1972parafac2,kiers1999parafac2`, which uses a constant cross-product 
constraint to achieve uniqueness under mild conditions :cite:p:`harshman1996uniqueness`. 
Lately, interest in such methods have increased :cite:p:`ruckebusch2013multivariate,madsen2017quantifying,ren2020robust,roald2020tracing`, 
but there is a lack of software support, especially free open-source software. 
The availability of free accessible software 
(like `scikit-learn <scikit-learn.org>`_ and `PyTorch <pytorch.org>`_) has been important for the rapid progress in machine learning research. 
Recently, `TensorLy <tensorly.org>`_ :cite:p:`kossaifi2019tensorly` has provided open source software support to tensor decomposition models as well.
However, there is not yet such a software for constrained coupled matrix factorization models. 
Therefore, given the growing interest in these methods for data mining, there is a pressing need for well-developed, 
easy to use and documented open-source implementations. 

MatCoupLy aims to meet that need by building on top of the popular TensorLy framework and implementing 
coupled matrix factorization with alternating optimization with the alternating direction method of multipliers 
(AO-ADMM) which supports flexible constraints :cite:p:`huang2016flexible,roald2021admm`. MatCoupLy implements a 
:doc:`selection of useful constraints<autodoc/penalties>` and provides an easy-to-use foundation to make it straightforward and 
painless for researchers to implement, 
test and use custom constraints. 
The MIT licence makes MatCoupLy suitable for academic and commercial purposes. 

**Why Python?**

Python is a free open source programming language that’s easy to use for beginners and professionals alike. Lately, Python has emerged as a natural choice for machine learning and data analysis, 
and is used both for research and industrial applications. 
The increasing popularity of TensorLy for tensor learning in Python further establishes 
Python as a natural language for a coupled matrix factorization library.