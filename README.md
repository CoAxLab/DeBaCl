DeBaCl: DEnsity-BAsed CLustering with level set trees
=====================================================
[![Travis CI](https://travis-ci.org/CoAxLab/DeBaCl.svg?branch=dev)](https://travis-ci.org/CoAxLab/DeBaCl)
[![Pending Pull-Requests](http://githubbadges.herokuapp.com/CoAxLab/DeBaCl/pulls)](https://github.com/CoAxLab/DeBaCl/pulls)
[![Github Issues](http://githubbadges.herokuapp.com/CoAxLab/DeBaCl/issues)](https://github.com/CoAxLab/DeBaCl/issues)
[![License](http://img.shields.io/:license-bsd-blue.svg)](http://opensource.org/licenses/BSD-3-Clause)

DeBaCl is a Python library for density-based clustering with level set trees.

Introduction
------------
Level set trees are based on the statistically-principled definition of
clusters as modes of a probability density function. They are particularly
useful for analyzing structure in complex datasets that exhibit multi-scale
clustering behavior. DeBaCl is intended to promote the practical use of level
set trees through improvements in computational efficiency, flexible
algorithms, and an emphasis on modularity and user customizability.

License
-------
DeBaCl is available under the 3-clause BSD license.

Installation
------------
From the Python package installer:

    pip install debacl

From source:

    git clone https://github.com/CoAxLab/DeBaCl/
    export PYTHONPATH='...'

Dependencies
------------
All of the following dependencies can be installed with either conda or pip, except prettytable, which must be installed with pip.

<h3>Required</h3>
- numpy
- networkx
- prettytable

<h3>Strongly recommended</h3>
- matplotlib
- scipy

<h3>Optional</h3>
- scikit-learn

Quickstart
----------
From a Python console (e.g. IPython):

    >>> import debacl as dcl
    >>> import sklearn
    >>> X = sklearn.datasets.make_moons()
    >>> tree = dcl.construct_tree(X, k=10, prune_threshold=5)
    >>> print tree
    ...

    >>> fig = tree.plot()[0]

Documentation
-------------
The [tutorial for DeBaCl] (http://nbviewer.ipython.org/url/raw.github.com/CoAxLa
b/DeBaCl/master/docs/debacl_tutorial.ipynb) is an IPython Notebook. It is
viewable on nbviewer, or as a [PDF in the docs
folder](docs/debacl_tutorial.pdf).

The docs folder also contains a [user manual](docs/debacl_manual.pdf) with
documentation for each function and [a paper](docs/debacl_paper.pdf) describing
the statistical background of level set trees and density-based clustering.

Running unit tests
------------------