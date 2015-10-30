DeBaCl: DEnsity-BAsed CLustering with level set trees
=====================================================
[![Travis CI](https://travis-ci.org/CoAxLab/DeBaCl.svg?branch=dev)](https://travis-ci.org/CoAxLab/DeBaCl)
[![Pending Pull-Requests](http://githubbadges.herokuapp.com/CoAxLab/DeBaCl/pulls)](https://github.com/CoAxLab/DeBaCl/pulls)
[![Github Issues](http://githubbadges.herokuapp.com/CoAxLab/DeBaCl/issues)](https://github.com/CoAxLab/DeBaCl/issues)
[![License](http://img.shields.io/:license-bsd-blue.svg)](http://opensource.org/licenses/BSD-3-Clause)

DeBaCl is a Python library for density-based clustering with level set trees.

Introduction
------------
Level set trees are a statistically-principled way to represent the topology of
a probability density function. This representation is particularly useful for
several core tasks in statistics:
    - **clustering**, especially for data with multi-scale clustering behavior
    - **exploratory data analysis** and **data visualization**
    - **anomaly detection**

DeBaCl is an Python implementation of the Level Set Tree method, with an
emphasis on computational speed, algorithmic simplicity, and extensibility.

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

    ```python
    >>> import debacl as dcl
    >>> import sklearn
    >>> X = sklearn.datasets.make_moons()
    >>> tree = dcl.construct_tree(X, k=10, prune_threshold=5)
    >>> print tree
    ```
    ```no-highlight
    ...
    ```

    >>> fig = tree.plot()[0]

Running unit tests
------------------

Documentation
-------------
The [tutorial for DeBaCl] (http://nbviewer.ipython.org/url/raw.github.com/CoAxLa
b/DeBaCl/master/docs/debacl_tutorial.ipynb) is an IPython Notebook. It is
viewable on nbviewer, or as a [PDF in the docs
folder](docs/debacl_tutorial.pdf).

The docs folder also contains a [user manual](docs/debacl_manual.pdf) with
documentation for each function and [a paper](docs/debacl_paper.pdf) describing
the statistical background of level set trees and density-based clustering.

References
----------
- Chaudhuri, K., & Dasgupta, S. (2010). [Rates of Convergence for the Cluster T
  ree](http://www.cse.ucsd.edu/sites/cse/files/cse/assets/research/theory/Chaud
  huriDasgupta_2010.pdf). In Advances in Neural Information Processing Systems
  23 (pp. 343–351). Vancouver, BC.

- Kent, B. P., Rinaldo, A., Yeh, F.-C., & Verstynen, T. (2014). [Mapping
  Topographic Structure in White Matter Pathways with Level Set Trees](http://j
  ournals.plos.org/plosone/article?id=10.1371/journal.pone.0093344#pone-0093344
  -g009). PLoS ONE.

- Kent, B. P., Rinaldo, A., & Verstynen, T. (2013). [DeBaCl: A Python Package
  for Interactive DEnsity-BAsed CLustering](http://arxiv.org/abs/1307.8136).
  arXiv preprint:1307.8136.

- Kent, B.P. (2013). [Level Set Trees for Applied Statistics](http://www.scribd
  .com/doc/242026196/Level-Set-Trees-for-Applied-Statistics).
