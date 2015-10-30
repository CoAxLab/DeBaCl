DeBaCl: DEnsity-BAsed CLustering with level set trees
=====================================================
[![Travis CI](https://travis-ci.org/CoAxLab/DeBaCl.svg?branch=dev)](https://travis-ci.org/CoAxLab/DeBaCl)
[![Pending Pull-Requests](http://githubbadges.herokuapp.com/CoAxLab/DeBaCl/pulls)](https://github.com/CoAxLab/DeBaCl/pulls)
[![Github Issues](http://githubbadges.herokuapp.com/CoAxLab/DeBaCl/issues)](https://github.com/CoAxLab/DeBaCl/issues)
[![License](http://img.shields.io/:license-bsd-blue.svg)](http://opensource.org/licenses/BSD-3-Clause)

DeBaCl is a Python library for **density-based clustering** with **level set trees**.

Level set trees are a statistically-principled way to represent the topology of
a probability density function. This representation is particularly useful for
several core tasks in statistics:

  - *clustering,* especially for data with multi-scale clustering behavior
  - *describing data topology*
  - *exploratory data analysis*
  - *data visualization*
  - *anomaly detection*

DeBaCl is an Python implementation of the Level Set Tree method, with an
emphasis on computational speed, algorithmic simplicity, and extensibility.

License
-------
DeBaCl is available under the 3-clause BSD license.

Installation
------------
DeBaCl can be downloaded and installed from the [Python package installer](https://pypi.python.org/pypi/debacl/0.2.0). From a terminal:

```bash
pip install debacl
```

It can also be installed by cloning this GitHub repo. This requires updating the Python path to include the cloned repo. One linux, this looks something like:

```bash
git clone https://github.com/CoAxLab/DeBaCl/
export PYTHONPATH='/home/brian/projects/DeBaCl'
```

Dependencies
------------
All of the dependencies are Python packages that can be installed with either conda or pip. DeBaCl 0.3 no longer includes the dependency on igraph, which required tricky manual installation.

**Required packages:**
  - numpy
  - networkx
  - prettytable

**Strongly recommended packages**
- matplotlib
- scipy

**Optional packages**
- scikit-learn

Quickstart
----------
<h4>Construct the level set tree</h4>
```python
import debacl as dcl
from sklearn.datasets import make_moons

X = make_moons(n_samples=100, noise=0.1)[0]

tree = dcl.construct_tree(X, k=10, prune_threshold=10)
print tree
```
```no-highlight
+----+-------------+-----------+------------+----------+------+--------+----------+
| id | start_level | end_level | start_mass | end_mass | size | parent | children |
+----+-------------+-----------+------------+----------+------+--------+----------+
| 0  |    0.000    |   0.162   |   0.000    |  0.140   | 100  |  None  |  [1, 2]  |
| 1  |    0.162    |   0.218   |   0.140    |  0.350   |  44  |   0    |  [3, 4]  |
| 2  |    0.162    |   0.468   |   0.140    |  1.000   |  42  |   0    |    []    |
| 3  |    0.218    |   0.423   |   0.350    |  0.980   |  16  |   1    |    []    |
| 4  |    0.218    |   0.373   |   0.350    |  0.940   |  18  |   1    |    []    |
+----+-------------+-----------+------------+----------+------+--------+----------+
```

<h4>Plot the level set tree</h4>
```python
fig = tree.plot()[0]
fig.show()
```

<h4>Query the level set tree for cluster labels</h4>
```python
import matplotlib.pyplot as plt

clusters = tree.get_clusters(method='leaf')  # each leaf node is a cluster

fig, ax = plt.subplots()
ax.scatter(X[:, 0], X[:, 1], c='black', alpha=0.4)
fig.show()
```

Running unit tests
------------------
From the top level of the repo:

```bash
$ nosetests -s -v debacl/test
```

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
- Chaudhuri, K., & Dasgupta, S. (2010). [Rates of Convergence for the Cluster        
  Tree](http://www.cse.ucsd.edu/sites/cse/files/cse/assets/research/theory/Chaud
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
