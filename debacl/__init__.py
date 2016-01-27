"""
DeBaCl is a Python library for estimation of density level set trees and
nonparametric DEnsity-BAsed CLustering. Level set trees are based on the
statistically-principled definition of clusters as modes of a probability
density function. They are particularly useful for analyzing structure in
complex datasets that exhibit multi-scale clustering behavior. DeBaCl is
intended to promote the practical use of level set trees through improvements
in computational efficiency, flexible algorithms, and an emphasis on
modularity and user customizability.

DeBaCl is available under the 3-clause BSD license. Some useful links:

- `Source code <https://github.com/CoAxLab/DeBaCl>`_
- `API documentation <http://debacl.readthedocs.org/en/master/>`_
- `PyPI page <https://pypi.python.org/pypi/debacl>`_
"""

__version__ = '1.1'

from debacl.level_set_tree import construct_tree
from debacl.level_set_tree import construct_tree_from_graph
from debacl.level_set_tree import load_tree

from debacl.level_set_tree import LevelSetTree
