"""
General utility functions for the DEnsity-BAsed CLustering (DeBaCl) toolbox.
"""

## Required packages
try:
    import numpy as _np
except:
    raise ImportError("DeBaCl requires the numpy, networkx, and " +  
                      "prettytable packages.")

## Soft dependencies
try:
    import scipy.spatial.distance as _spd
    import scipy.special as _spspec
    _HAS_SCIPY = True
except:
    _HAS_SCIPY = False

try:
    import sklearn.neighbors as _sknbr
    _HAS_SKLEARN = True
except:
    _HAS_SKLEARN = False



#####################################
### SIMILARITY GRAPH CONSTRUCTION ###
#####################################

def knn_graph(X, k, method='brute_force', leaf_size=30):
    """
    Compute the symmetric k-nearest neighbor graph for a set of points. Assume
    Euclidean distance metric.

    Parameters
    ----------
    X : numpy array | list [numpy arrays]
        Data points, with each row as an observation.

    k : int
        The number of points to consider as neighbors of any given observation.

    method : {'brute-force', 'kd-tree', 'ball-tree'}, optional
        Computing method.

        - 'brute-force': computes the (Euclidean) distance between all O(n^2)
          pairs of rows in 'X', then for every point finds the k-nearest. It is
          limited to tens of thousands of observations (depending on available
          RAM).

        - 'kd-tree': partitions the data into axis-aligned rectangles to avoid
          computing all O(n^2) pairwise distances. Much faster than
          'brute-force', but only works for data in fewer than about 20
          dimensions. Requires the scikit-learn library.

        - 'ball-tree': partitions the data into balls and uses the metric
          property of euclidean distance to avoid computing all O(n^2)
          distances. Typically much faster than 'brute-force', and works with
          up to a few hundred dimensions. Requires the scikit-learn library.

    leaf_size : int, optional
        For the 'kd-tree' and 'ball-tree' methods, the number of observations
        in the leaf nodes. Leaves are not split further, so distance
        computations within leaf nodes are done by brute force. 'leaf_size' is
        ignored for the 'brute-force' method.

    Returns
    -------
    neighbors : numpy array
        Each row contains the nearest neighbors of the corresponding row in
        'X', indicated by row indices.

    radii : list[float]
        For each row of 'X' the distance to its k'th nearest neighbor
        (including itself).

    See Also
    --------
    epsilon_graph

    Examples
    --------
    >>> X = numpy.random.rand(100, 2)
    >>> knn, radii = debacl.utils.knn_graph(X, k=8, method='kd-tree')
    """

    n, p = X.shape

    if method == 'kd_tree':
        if _HAS_SKLEARN:
            kdtree = _sknbr.KDTree(X, leaf_size=leaf_size, metric='euclidean')
            distances, neighbors = kdtree.query(X, k=k,
                return_distance=True, sort_results=True)
            radii = distances[:, -1]
        else:
            raise ImportError("The scikit-learn library could not be loaded." +
                " It is required for the 'kd-tree' method.")

    if method == 'ball_tree':
        if _HAS_SKLEARN:
            btree = _sknbr.BallTree(X, leaf_size=leaf_size, metric='euclidean')
            distances, neighbors = btree.query(X, k=k,
                return_distance=True, sort_results=True)
            radii = distances[:, -1]
        else:
            raise ImportError("The scikit-learn library could not be loaded." +
                " It is required for the 'ball-tree' method.")

    else:  # assume brute-force
        if not _HAS_SCIPY:
            raise ImportError("The 'scipy' module could not be loaded. " +
                              "It is required for the 'brute_force' method " +
                              "for building a knn similarity graph.")

        d = _spd.pdist(X, metric='euclidean')
        D = _spd.squareform(d)
        rank = _np.argsort(D, axis=1)
        neighbors = rank[:, 0:k]

        k_nbr = neighbors[:, -1]
        radii = D[_np.arange(n), k_nbr]

    return neighbors, radii


def epsilon_graph(X, epsilon=None, percentile=0.05):
    """
    Construct an epsilon-neighborhood graph, represented by an adjacency list.
    Two vertices are connected by an edge if they are within 'epsilon' distance
    of each other, according to the Euclidean metric. The implementation is a
    brute-force computation of all O(n^2) pairwise distances of the rows in X.

    Parameters
    ----------
    X : 2D numpy array
        The rows of 'X' are the observations which become graph vertices.

    epsilon : float, optional
        The distance threshold for neighbors.

    percentile : float, optional
        If 'epsilon' is unspecified, this determines the distance threshold.
        'epsilon' is set to the desired percentile of all (n choose 2) pairwise
        distances, where n is the number of rows in 'X'.

    Returns
    -------
    neighbors : numpy array
        Each row contains the nearest neighbors of the corresponding row in
        'X', indicated by row indices.

    See Also
    --------
    knn_graph

    Examples
    --------
    >>> X = numpy.random.rand(100, 2)
    >>> neighbors = debacl.utils.epsilon_graph(X, epsilon=0.2)
    """

    if not _HAS_SCIPY:
        raise ImportError("The 'scipy' module could not be loaded. " +
                          "It is required for constructing an epsilon " +
                          "neighborhood similarity graph.")

    d = _spd.pdist(X, metric='euclidean')
    D = _spd.squareform(d)

    if epsilon == None:
        epsilon = _np.percentile(d, round(percentile*100))

    adjacency_matrix = D <= epsilon
    neighbors = [_np.where(row)[0] for row in adjacency_matrix]

    return neighbors



##########################
### DENSITY ESTIMATION ###
##########################

def knn_density(k_radius, n, p, k):
    """
    Compute the kNN density estimate for a set of points.

    Parameters
    ----------
    k_radius : 1-dimensional numpy array of floats
        The distance to each points k'th nearest neighbor.

    n : int
        The number of points.

    p : int
        The dimension of the data.

    k : int
        The number of observations considered neighbors of each point.

    Returns
    -------
    fhat : 1D numpy array of floats
        Estimated density for the points corresponding to the entries of
        'k_radius'.

    See Also
    --------
    knn_graph

    Examples
    --------
    >>> X = numpy.random.rand(100, 2)
    >>> knn, radii = debacl.utils.knn_graph(X, k=8, method='kd-tree')
    >>> density = debacl.utils.knn_density(radii, n=100, p=2, k=8)
    """

    if not _HAS_SCIPY:
        raise ImportError("The 'scipy' module could not be loaded." +
                          "It is required for computing knn density.")

    unit_vol = _np.pi**(p/2.0) / _spspec.gamma(1 + p/2.0)
    const = (1.0 * k) / (n * unit_vol)
    fhat = const / k_radius**p

    return fhat



##########################################
### LEVEL SET TREE CLUSTERING PIPELINE ###
##########################################

def define_density_mass_grid(density, num_levels=None):
    """
    Create a grid of density levels, such that a uniform number of points have
    density values between each level in the grid.

    Parameters
    ----------
    density : numpy array[float] or list[float]
        Values of a density estimate.

    num_levels : int, optional
        Number of density levels in the grid. This is essentially the vertical
        resolution of a level set tree built from the 'density' input.

    Returns
    -------
    levels : numpy array
        Grid of density levels that will define the iterations in level set
        tree construction.

    See Also
    --------
    define_density_level_grid

    Notes
    -----
    - The level set tree is constructed by filtering a similarity graph
      according to this grid. This function simply defines the density levels,
      but it does not do the actual tree construction.
    """
    ## Validate inputs
    if num_levels and not isinstance(num_levels, int):
        raise TypeError("Input 'num_levels' must be an integer.")

    if num_levels is not None and num_levels < 2:
        raise ValueError("Input 'num_levels' must be greater than or " +
                         "equal to 2.")

    if not isinstance(density, (_np.ndarray, list)):
        raise TypeError("Input 'density' must be a 1D numpy array or a " +
                        "list.")

    if isinstance(density, _np.ndarray) and len(density.shape) != 1:
        raise ValueError("Input 'density' must be 1-dimensional.")

    if len(density) < 1:
        raise ValueError("Input 'density' must contain at least one value.")

    ## Construct the grid
    n = len(density)

    if num_levels is None or num_levels > n:
        num_levels = n

    idx = _np.linspace(0, n-1, num_levels)
    idx = idx.astype(int)
    levels = _np.sort(density)[idx]
    levels = _np.unique(levels)
    return levels


def define_density_level_grid(density, num_levels=None):
    """
    Create a grid of density levels, evenly spaced between 0 and the maximum
    value of the input 'density'.

    Parameters
    ----------
    density : numpy array[float] or list[float]
        Values of a density estimate. The coordinates of the observation are
        not needed for this function.

    num_levels : int, optional
        Number of density levels in the grid. This is essentially the vertical
        resolution of a level set tree built from the 'density' input.

    Returns
    -------
    levels : numpy array
        Grid of density levels that will define the iterations in level set
        tree construction.

    See Also
    --------
    define_density_mass_grid

    Notes
    -----
    - The level set tree is constructed by filtering a similarity graph
      according to this grid. This function simply defines the density levels,
      but it does not do the actual tree construction.
    """

    ## Validate inputs
    if num_levels and not isinstance(num_levels, int):
        raise TypeError("Input 'num_levels' must be an integer.")

    if num_levels is not None and num_levels < 2:
        raise ValueError("Input 'num_levels' must be greater than or equal " +
                         "to 2.")

    if not isinstance(density, (_np.ndarray, list)):
        raise TypeError("Input 'density' must be a 1D numpy array or a " +
                        "list.")

    if isinstance(density, _np.ndarray) and len(density.shape) != 1:
        raise ValueError("Input 'density' must be 1-dimensional.")

    if len(density) < 1:
        raise ValueError("Input 'density' must contain at least one value.")

    ## Construct the grid
    n = len(density)

    if num_levels is None or num_levels > n:
        num_levels = n

    levels = _np.linspace(_np.min(density), _np.max(density), num_levels)
    levels = _np.unique(levels)
    return levels


def reindex_cluster_labels(labels):
    """
    Re-index integer cluster labels to be consecutive non-negative integers.
    This is useful because the `LevelSetTree.get_clusters` method returns
    cluster labels that match level set tree node indices. These are generally
    not consecutive whole numbers.

    Parameters
    ----------
    labels : numpy.array
        Cluster labels returned from the `LevelSetTree.get_clusters` method.
        The first column should be row indices and the second column should be
        integers corresponding to ID numbers of nodes in the level set tree.

    Returns
    -------
    new_labels : numpy.array
        Cluster labels in the same form of the input 'labels', but with cluster
        labels re-indexed to be consecutive non-negative integers.

    See Also
    --------
    LevelSetTree.get_clusters

    Examples
    --------
    >>> X = numpy.random.rand(100, 2)
    >>> tree = debacl.construct_tree(X, k=8, prune_threshold=5)
    >>> labels = tree.get_clusters(method='leaf')
    >>> numpy.unique(labels[:, 1])
    array([1, 5, 6])
    ...
    >>> new_labels = debacl.utils.reindex_cluster_labels(labels)
    >>> numpy.unique(new_labels[:, 1])
    array([0, 1, 2])
    """

    if not isinstance(labels, _np.ndarray):
        raise TypeError("Input 'labels' must be a numpy array.")

    if labels.ndim != 2:
        raise TypeError("Input 'labels' must be a 2-dimensional numpy array.")

    if labels.shape[1] != 2:
        raise TypeError("Input 'labels' must have two columns.")

    if not issubclass(labels.dtype.type, _np.integer):
        raise TypeError("Input 'labels' must contain integers.")

    unique_labels = _np.unique(labels[:, 1])
    label_map = {v: k for k, v in enumerate(unique_labels)}
    new_labels = map(lambda x: label_map[x], labels[:, 1])
    new_labels = _np.vstack((labels[:, 0], new_labels)).T
    return new_labels