"""
General utility functions for the DEnsity-BAsed CLustering (DeBaCl) toolbox.
"""

import numpy as np
import scipy.spatial.distance as spd
import scipy.special as spspec

try:
	import matplotlib.pyplot as plt
	from mpl_toolkits.mplot3d import Axes3D
	import matplotlib as mpl
	from matplotlib import ticker
except:
	print "Matplotlib could not be loaded. DeBaCl plot functions will not work."

try:
	import sklearn.neighbors as sknbr
	_HAS_SKLEARN = True
except:
	_HAS_SKLEARN = False



#####################################
### SIMILARITY GRAPH CONSTRUCTION ###
#####################################

def knn_graph(X, k, method='brute-force', leaf_size=30):
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
          distances. Typically much faster than 'brute-force', and works with up
          to a few hundred dimensions. Requires the scikit-learn library.

    leaf_size : int, optional
    	For the 'kd-tree' and 'ball-tree' methods, the number of observations in
    	the leaf nodes. Leaves are not split further, so distance computations
    	within leaf nodes are done by brute force. 'leaf_size' is ignored for
    	the 'brute-force' method.

	Returns
	-------
	neighbors : dict [list]
		The keys correspond to rows of X and the values are the k-nearest
		neighbors to the key's row.

	k_radius : list [float]
		For each row of 'X' the distance to its k'th nearest neighbor (including
		itself).
	"""

	n, p = X.shape

	if method == 'kd-tree':
		if _HAS_SKLEARN:
			kdtree = sknbr.KDTree(X, leaf_size=leaf_size, metric='euclidean')
			distances, idx_neighbors = kdtree.query(X, k=k,
				return_distance=True, sort_results=True)
			k_radius = distances[:, -1]
		else:
			raise ImportError("The scikit-learn library could not be loaded." + \
				" It is required for the 'kd-tree' method.")

	if method == 'ball-tree':
		if _HAS_SKLEARN:
			btree = sknbr.BallTree(X, leaf_size=leaf_size, metric='euclidean')
			distances, idx_neighbors = btree.query(X, k=k,
				return_distance=True, sort_results=True)
			k_radius = distances[:, -1]
		else:
			raise ImportError("The scikit-learn library could not be loaded." +\
				" It is required for the 'ball-tree' method.")

	else:  # assume brute-force
		d = spd.pdist(X, metric='euclidean')
		D = spd.squareform(d)
		rank = np.argsort(D, axis=1)
		idx_neighbors = rank[:, 0:k]

		k_nbr = idx_neighbors[:, -1]
		k_radius = D[np.arange(n), k_nbr]

	return idx_neighbors, k_radius


def epsilon_graph(X, epsilon=None, percentile=0.05):
	"""
	Construct an epsilon-neighborhood graph, represented by an adjacency list.
	Two vertices are connected by an edge if they are within 'epsilon' distance
	of each other, according to the Euclidean metric. The implementation is a
	brute-force computation of all O(n^2) pairwise distances of the rows in X.

	Parameters
	----------
	X : 2D numpy array
		The rows of x are the observations which become graph vertices.

	epsilon : float, optional
		The distance threshold for neighbors.

	percentile : float, optional
		If 'epsilon' is unspecified, this determines the distance threshold.
		'epsilon' is set to the desired percentile of all (n choose 2) pairwise
		distances, where n is the number of rows in 'X'.

	Returns
	-------
	neighbors : dict [list]
		The keys correspond to rows of X and the values are the k-nearest
		neighbors to the key's row.
	"""

	d = spd.pdist(X, metric='euclidean')
	D = spd.squareform(d)

	if epsilon == None:
		epsilon = np.percentile(d, round(percentile*100))

	neighbor_flag = D <= epsilon
	neighbors = {i: np.where(row)[0] for i, row in enumerate(neighbor_flag)}

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
	"""

	unit_vol = np.pi**(p/2.0) / spspec.gamma(1 + p/2.0)
	const = (1.0 * k) / (n * unit_vol)
	fhat = const / k_radius**p

	return fhat



##########################################
### LEVEL SET TREE CLUSTERING PIPELINE ###
##########################################

def define_density_grid(density, mode='mass', num_levels=None):
    """
    Create the inputs to level set tree estimation by pruning observations from
    a density estimate at successively higher levels. This function merely
    records the density levels and which points will be removed at each level,
    but it does not do the actual tree construction.

    Parameters
    ----------
    density : numpy array
        Values of a density estimate. The coordinates of the observation are not
        needed for this function.

    mode : {'mass', 'levels'}, optional
        If 'mass', the level set tree will be built by removing a constant
        number of points (mass) at each iteration. If 'levels', the density
        levels are evenly spaced between 0 and the maximum density estimate
        value. If 'num_levels' is 'None', the 'mass' option removes 1 point at a
        time and the 'levels' option iterates through unique values of the
        'density' array.

    num_levels : int, optional
        Number of density levels at which to construct the level set tree.
        This is essentially the resolution of a level set tree built for the
        'density' array.

    Returns
    -------
    levels : numpy array
        Grid of density levels that will define the iterations in level set tree
        construction.
    """

    n = len(density)

    if mode == 'mass':  # remove blocks of points of uniform mass
        pt_order = np.argsort(density)

        if num_levels is None:
            levels = density[pt_order]
        else:
            bin_size = n / num_levels  # this should be only the integer part
            background_sets = [pt_order[i:(i + bin_size)]
                for i in range(0, n, bin_size)]
            levels = [max(density[x]) for x in background_sets]

    elif mode == 'levels':  # remove points at evenly spaced density levels
        uniq_dens = np.unique(density)
        uniq_dens.sort()

        if num_levels is None:
            levels = uniq_dens
        else:
            grid = np.linspace(0., np.max(uniq_dens), num_levels)
            levels = grid.copy()

    else:
        raise ValueError("Sorry, that's not a valid mode.")

    return levels

def assign_background_points(X, clusters, method=None, k=1):
	"""
	Assign level set tree background points to existing foreground clusters.
	This function packages a few very basic classification methods. Any
	classification method could work for this step of the data segmentation
	pipeline.

	Parameters
	----------
	X : 2-dimensional numpy array
		The original data, with rows as observations.

	clusters : 2D numpy array
		Foreground cluster assignments. Observation index is in the first entry
		of each row, with cluster label in the second entry. This is exactly
		what is returned by any of the LevelSetTree clustering methods.

	method : {None, 'centers', 'knn', 'zero'}, optional
		Which classification technique to use. The default of None sets
		background points to be a separate cluster. Option 'zero' does the same,
		but resets the cluster labels to the background points are labeled as
		'0'. The 'knn' method does a k-nearest neighbor classified, while option
		'centers' assigns each background point to the cluster with the closet
		center (mean) point.

	k : int, optional
		If 'method' is 'knn', this is the number of neighbors to use for each
		observation.

	Returns
	-------
	labels : 2-dimensional numpy array
		Follows the same pattern as the 'clusters' parameter: each row is a data
		point, with the first entry as the observation index and the second
		entry the integer cluster label. Here though all points should be
		assigned, so the first column is just 1, ..., n, where n is the number
		of points.
	"""

	n, p = X.shape
	labels = np.unique(clusters[:,1])
	n_label = len(labels)

	assignments = np.zeros((n, ), dtype=np.int) - 1
	assignments[clusters[:,0]] = clusters[:,1]
	ix_background = np.where(assignments == -1)[0]

	if len(ix_background) == 0:
		return clusters


	if method == 'centers':
		# get cluster centers
		ctrs = np.empty((n_label, p), dtype=np.float)
		ctrs.fill(np.nan)

		for i, c in enumerate(labels):
			ix_c = clusters[np.where(clusters[:,1] == c)[0], 0]
			ctrs[i, :] = np.mean(X[ix_c,:], axis=0)

		# get the background points
		X_background = X[ix_background, :]

		# distance between each background point and all cluster centers
		d = spd.cdist(X_background, ctrs)
		ctr_min = np.argmin(d, axis=1)
		assignments[ix_background] = labels[ctr_min]


	elif method == 'knn':
		# make sure k isn't too big
		k = min(k, np.min(np.bincount(clusters[:,1])))

		# find distances between background and upper points
		X_background = X[ix_background, :]
		X_upper = X[clusters[:,0]]
		d = spd.cdist(X_background, X_upper)

		# find the k-nearest neighbors
		rank = np.argsort(d, axis=1)
		ix_nbr = rank[:, 0:k]

		# find the cluster membership of the k-nearest neighbors
		knn_clusters = clusters[ix_nbr, 1]
		knn_cluster_counts = np.apply_along_axis(np.bincount, 1, knn_clusters,
			None, n_label)
		knn_vote = np.argmax(knn_cluster_counts, axis=1)

		assignments[ix_background] = labels[knn_vote]


	elif method == 'zero':
		assignments += 1


	else:  # assume method == None
		assignments[ix_background] = max(labels) + 1


	labels = np.array([range(n), assignments], dtype=np.int).T
	return labels


