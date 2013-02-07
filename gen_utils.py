############################################
## Brian P. Kent
## gen_utils.py
## Created: 20120718
## Updated: 20130205
###########################################

##############
### SET UP ###
##############
"""
General utility functions for the DEnsity-BAsed CLustering (DeBaCl) toolbox.
"""

import numpy as np
import scipy.spatial.distance as spdist
import scipy.special as spspec




#################################
### GENERIC UTILITY FUNCTIONS ###
#################################

def drawSample(n, k):
	"""
	Draw a sample of size k from n items without replacement.
	
	Chooses k indices from range(n) without replacement by shuffling range(n) uniformly
	over all permutations. In numpy 1.7 and beyond, the "choice" function is a better
	option.
	
	Parameters
	----------
	n : int
		Total number of objects.
		
	k : int
		Sample size.
		
	Returns
	-------
	ix_keep : list of ints
		Indices of objects selected in the sample.
	"""
	
	ix = np.arange(n)
	np.random.shuffle(ix)
	ix_keep = ix[0:k]
	
	return ix_keep




#####################################
### SIMILARITY GRAPH CONSTRUCTION ###
#####################################

def knnGraph(x, k=None, q=0.05):
	"""Compute the symmetric k-NN adjacency matrix for a set of points.
	
	Parameters
	----------
	x : numpy array
		Data points, with each row as an observation.
	
	k : int, optional
		The number of points to consider as neighbors of any given observation. If not
		specified, use the default value of 'q'.
	
	q : float, optional
		The proportion of points to use as neighbors of a given observation. Defaults to
		0.05.
	
	Returns
	-------
	W : 2D numpy array of bool
		A 2D numpy array of shape n x n, where n is the number of rows in 'x'. The entry
		at position (i, j) is True if observations i and j are neighbors, False
		otherwise.
		
	k_radius : list of float
		For each row of 'x' the distance to its k-1'th nearest neighbor.
	"""

	n, p = x.shape
	if k == None:
		k = int(round(q * n))

	d = spdist.pdist(x, metric='euclidean')
	D = spdist.squareform(d)
			
	## identify which indices count as neighbors for each node
	rank = np.argsort(D, axis=1)
	ix_nbr = rank[:, 0:k]   # should this be k+1 to match Kpotufe paper?
	ix_row = np.tile(np.arange(n), (k, 1)).T
	
	## make adjacency matrix for unweighted graph
	W = np.zeros(D.shape, dtype=np.bool)
	W[ix_row, ix_nbr] = True
	W = np.logical_or(W, W.T)
	
	## find the radius of the k'th neighbor
	k_nbr = ix_nbr[:, -1]
	k_radius = D[np.arange(n), k_nbr]
		
		
	return W, k_radius
	
	

def gaussianGraph(x, sigma):
	"""
	Constructs a complete graph adjacency matrix with a Gaussian similarity kernel.
	
	Uses the rows of 'x' as vertices in a graph and connects each pair of vertices with
	an edge whose weight is the Gaussian kernel of the distance between the two
	vertices.
	
	Parameters
	----------
	x : 2D numpy array
		Rows of 'x' are locations of graph vertices.
	sigma : float
		The denominator of the Gaussian kernel.
	
	Returns
	-------
	W : 2D numpy array of floats
		Adjacency matrix of the Gaussian kernel complete graph on rows of 'x'. Each
		entry is a float representing the gaussian similarity between the corresponding
		rows of 'x'.
	"""
	
	d = spdist.pdist(x, metric='sqeuclidean')
	W = np.exp(-1 * d / sigma)
	W = spdist.squareform(W)
	
	return W
	


def epsilonGraph(x, eps=None, q=0.05):
	"""
	Constructs an epsilon-neighborhood graph adjacency matrix.
	
	Constructs a graph where the rows of 'x' are vertices and pairs of vertices are
	connected by edges if they are within euclidean distance epsilon of each other.
	Return the adjacency matrix for this graph.
	
	Parameters
	----------
	x : 2D numpy array
		The rows of x are the observations which become graph vertices.
	eps : float, optional
		The distance threshold for neighbors. If unspecified, defaults to the proportion
		in 'q'.
	q : float, optional
		If 'eps' is unspecified, this determines the neighbor threshold distance. 'eps'
		is set to the 'q' quantile of all (n choose 2) pairwise distances, where n is
		the number of rows in 'x'.
		
	Returns
	-------
	W : 2D numpy array of booleans
		The adjacency matrix for the graph.
	eps: float
		The neighbor threshold distance, useful particularly if not initially specified.
	"""
	
	d = spdist.pdist(x, metric='euclidean')
	D = spdist.squareform(d)

	if eps == None:
		eps = np.percentile(d, round(q*100))
		
	W = D <= eps

	return W, eps




##########################
### DENSITY ESTIMATION ###
##########################

def knnDensity(k_radius, n, p, k):
	"""
	Compute the kNN density estimate for a set of points.	
	
	Parameters
	----------
	k_radius : 1D numpy array of floats
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
		Estimated density for the points corresponding to the entries of 'k_radius'.
	"""
	
	unit_vol = np.pi**(p/2.0) / spspec.gamma(1 + p/2.0)
	const = (1.0 * k) / (n * unit_vol)
	fhat = const / k_radius**p
	
	return fhat




##########################################
### LEVEL SET TREE CLUSTERING PIPELINE ###
##########################################

def constructDensityGrid(density, mode='mass', n_grid=None):
	"""Create the inputs to a level set tree object.
	
	Create a list of lists of points to remove at each iteration of a level set or mass
	tree. Also create a list of the density level at each iteration.
	
	Parameters
	----------
	density : 1D numpy array
		An array with one value for each data point. Typically this is a density
		estimate, but it can be any function.
			
	mode : {'mass', 'levels'}, optional
		Determines if the tree should be built by removing a constant number of points
		(mass) at each iteration, or on a grid of evenly spaced density levels. If
		'n_grid' is set to None, the 'mass' option will remove 1 point at a time and the
		'levels' option will iterate through every unique value of the 'density' array.
	
	n_grid : int, optional
		The number of tree heights at which to estimate connected components. This is
		essentially the resolution of a level set tree built for the 'density' array.
	
	Returns
	-------
	bg_sets : list of lists
		Defines the points to remove as background at each iteration of level set tree
		construction.
	
	levels : array-like
		The density level at each iteration of level set tree construction.
	"""
	
	n = len(density)
	
	if mode == 'mass':
		pt_order = np.argsort(density)

		if n_grid is None:
			bg_sets = [[pt_order[i]] for i in range(n)]
			levels = density[pt_order]
		else:
			grid = np.linspace(0, n, n_grid)
			bg_sets = [pt_order[grid[i]:grid[i+1]] for i in range(n_grid-1)]
			levels = [max(density[x]) for x in bg_sets]
			
	elif mode == 'levels':
		uniq_dens = np.unique(density)
		uniq_dens.sort()

		if n_grid is None:
			bg_sets = [list(np.where(density==uniq_dens[i])[0])
				for i in range(len(uniq_dens))]
			levels = uniq_dens
		else:
			grid = np.linspace(np.min(uniq_dens), np.max(uniq_dens), n_grid)
			levels = grid.copy()
			grid = np.insert(grid, 0, -1)
			bg_sets = [list(np.where(np.logical_and(density > grid[i],
				density <= grid[i+1]))[0]) for i in range(n_grid)]
	
	else:
		bg_sets = []
		levels = []
		print "Sorry, didn't understand that mode."
	
	return bg_sets, levels




def assignBackgroundPoints(X, clusters, method=None, k=1):
	"""
	Assign level set tree background points to existing foreground clusters.
	
	This function packages a few very basic classification methods. Any classification
	method could work for this step of the data segmentation pipeline.
	
	Parameters
	----------
	X : 2D numpy array
		The original data, with rows as observations.
		
	clusters : 2D numpy array
		Foreground cluster assignments. Observation index is in the first entry of each
		row, with cluster label in the second entry. This is exactly what is returned by
		any of the LevelSetTree clustering methods.
		
	method : {None, 'centers', 'knn', 'zero'}, optional
		Which classification technique to use. The default of None sets background
		points to be a separate cluster. Option 'zero' does the same, but resets the
		cluster labels to the background points are labeled as '0'. The 'knn' method
		does a k-nearest neighbor classified, while option 'centers' assigns each
		background point to the cluster with the closet center (mean) point.
	
	k : int, optional
		If 'method' is 'knn', this is the number of neighbors to use for each
		observation.
	
	Returns
	-------
	labels : 2D numpy array
		Follows the same pattern as the 'clusters' parameter: each row is a data point,
		with the first entry as the observation index and the second entry the integer
		cluster label. Here though all points should be assigned, so the first column is
		just 1, ..., n, where n is the number of points.
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
		d = spdist.cdist(X_background, ctrs)
		ctr_min = np.argmin(d, axis=1)
		assignments[ix_background] = labels[ctr_min]	

		
	elif method == 'knn':
		# make sure k isn't too big
		k = min(k, np.min(np.bincount(clusters[:,1])))		

		# find distances between background and upper points
		X_background = X[ix_background, :]
		X_upper = X[clusters[:,0]]
		d = spdist.cdist(X_background, X_upper)

		# find the k-nearest neighbors
		rank = np.argsort(d, axis=1)
		ix_nbr = rank[:, 0:k]

		# find the cluster membership of the k-nearest neighbors
		knn_clusters = clusters[ix_nbr, 1]	
		knn_cluster_counts = np.apply_along_axis(np.bincount, 1, knn_clusters, None,
			n_label)
		knn_vote = np.argmax(knn_cluster_counts, axis=1)

		assignments[ix_background] = labels[knn_vote]
		
		
	elif method == 'zero':
		assignments += 1


	else:  # assume method == None
		assignments[ix_background] = max(labels) + 1


	labels = np.array([range(n), assignments], dtype=np.int).T
	return labels




