############################################
## Brian P. Kent
## debacl_utils.py
## Created: 20120718
## Updated: 20140623
## A library of helper functions for the DEnsity-BAsed CLustering (DeBaCl)
## package.
###########################################

##############
### SET UP ###
##############
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
	raise ImportError("Matplotlib could not be loaded. " +\
		"DeBaCl plot functions will not work.")



#####################################
### SIMILARITY GRAPH CONSTRUCTION ###
#####################################

def knn_graph(X, k, output_type='adjacency-list'):
	"""
	Compute the symmetric k-nearest neighbor graph for a set of points. Assume
	Euclidean distance metric.
	
	Parameters
	----------
	X : numpy array
		Data points, with each row as an observation.
	
	k : int
		The number of points to consider as neighbors of any given observation.

	output_type : {'adjacency-list', 'edge-list'}
		Form of the graph representation.
	
	Returns
	-------
	neighbors : numpy array [int]
		A 2-dimensional numpy array representing the k-nearest neighbor graph.
		If output_type is 'adjacency-list', this is a 'n x k' matrix, where 'n'
		is the number of rows in 'X'. The entry at position (i, j) is the index
		of the j'th nearest neighbor to the point at row index i. If output_type
		is 'edge-list', this is a two-column numpy array where each row
		corresponds to an edge in the graph. The edges are undirected and
		duplicates are removed.
		
	k_radius : list [float]
		For each row of 'X' the distance to its k'th nearest neighbor (including
		itself).
	"""

	n, p = X.shape

	## Find the index of the nearest neighbors for each point
	d = spd.pdist(X, metric='euclidean')
	D = spd.squareform(d)
			
	rank = np.argsort(D, axis=1)
	idx_neighbor = rank[:, 0:k]

	## Retrieve the k-neighbor radius
	k_nbr = idx_neighbor[:, -1]
	k_radius = D[np.arange(n), k_nbr]

	## Convert to an edge list, if desired
	if output_type == 'edge_list':
		neighbors = []
		for i, row in enumerate(idx_neighbor):
			v_incident = [tuple(sorted((i, v))) for v in row]
			neighbors += v_incident[:]

		neighbors = list(set(neighbors))

	else:  # output_type == 'adjacency_list'
		neighbors = idx_neighbor
		
	return neighbors, k_radius
	
	
def epsilon_graph(x, eps):
	"""
	Construct an epsilon-neighborhood graph, represented by an edge list. The
	rows of 'x' are vertices and pairs of vertices are connected by edges if
	they are within euclidean distance epsilon of each other.
	
	Parameters
	----------
	x : 2D numpy array
		The rows of x are the observations which become graph vertices.
		
	eps : float
		The distance threshold for neighbors. If unspecified, defaults to the
		proportion in 'q'.
		
	Returns
	-------
	neighbors : 2-dimensional numpy array [int]
	"""
	
	d = spd.pdist(x, metric='euclidean')
	D = spd.squareform(d)
	idx_neighbor = np.where(D <= eps)
	edge_list = [tuple(sorted(x)) for x in zip(idx_neighbor[0], idx_neighbor[1])]
	edge_list = list(set(edge_list))

	return edge_list


def remove_self_edges(edges):
	"""
	Removes self-edges from an edge list.

	Parameters
	----------
	edges : list [tuple [int]]
		Input edge list, possibly with self edges.

	Returns
	-------
	edges : list [tuple [int]]
		Edge list with no self edges.
	"""
	clean_edges = [x for x in edges if not x[0] == x[1]]
	return clean_edges


def adjacency_to_edge_list(adj_list, self_edge=False):
	"""
	Converts an adjacency list to a list of edges.

	Parameters
	----------
	adj_list : dict of lists.

	self_edge : boolean, optional

	Returns 
	-------
	edge_list : list of 2-tuples
	"""

	edge_list = []
	for k, v in adj_list.items():
		if self_edge:
			v_incident = [tuple(sorted(x)) for x in zip((k,)*len(v), v)]
		else:
			v_incident = [tuple(sorted(x)) for x in zip((k,)*len(v), v) 
				if not x[0] == x[1]]
		edge_list.append(v_incident[:])

	edge_list = [e for v in edge_list for e in v]
	edge_list = list(set(edge_list))

	return edge_list 


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



######################################
### PLOTTING FUNCITONS AND CLASSES ###
######################################
class Palette(object):
	"""
	Define some good RGB sscolors manually to simplify plotting upper level sets
	and foreground clusters.
	
	Parameters
	----------
	use : {'scatter', 'lines', 'neuroimg'}, optional
		Application for the palette. Different palettes work better in different
		settings.
	"""

	def __init__(self, use='scatter'):
		self.black = np.array([0.0, 0.0, 0.0])
		if use == 'lines':
			self.colorset = np.array([
				(228, 26, 28), #red
				(55, 126, 184), #blue
				(77, 175, 74), #green
				(152, 78, 163), #purple
				(255, 127, 0), #orange
				(166, 86, 40), #brown
				(0, 206, 209), #turqoise
				(82, 82, 82), #dark gray
				(247, 129, 191), #pink	
				(184, 134, 11), #goldenrod									
				]) / 255.0
				
		elif use == 'neuroimg':
			self.colorset = np.array([
				(170, 0, 0), # dark red
				(255, 0, 0), # red
				(0, 255, 0), # green
				(0, 0, 255), # blue
				(0, 255, 255), # cyan
				(255, 0, 255), # violet
				(255, 255, 0), # yellow
				]) / 255.0
			
		else:
			self.colorset = np.array([
					(228, 26, 28), #red
					(55, 126, 184), #blue
					(77, 175, 74), #green
					(152, 78, 163), #purple
					(255, 127, 0), #orange
					(247, 129, 191), #pink
					(166, 86, 40), #brown
					(0, 206, 209), #turqoise
					(85, 107, 47), #olive green
					(127, 255, 0), #chartreuse
					(205, 92, 92), #light red
					(0, 0, 128), #navy
					(255, 20, 147), #hot pink
					(184, 134, 11), #goldenrod
					(176, 224, 230), #light blue
					(255, 255, 51), #yellow
					(0, 250, 192),
					(13, 102, 113),
					(83, 19, 67),
					(162, 38, 132),
					(171, 15, 88),
					(204, 77, 51),
					(118, 207, 23), #lime green
					(207, 203, 23), #pea green
					(238, 213, 183), #bisque
					(82, 82, 82), #dark gray
					(150, 150, 150), #gray
					(240, 240, 240) # super light gray
					]) / 255.0
	
					
 	def apply_colorset(self, ix):
 		"""
 		Turn a numpy array of group labels (integers) into RGBA colors.
 		"""
 		n_clr = np.alen(self.colorset)
		return self.colorset[ix % n_clr] 		
 	 	
 	
def make_color_matrix(n, bg_color, bg_alpha, ix=None,
	fg_color=[228/255.0, 26/255.0, 28/255.0], fg_alpha=1.0):
	"""
	Construct the RGBA color parameter for a matplotlib plot.
	
	This function is intended to allow for a set of "foreground" points to be
	colored according to integer labels (e.g. according to clustering output),
	while "background" points are all colored something else (e.g. light gray).
	It is used primarily in the interactive plot tools for DeBaCl but can also
	be used directly by a user to build a scatterplot from scratch using more
	complicated DeBaCl output. Note this function can be used to build an RGBA
	color matrix for any aspect of a plot, including point face color, edge
	color, and line color, despite use of the term "points" in the descriptions
	below.
	
	Parameters
	----------
	n : int
		Number of data points.
		
	bg_color : list of floats
		A list with three entries, specifying a color in RGB format.
	
	bg_alpha : float
		Specifies background point opacity.
	
	ix : list of ints, optional
		Identifies foreground points by index. Default is None, which does not
		distinguish between foreground and background points.
	
	fg_color : list of ints or list of floats, optional
		Only relevant if 'ix' is specified. If 'fg_color' is a list of integers
		then each entry in 'fg_color' indicates the color of the corresponding
		foreground point. If 'fg_color' is a list of 3 floats, then all
		foreground points will be that RGB color. The default is to color all
		foreground points red.
	
	fg_alpha : float, optional
		Opacity of the foreground points.
	
	Returns
	-------
	rgba : 2D numpy array
		An 'n' x 4 RGBA array, where each row corresponds to a plot point.
	"""

	rgba = np.zeros((n, 4), dtype=np.float)
	rgba[:, 0:3] = bg_color
	rgba[:, 3] = bg_alpha
	
	if ix is not None:
		if np.array(fg_color).dtype.kind == 'i':
			palette = Palette()
			fg_color = palette.applyColorset(fg_color)
		
		rgba[ix, 0:3] = fg_color
		rgba[ix, 3] = fg_alpha
		
	return rgba
	
	
def cluster_histogram(x, cluster, fhat=None, f=None, levels=None):
	"""
	Plot a histogram and illustrate the location of selected cluster points.
	
	The primary plot axis is a histogram. Under this plot is a second axis that
	shows the location of the points in 'cluster', colored according to cluster
	label. If specified, also plot a density estimate, density function (or any
	function), and horizontal guidelines. This is the workhorse of the DeBaCl
	interactive tools for 1D data.
	
	Parameters
	----------
	x : 1D numpy array of floats
		The data.
	
	cluster : 2D numpy array
		A cluster matrix: rows represent points in 'x', with first entry as the
		index and second entry as the cluster label. The output of all
		LevelSetTree clustering methods are in this format.
		
	fhat : list of floats, optional
		Density estimate values for the data in 'x'. Plotted as a black curve,
		with points colored according to 'cluster'.
	
	f : 2D numpy array, optional
		Any function. Arguments in the first column and values in the second.
		Plotted independently of the data as a blue curve, so does not need to
		have the same number of rows as values in 'x'. Typically this is the
		generating probability density function for a 1D simulation.
	
	levels : list of floats, optional
		Each entry in 'levels' causes a horizontal dashed red line to appear at
		that value.
	
	Returns
	-------
	fig : matplotlib figure
		Use fig.show() to show the plot, fig.savefig() to save it, etc.
	"""
	
	n = len(x)
	palette = Palette()
	
	## set up the figure and plot the data histogram
	fig, (ax0, ax1) = plt.subplots(2, sharex=True)
	ax0.set_position([0.125, 0.12, 0.8, 0.78])
	ax1.set_position([0.125, 0.05, 0.8, 0.05])

	ax1.get_yaxis().set_ticks([])
	ax0.hist(x, bins=n/20, normed=1, alpha=0.18)
	ax0.set_ylabel('Density')
	
	
	## plot the foreground points in the second axes
	for i, c in enumerate(np.unique(cluster[:, 1])):
		ix = cluster[np.where(cluster[:, 1] == c)[0], 0]
		ax1.scatter(x[ix], np.zeros((len(ix),)), alpha=0.08, s=20,
			color=palette.colorset[i])
		
		if fhat is not None:
			ylim = ax0.get_ylim()
			eps = 0.02 * (max(fhat) - min(fhat))
			ax0.set_ylim(bottom=min(0.0-eps, ylim[0]), top=max(max(fhat)+eps,
				ylim[1]))
			ax0.scatter(x[ix], fhat[ix], s=12, alpha=0.5,
				color=palette.colorset[i])
	
	if f is not None:	# plot the density
		ax0.plot(f[:,0], f[:,1], color='blue', ls='-', lw=1)
	
	if fhat is not None:  # plot the estimated density 
		ax0.plot(x, fhat, color='black', lw=1.5, alpha=0.6)

	if levels is not None:  # plot horizontal guidelines
		for lev in levels:
			ax0.axhline(lev, color='red', lw=1, ls='--', alpha=0.7)
			
	return fig
	

def plot_foreground(X, clusters, title='', xlab='x', ylab='y', zlab='z',
	fg_alpha=0.75, bg_alpha=0.3, edge_alpha=1.0, **kwargs):
	"""
	Draw a scatter plot of 2D or 3D data, colored according to foreground
	cluster label.
	
	Parameters
	----------
	X : 2-dimensional numpy array
		Data points represented by rows. Must have 2 or 3 columns.
		
	clusters : 2-dimensional numpy array
		A cluster matrix: rows represent points in 'x', with first entry as the
		index and second entry as the cluster label. The output of all
		LevelSetTree clustering methods are in this format.
		
	title : string
		Axes title
		
	xlab, ylab, zlab : string
		Axes axis labels
		
	fg_alpha : float
		Transparency of the foreground (clustered) points. A float between 0
		(transparent) and 1 (opaque).
		
	bg_alpha : float
		Transparency of the background (unclustered) points. A float between 0
		(transparent) and 1 (opaque).
		
	kwargs : keyword parameters
		Plot parameters passed through to Matplotlib Axes.scatter function.
			
	Returns
	-------
	fig : matplotlib figure
		Use fig.show() to show the plot, fig.savefig() to save it, etc.
		
	ax : matplotlib axes object
		Allows more direct plot customization in the client function.
	"""
	
	## make the color matrix
	n, p = X.shape
	base_clr = [190.0 / 255.0] * 3  ## light gray
	black = [0.0, 0.0, 0.0]
	
	rgba_edge = makeColorMatrix(n, bg_color=black, bg_alpha=edge_alpha, ix=None)
	rgba_clr = makeColorMatrix(n, bg_color=base_clr, bg_alpha=bg_alpha,
		ix=clusters[:, 0], fg_color=clusters[:, 1], fg_alpha=fg_alpha)
		
	if p == 2:
		fig, ax = plt.subplots()
		ax.scatter(X[:,0], X[:,1], c=rgba_clr, edgecolors=rgba_edge, **kwargs)
		
	elif p == 3:
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		fig.subplots_adjust(bottom=0.0, top=1.0, left=-0.05, right=0.98)
		ax.set_zlabel(zlab)
		ax.scatter(X[:,0], X[:,1], X[:,2], c=rgba_clr, edgecolors=rgba_edge,
		 	**kwargs)
		
	else:
		fig, ax = plt.subplots()
		print "Plotting failed due to a dimension problem."
		
	ax.set_title(title)
	ax.set_xlabel(xlab); ax.set_ylabel(ylab)
			
	return fig, ax


def set_plot_params(axes_titlesize=22, axes_labelsize=18, xtick_labelsize=14,
	ytick_labelsize=14, figsize=(9, 9), n_ticklabel=4):
	"""
	A handy function for setting matplotlib parameters without adding trival
	code to working scripts.
	
	Parameters
	----------
	axes_titlesize : integer
		Size of the axes title.
	
	axes_labelsize : integer
		Size of axes dimension labels.
	
	xtick_labelsize : integer
		Size of the ticks on the x-axis.
	
	ytick_labelsize : integer
		Size of the ticks on the y-axis.
	
	figure_size : tuple (length 2)
		Size of the figure in inches.
	
	Returns
	-------
	"""
	
	mpl.rc('axes', labelsize=axes_labelsize)
	mpl.rc('axes', titlesize=axes_titlesize) 
	mpl.rc('xtick', labelsize=xtick_labelsize)
	mpl.rc('ytick', labelsize=ytick_labelsize) 
	mpl.rc('figure', figsize=figsize)

	def autoloc(self):
		ticker.MaxNLocator.__init__(self, nbins=n_ticklabel)
	ticker.AutoLocator.__init__ = autoloc





