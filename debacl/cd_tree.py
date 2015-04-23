##############################################################
## Brian P. Kent
## cd_tree.py
## Created: 20130417
## Updated: 20130417
##############################################################

##############
### SET UP ###
##############
"""
Main functions and classes for construction and use of Chaudhuri-Dasgupta level
set trees. A companion to debacl.py, which has a more developed set of tools for
working with generic level set trees.
"""

try:
	import numpy as np
	import scipy.spatial.distance as spd
	import scipy.io as spio
	import igraph as igr
	import utils as utl
except:
	print "Critical packages are not installed."

try:
	import matplotlib.pyplot as plt
	from matplotlib.collections import LineCollection
	import pandas as pd
except:
	print "Matplotlib and/or Pandas packages are not installed, so plot and " +\
		"print functions may fail."



#####################
### BASIC CLASSES ###
#####################
class ConnectedComponent(object):
	"""
	Defines a connected component for level set tree construction. A level set
	tree is really just a set of ConnectedComponents.
	"""

	def __init__(self, idnum, parent, children, start_radius, end_radius,
		members):
		self.idnum = idnum
		self.parent = parent
		self.children = children
		self.start_radius = start_radius
		self.end_radius = end_radius
		self.members = members


	def copy(self):
		"""
		Creates and returns a copy of a CD_Component object.

		Parameters
		----------

		Returns
		-------
		component : CD_Component
		"""

		component = ConnectedComponent(self.idnum, self.parent, self.children,
			self.start_radius, self.end_radius,	self.members)

		return component




class CDTree(object):
	"""
	Defines methods and attributes for a Chaudhuri-Dasgupta level set tree.
	"""

	def __init__(self):
		self.nodes = {}
		self.subgraphs = {}


	def __str__(self):
		"""
		Produce a tree summary table with Pandas. This can be printed to screen
		or saved easily to CSV. This is much simpler than manually formatting
		the strings in the CoAxLab version of DeBaCl, but it does require
		Pandas.
		"""

		summary = pd.DataFrame()
		for u, v in self.nodes.items():
			row = {
				'key': u,
				'r1': v.start_radius,
				'r2': v.end_radius,
				'size': len(v.members),
				'parent': v.parent,
				'children': v.children
				}
			row = pd.DataFrame([row])
			summary = summary.append(row)

		summary.set_index('key', inplace=True)
		out = summary.to_string()
		return out


	def prune(self, method='size-merge', **kwargs):
		"""
		Prune the tree. A dispatch function to other methods.

		Parameters
		----------
		method : {'size-merge'}

		gamma : integer
			Nodes smaller than this will be merged (for 'size-merge') or cut
			(for 'size-cut')

		Notes
		-----
		Modifies the tree in-place.
		"""

		if method == 'size-merge':
			required = set(['gamma'])
			if not set(kwargs.keys()).issuperset(required):
				raise ValueError("Incorrect arguments for size-merge pruning.")
			else:
				gamma = kwargs.get('gamma')
				self.mergeBySize(gamma)

		else:
			print "Pruning method not understood. 'size-merge' is the only " +\
			"pruning method currently implemented. No changes were made to " + \
			"the tree."


	def save(self, fname):
		"""
		Save a level set tree object to file.

		Saves a level set tree as a MATLAB struct using the scipy.io module.
		Ignore the warning about using oned_as default value ('column').

		Parameters
		----------
		fname : string
			File to save the tree to. The .mat extension is not necessary.
		"""

		tree_dict = {
			'bg_sets': self.bg_sets,
			'levels': self.levels,
			'idnums': [x.idnum for x in self.nodes.values()],
			'start_radii': [x.start_radius for x in self.nodes.values()],
			'end_radii': [x.end_radius for x in self.nodes.values()],
			'parents': [(-1 if x.parent is None else x.parent)
				for x in self.nodes.values()],
			'children': [x.children for x in self.nodes.values()],
			'members': [x.members for x in self.nodes.values()]
			}

		spio.savemat(fname, tree_dict)


	def makeSubtree(self, ix):
		"""
		Return the subtree with node 'ix' as the root, and all ancestors of 'ix'.

		Parameters
		----------
		ix : int
			Node to use at the root of the new tree.

		Returns
		-------
		T : LevelSetTree
			A completely indpendent level set tree, with 'ix' as the root node.
		"""

		T = CDTree()
		T.nodes[ix] = self.nodes[ix].copy()
		T.nodes[ix].parent = None
		queue = self.nodes[ix].children[:]

		while len(queue) > 0:
			branch_ix = queue.pop()
			T.nodes[branch_ix] = self.nodes[branch_ix]
			queue += self.nodes[branch_ix].children

		return T


	def mergeBySize(self, threshold):
		"""
		Prune splits from a tree based on size of child nodes. Merge members of
		child nodes rather than removing them.

		Parameters
		----------
		threshold : numeric
			Tree branches with fewer members than this will be merged into
			larger siblings or parents.

		Notes
		-----
		Modifies a level set tree in-place.
		"""

		## remove small root branches
		small_roots = [k for k, v in self.nodes.iteritems()
			if v.parent==None and len(v.members) <= threshold]

		for root in small_roots:
			root_tree = makeSubtree(self, root)
			for ix in root_tree.nodes.iterkeys():
				del self.nodes[ix]


		## main pruning
		parents = [k for k, v in self.nodes.iteritems() if len(v.children) >= 1]
		parents = np.sort(parents)[::-1]

		for ix_parent in parents:
			parent = self.nodes[ix_parent]

			# get size of each child
			kid_size = {k: len(self.nodes[k].members) for k in parent.children}

			# count children larger than 'threshold'
			n_bigkid = sum(np.array(kid_size.values()) >= threshold)

			if n_bigkid == 0:
				# update parent's end level and end mass
				parent.end_radius = max([self.nodes[k].end_radius
					for k in parent.children])

				# remove small kids from the tree
				for k in parent.children:
					del self.nodes[k]
				parent.children = []

			elif n_bigkid == 1:
				pass
				# identify the big kid
				ix_bigkid = [k for k, v in kid_size.iteritems() if v >= threshold][0]
				bigkid = self.nodes[ix_bigkid]

				# update k's end radius
				parent.end_radius = bigkid.end_radius

				# set grandkids' parent to k
				for c in bigkid.children:
					self.nodes[c].parent = ix_parent

				# delete small kids
				for k in parent.children:
					if k != ix_bigkid:
						del self.nodes[k]

				# set k's children to grandkids
				parent.children = bigkid.children

				# delete the single bigkid
				del self.nodes[ix_bigkid]

			else:
				pass  # do nothing here


	def plot(self, width='uniform', gap=0.05):
		"""
		Create a static plot of a Chaudhuri-Dasgupta level set tree.

		Parameters
		----------
		width : {'uniform', 'mass'}, optional
			Determines how much horzontal space each level set tree node is
			given. The default of "uniform" gives each child node an equal
			fraction of the parent node's horizontal space. If set to 'mass',
			then horizontal space is allocated proportional to the mass (i.e.
			fraction of points) of a node relative to its siblings.

		sort : bool
			If True, sort sibling nodes from most to least points and draw left
			to right. Also sorts root nodes in the same way.

		gap : float
			Fraction of vertical space to leave at the bottom. Default is 5%,
			and 0% also works well. Higher values are used for interactive tools
			to make room for buttons and messages.

		Returns
		-------
		fig : matplotlib figure
			Use fig.show() to view, fig.savefig() to save, etc.
		"""

		## Initialize the plot containers
		segments = {}
		splits = {}
		segmap = []
		splitmap = []

		## Find the root connected components and corresponding plot intervals
		ix_root = np.array([k for k, v in self.nodes.iteritems()
			if v.parent is None])
		n_root = len(ix_root)
		census = np.array([len(self.nodes[x].members) for x in ix_root],
			dtype=np.float)
		n = sum(census)

		## Order the roots by mass decreasing from left to right
		seniority = np.argsort(census)[::-1]
		ix_root = ix_root[seniority]
		census = census[seniority]

		if width == 'mass':
			weights = census / n
			intervals = np.cumsum(weights)
			intervals = np.insert(intervals, 0, 0.0)
		else:
			intervals = np.linspace(0.0, 1.0, n_root+1)


		## Do a depth-first search on each root to get segments for each branch
		for i, ix in enumerate(ix_root):
			branch = self.constructBranchMap(ix, (intervals[i], intervals[i+1]),
				width)
			branch_segs, branch_splits, branch_segmap, branch_splitmap = branch

			segments = dict(segments.items() + branch_segs.items())
			splits = dict(splits.items() + branch_splits.items())
			segmap += branch_segmap
			splitmap += branch_splitmap


		## get the the vertical line segments in order of the segment map (segmap)
		verts = [segments[k] for k in segmap]
		lats = [splits[k] for k in splitmap]

		## Find the fraction of nodes in each segment (to use as linewidths)
		thickness = [max(1.0, 12.0 * len(self.nodes[x].members)/n) for x in segmap]


		## Find the right tick marks for the plot
#		radius_ticks = np.sort(list(set(
#			[v.start_radius for v in self.nodes.itervalues()] + \
#			[v.end_radius for v in self.nodes.itervalues()])))
#		radius_tick_labels = [str(round(lvl, 2)) for lvl in radius_ticks]

		primary_ticks = [(x[0][1], x[1][1]) for x in segments.values()]
		primary_ticks = np.unique(np.array(primary_ticks).flatten())
		primary_labels = [str(round(tick, 2)) for tick in primary_ticks]


		## Set up the plot framework
		frame_dims = [0.15, 0.05, 0.8, 0.93]

		fig, ax = plt.subplots()
		ax.set_position(frame_dims)
		ax.set_xlim((-0.04, 1.04))
		ax.set_xticks([])
		ax.set_xticklabels([])
		ax.yaxis.grid(color='gray')
		ax.set_yticks(primary_ticks)
		ax.set_yticklabels(primary_labels)


		## Add the line segments
		segclr = np.array([[0.0, 0.0, 0.0]] * len(segmap))
		splitclr = np.array([[0.0, 0.0, 0.0]] * len(splitmap))

		linecol = LineCollection(verts, linewidths=thickness, colors=segclr)
		ax.add_collection(linecol)

		splitcol = LineCollection(lats, colors=splitclr)
		ax.add_collection(splitcol)


		## Make the plot
		ax.set_ylabel("Radius")
		ymax = max([v.start_radius for v in self.nodes.itervalues()])
		ymin = min([v.end_radius for v in self.nodes.itervalues()])
		rng = ymax - ymin
		ax.set_ylim(ymin - gap*rng, ymax + 0.05*rng)
		ax.invert_yaxis()

		return fig


	def getClusterLabels(self, method='all-mode', **kwargs):
		"""
		Umbrella function for retrieving custer labels from the level set tree.

		Parameters
		----------
		method : {'all-mode', 'first-k', 'upper-set', 'k-level'}, optional
			Method for obtaining cluster labels from the tree. 'all-mode' treats
			each leaf of the tree as a separate cluter. 'first-k' finds the
			first K non-overlapping clusters from the roots of the tree.
			'upper-set' returns labels by cutting the tree at a specified
			density (lambda) or mass (alpha) level. 'k-level' returns labels at
			the lowest density level that has k nodes.

		k : integer
			If method is 'first-k' or 'k-level', this is the desired number of
			clusters.

 		threshold : float
 			If method is 'upper-set', this is the threshold at which to cut the
 			tree.

		Returns
		-------
		labels : 2-dimensional numpy array
			Each row corresponds to an observation. The first column indicates
			the index of the observation in the original data matrix, and the
			second column is the integer cluster label (starting at 0). Note
			that the set of observations in this "foreground" set is typically
			smaller than the original dataset.

		nodes : list
			Indices of tree nodes corresponding to foreground clusters.
		"""

		if method == 'all-mode':
			labels, nodes = self.allModeCluster()

		elif method == 'first-k':
			required = set(['k'])
			if not set(kwargs.keys()).issuperset(required):
				raise ValueError("Incorrect arguments for the first-k " + \
				"cluster labeling method.")
			else:
				k = kwargs.get('k')
				labels, nodes = self.firstKCluster(k)

		elif method == 'upper-set':
			required = set(['threshold'])
			if not set(kwargs.keys()).issuperset(required):
				raise ValueError("Incorrect arguments for the upper-set " + \
				"cluster labeling method.")
			else:
				threshold = kwargs.get('threshold')
				labels, nodes = self.upperSetCluster(threshold)

		else:
			print 'method not understood'
			labels = np.array([])
			nodes = []

 		return labels, nodes


 	def upperSetCluster(self, threshold):
		"""
		Set foreground clusters by finding connected components at an upper
		level set. This is slightly different than GeomTree.upperSetCluster in
		that this method returns all members of tree nodes that cross the
		desired threshold, rather than the components of the true upper level
		set.

		Parameters
		----------
		threshold : float
			The radius that defines the foreground set of points.

		Returns
		-------
		labels : 2-dimensional numpy array
			Each row corresponds to an observation. The first column indicates
			the index of the observation in the original data matrix, and the
			second column is the integer cluster label (starting at 0). Note
			that the set of observations in this "foreground" set is typically
			smaller than the original dataset.

		nodes : list
			Indices of tree nodes corresponding to foreground clusters.
		"""

		## identify upper level points and the nodes active at the cut
		nodes = [k for k, v in self.nodes.iteritems()
			if v.start_radius >= threshold and v.end_radius < threshold]

		## find intersection between upper set points and each active component
		points = []
		cluster = []

		for i, c in enumerate(nodes):
			points.extend(self.nodes[c].members)
			cluster += ([i] * len(self.nodes[c].members))

		labels = np.array([points, cluster], dtype=np.int).T
		return labels, nodes


	def allModeCluster(self):
		"""
		Set every leaf node as a foreground cluster.

		Parameters
		----------

		Returns
		-------
		labels : 2-dimensional numpy array
			Each row corresponds to an observation. The first column indicates
			the index of the observation in the original data matrix, and the
			second column is the integer cluster label (starting at 0). Note
			that the set of observations in this "foreground" set is typically
			smaller than the original dataset.

		leaves : list
			Indices of tree nodes corresponding to foreground clusters. This is
			the same as 'nodes' for other clustering functions, but here they
			are also the leaves of the tree.
		"""

		leaves = [k for k, v in self.nodes.items() if v.children == []]

		## find components in the leaves
		points = []
		cluster = []

		for i, k in enumerate(leaves):
			points.extend(self.nodes[k].members)
			cluster += ([i] * len(self.nodes[k].members))

		labels = np.array([points, cluster], dtype=np.int).T
		return labels, leaves


	def firstKCluster(self, k):
		"""
		Returns foreground cluster labels for the 'k' modes with the lowest
		start levels. In principle, this is the 'k' leaf nodes with the smallest
		indices, but this function double checks by finding and ordering all
		leaf start values and ordering.

		Parameters
		----------
		k : integer
			The desired number of clusters.

		Returns
		-------
		labels : 2-dimensional numpy array
			Each row corresponds to an observation. The first column indicates
			the index of the observation in the original data matrix, and the
			second column is the integer cluster label (starting at 0). Note
			that the set of observations in this "foreground" set is typically
			smaller than the original dataset.

		nodes : list
			Indices of tree nodes corresponding to foreground clusters.
		"""

		parents = np.array([u for u, v in self.nodes.items()
			if len(v.children) > 0])
		roots = [u for u, v in self.nodes.items() if v.parent is None]
		splits = [self.nodes[u].end_radius for u in parents]
		order = np.argsort(splits)
		star_parents = parents[order[:(k-len(roots))]]

		children = [u for u, v in self.nodes.items() if v.parent is None]
		for u in star_parents:
			children += self.nodes[u].children

		nodes = [x for x in children if
			sum(np.in1d(self.nodes[x].children, children))==0]


		points = []
		cluster = []

		for i, c in enumerate(nodes):
			cluster_pts = self.nodes[c].members
			points.extend(cluster_pts)
			cluster += ([i] * len(cluster_pts))

		labels = np.array([points, cluster], dtype=np.int).T
		return labels, nodes


	def constructBranchMap(self, ix, interval, width):
		"""
		Map level set tree nodes to locations in a plot canvas. Finds the plot
		coordinates of vertical line segments corresponding to LST nodes and
		horizontal line segments corresponding to node splits. Also provides
		indices of vertical segments and splits for downstream use with
		interactive plot picker tools. This function is not meant to be called
		by the user; it is a helper function for the LevelSetTree.plot() method.
		This function is recursive: it calls itself to map the coordinates of
		children of the current node 'ix'.

		Parameters
		----------
		ix : int
			The tree node to map.

		interval: length 2 tuple of floats
			Horizontal space allocated to node 'ix'.

		width : {'uniform', 'mass'}, optional
			Determines how much horzontal space each level set tree node is
			given. See LevelSetTree.plot() for more information.

		Returns
		-------
		segments : dict
			A dictionary with values that contain the coordinates of vertical
			line segment endpoints. This is only useful to the interactive
			analysis tools.

		segmap : list
			Indicates the order of the vertical line segments as returned by the
			recursive coordinate mapping function, so they can be picked by the
			user in the interactive tools.

		splits : dict
			Dictionary values contain the coordinates of horizontal line
			segments (i.e. node splits).

		splitmap : list
			Indicates the order of horizontal line segments returned by
			recursive coordinate mapping function, for use with interactive
			tools.
		"""

		## get children
		children = np.array(self.nodes[ix].children)
		n_child = len(children)


		## if there's no children, just one segment at the interval mean
		if n_child == 0:
			xpos = np.mean(interval)
			segments = {}
			segmap = [ix]
			splits = {}
			splitmap = []

			segments[ix] = (([xpos, self.nodes[ix].start_radius],
				[xpos, self.nodes[ix].end_radius]))


		## else, construct child branches then figure out parent's position
		else:
			parent_range = interval[1] - interval[0]
			segments = {}
			segmap = [ix]
			splits = {}
			splitmap = []

			census = np.array([len(self.nodes[x].members) for x in children],
				dtype=np.float)
			weights = census / sum(census)

			## sort branches by mass in decreasing order from left to right
			seniority = np.argsort(weights)[::-1]
			children = children[seniority]
			weights = weights[seniority]

			## get relative branch intervals
			if width == 'mass':
				child_intervals = np.cumsum(weights)
				child_intervals = np.insert(child_intervals, 0, 0.0)
			else:
				child_intervals = np.linspace(0.0, 1.0, n_child+1)

			## loop over the children
			for j, child in enumerate(children):

				## translate local interval to absolute interval
				branch_interval = (interval[0] + child_intervals[j] * parent_range,
					interval[0] + child_intervals[j+1] * parent_range)

				## recurse on the child
				branch = self.constructBranchMap(child, branch_interval, width)
				branch_segs, branch_splits, branch_segmap, branch_splitmap = branch

				segmap += branch_segmap
				splitmap += branch_splitmap
				splits = dict(splits.items() + branch_splits.items())
				segments = dict(segments.items() + branch_segs.items())


			## find the middle of the children's x-position and make vertical segment ix
			children_xpos = np.array([segments[k][0][0] for k in children])
			xpos = np.mean(children_xpos)


			## add horizontal segments to the list
			for child in children:
				splitmap.append(child)
				child_xpos = segments[child][0][0]

				splits[child] = ([xpos, self.nodes[ix].end_radius],
					[child_xpos, self.nodes[ix].end_radius])


			## add vertical segment for current node
			segments[ix] = (([xpos, self.nodes[ix].start_radius],
				[xpos, self.nodes[ix].end_radius]))

		return segments, splits, segmap, splitmap




#############################################
### LEVEL SET TREE CONSTRUCTION FUNCTIONS ###
#############################################

def cdTree(X, k, alpha=1.0, start='complete', verbose=False):
	"""
	Construct a Chaudhuri-Dasgupta level set tree. A level set tree is
	constructed by identifying connected components of observations as edges are
	removed from the geometric graph in descending order of pairwise distance.

	Parameters
	----------
	X : 2D array
		Data matrix, with observations as rows.

	k : integer
		Number of observations to consider as neighbors of each point.

	alpha : float
		A robustness parameter. Dilates the threshold for including edges in an
		upper level set similarity graph.

	start : {'complete', 'knn'}, optional
		Initialization of the similarity graph. 'Complete' starts with a
		complete similarity graph (as written in the Chaudhuri-Dasgupta paper)
		and knn starts with a k-nearest neighbor similarity graph.

	verbose: {False, True}, optional
		If set to True, then prints to the screen a progress indicator every 100
		levels.

	Returns
	-------
	T : levelSetTree
		See debacl.levelSetTree for class and method definitions.
	"""

	n, p = X.shape


	## Find the distance between each pair of points
	r_node = spd.pdist(X, metric='euclidean')
	D = spd.squareform(r_node)


	## Get the k-neighbor radius for each point
	rank = np.argsort(D, axis=1)
	ix_nbr = rank[:, 0:k]   # should this be k+1 to match Kpotufe paper?
	k_nbr = ix_nbr[:, -1]
	k_radius = D[np.arange(n), k_nbr]


	## Construct a complete graph
	G = igr.Graph.Full(n)
	G.vs['name'] = range(n)
	G.vs['radius'] = k_radius
	G.es['name'] = range(G.ecount())
	G.es['length'] = r_node


	## Set all relevant distances
	r_edge = r_node / alpha
	r_levels = np.unique(np.append(r_node, r_edge))[::-1]


	## Instantiate the tree
	T = CDTree()

	if start == 'complete':
		T.subgraphs[0] = G
		T.nodes[0] = ConnectedComponent(0, parent=None, children=[],
			start_radius=r_levels[0], end_radius=None, members=G.vs['name'])

	elif start == 'knn':
		max_radius = max(k_radius)

		# remove edges longer than the maximum k-neighbor radius
		cut_edges = G.es.select(length_gt = max_radius)
		if len(cut_edges) > 0:
			cut_edges.delete()

		# initialize a subgraph and node for each root component
		cc0 = G.components()
		for i, c in enumerate(cc0):
			T.subgraphs[i] = G.subgraph(c)
			T.nodes[i] = ConnectedComponent(i, parent=None, children=[],
				start_radius=max_radius, end_radius=None,
					members=G.vs[c]['name'])

	else:
		print "Start value not understood."
		return


	## Iterate through relevant threshold values in descending order
	for i, r in enumerate(r_levels):

		n_iter = len(r_levels)
		if i % 1000 == 0 and verbose:
			print "iteration:", i, "/", n_iter

		deactivate_keys = []
		activate_subgraphs = {}

		for (k, H) in T.subgraphs.items():

			# remove nodes and edges with large weight
			cut_nodes = H.vs.select(radius_ge = r)
			if len(cut_nodes) > 0:
				cut_nodes.delete()

			cut_edges = H.es.select(length_ge = alpha * r)  # note this alpha
			if len(cut_edges) > 0:
				cut_edges.delete()


			# check if component has vanishe
			if H.vcount() == 0:
				T.nodes[k].end_radius = r
				deactivate_keys.append(k)


			# if the graph has changed, look for splits
			if len(cut_edges) > 0 or len(cut_nodes) > 0:
					cc = H.components()

					if len(cc) > 1:
						T.nodes[k].end_radius = r
						deactivate_keys.append(k)

						for c in cc:
							new_key = max(T.nodes.keys()) + 1
							T.nodes[k].children.append(new_key)
							activate_subgraphs[new_key] = H.subgraph(c)
							T.nodes[new_key] = ConnectedComponent(new_key, parent=k,
									children=[], start_radius=r, end_radius=None,
									members=H.vs[c]['name'])

		# update active components
		for k in deactivate_keys:
			del T.subgraphs[k]

		T.subgraphs.update(activate_subgraphs)

	return T



def loadTree(fname):
	"""
	Load a saved tree from file.

	Parameters
	----------
	fname : string
		Filename to load. The .mat extension is not necessary.

	Returns
	-------
	T : LevelSetTree
		The loaded and reconstituted level set tree object.
	"""

	indata = spio.loadmat(fname)

	## format inputs
	idnums = indata['idnums'].flatten()
	levels = list(indata['levels'].flatten())
	bg_sets = [np.array(x[0].flatten()) for x in indata['bg_sets']]
	start_radii = indata['start_radii'].flatten()
	end_radii = indata['end_radii'].flatten()
	parents = [(None if x == -1 else x) for x in indata['parents'].flatten()]
	children = [list(x[0].flatten()) for x in indata['children']]
	members = [list(x[0].flatten()) for x in indata['members']]

	if len(children) == 0:
		children = [[]]*len(idnums)

	## create tree
	T = CD_Tree()

	## add nodes to the tree
	nodes = {}
	for i, k in enumerate(idnums):
		nodes[k] = ConnectedComponent(k, parents[i], children[i],
			start_radii[i], end_radii[i], members[i])

	T.nodes = nodes
	return T





