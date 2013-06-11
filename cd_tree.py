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

import numpy as np
import pandas as pd
import scipy.spatial.distance as spdist
import scipy.io as spio
import igraph as igr
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection



#####################
### BASIC CLASSES ###
#####################
class CD_Component(object):
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
		
		component = CD_Component(self.idnum, self.parent, self.children,
			self.start_radius, self.end_radius,	self.members)
			
		return component
		
		
		
		
class CD_Tree(object):
	"""
	Defines methods and attributes for a Chaudhuri-Dasgupta level set tree.
	"""
	
	def __init__(self):
		self.nodes = {}
		self.subgraphs = {}
		
	
	def getSummary(self):
		"""
		Produce a tree summary table with Pandas. This can be printed to screen
		or saved easily to CSV. This is much simpler than manually formatting
		the strings in the CoAxLab version of DeBaCl, but it does require
		Pandas.
		
		Parameters
		----------
		
		Returns
		-------
		summary : pandas.DataFrame
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
		return summary
		
	
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
				
				
	def plot(self, width_mode='uniform', gap=0.05):
		"""
		Create a static plot of a Chaudhuri-Dasgupta level set tree.
		
		Parameters
		----------
		width_mode : {'uniform', 'mass'}, optional
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
		ix_root = np.array([k for k, v in self.nodes.iteritems() if v.parent is None])
		n_root = len(ix_root)
		census = np.array([len(self.nodes[x].members) for x in ix_root], dtype=np.float)
		n = sum(census)

		## Order the roots by mass decreasing from left to right		
		seniority = np.argsort(census)[::-1]
		ix_root = ix_root[seniority]
		census = census[seniority]
			
		if width_mode == 'mass':
			weights = census / n
			intervals = np.cumsum(weights)
			intervals = np.insert(intervals, 0, 0.0)
		else:
			intervals = np.linspace(0.0, 1.0, n_root+1)
		
		
		## Do a depth-first search on each root to get segments for each branch
		for i, ix in enumerate(ix_root):
			branch = self.constructBranchMap(ix, (intervals[i], intervals[i+1]),
				width_mode)
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
		radius_ticks = np.sort(list(set(
			[v.start_radius for v in self.nodes.itervalues()] + \
			[v.end_radius for v in self.nodes.itervalues()])))
		radius_tick_labels = [str(round(lvl, 2)) for lvl in radius_ticks]
		
	
		## Set up the plot framework
		fig, ax = plt.subplots()
		ax.set_position([0.11, 0.05, 0.78, 0.93])
		ax.set_xlabel("Connected component")
		ax.set_xlim((-0.04, 1.04))
		ax.set_xticks([])
		ax.set_xticklabels([])


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
		ax.yaxis.grid(color='gray')

		ax.set_yticks(radius_ticks)
		ax.set_yticklabels(radius_tick_labels)
				
		return fig
		
		
	def constructBranchMap(self, ix, interval, width_mode):
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
	
		width_mode : {'uniform', 'mass'}, optional
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
			if width_mode == 'mass':
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
				branch = self.constructBranchMap(child, branch_interval, width_mode)
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

def makeCDTree(X, k, start='max-edge', verbose=False):
	"""
	Construct a Chaudhuri-Dasgupta level set tree. A level set tree is
	constructed by identifying connected components of observations as edges are
	removed from the geometric graph in descending order of pairwise distance.
	
	Parameters
	----------
	X : 2D array
		Data matrix, with observations as rows.
	
	start : {'max-edge', 'max-radius'}, optional
		Should the tree start at the longest pairwise distance, or at the
		largest k-neighbor radius.
	
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
	d = spdist.pdist(X, metric='euclidean')
	D = spdist.squareform(d)
	
	
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
	G.es['length'] = d

	ix_order = np.argsort(d)[::-1]
	distances = d[ix_order]
		

	## Instantiate the tree	
	T = CD_Tree()
	
	if start == 'max-edge':
		T.subgraphs[0] = G
		T.nodes[0] = CD_Component(0, parent=None, children=[],
			start_radius=distances[0], end_radius=None, members=G.vs['name'])

	elif start == 'max-radius':
		max_radius = max(k_radius)
		
		# remove edges longer than the maximum k-neighbor radius
		cut_edges = G.es.select(length_gt = max_radius)
		if len(cut_edges) > 0:
			cut_edges.delete()

		# initialize a subgraph and node for each root component
		cc0 = G.components()
		for i, c in enumerate(cc0):
			T.subgraphs[i] = G.subgraph(c)
			T.nodes[i] = CD_Component(i, parent=None, children=[], 
				start_radius=max_radius, end_radius=None, members=G.vs[c]['name'])
				
		# trim the distances vector
		distances = distances[distances <= max_radius]
				
	else:
		print "Start value not understood."
		
		
	## Iterate through all pairwise distances in descending order
	for i, r in enumerate(distances):
		n_iter = len(distances)
		if i % 1000 == 0 and verbose:
			print "iteration:", i, "/", n_iter

		deactivate_keys = []
		activate_subgraphs = {}

		for (k, H) in T.subgraphs.items():
			cut_edges = H.es.select(length_ge = r)
			cut_nodes = H.vs.select(radius_ge = r)
		
			if len(cut_edges) > 0:
				cut_edges.delete()

			if len(cut_nodes) > 0:			
				cut_nodes.delete()
			
				# check if component has vanished - can only happen on node deletion
				if H.vcount() == 0:
					T.nodes[k].end_radius = r
					deactivate_keys.append(k)
		
			if len(cut_edges) > 0 or len(cut_nodes) > 0:	
					# check if component splits
					cc = H.components()	
					if len(cc) > 1:
						T.nodes[k].end_radius = r
						deactivate_keys.append(k)
					
						for c in cc:
							new_key = max(T.nodes.keys()) + 1
							T.nodes[k].children.append(new_key)
							activate_subgraphs[new_key] = H.subgraph(c)
							T.nodes[new_key] = CD_Component(new_key, parent=k,
									children=[], start_radius=r, end_radius=None,
									members=H.vs[c]['name'])
								
		# update active components
		for k in deactivate_keys:
			del T.subgraphs[k]
		
		T.subgraphs.update(activate_subgraphs)								
	
	return T
	


def makeSubtree(tree, ix):
	"""
	Return the subtree with node 'ix' as the root, and all ancestors of 'ix'.
	
	Parameters
	----------
	tree : LevelSetTree
	
	ix : int
		Node to use at the root of the new tree.
	
	Returns
	-------
	T : LevelSetTree
		A completely indpendent level set tree, with 'ix' as the root node.
	"""
		
	T = CD_Tree()
	T.nodes[ix] = tree.nodes[ix].copy()
	T.nodes[ix].parent = None
	queue = tree.nodes[ix].children[:]
	
	while len(queue) > 0:
		branch_ix = queue.pop()
		T.nodes[branch_ix] = tree.nodes[branch_ix]
		queue += tree.nodes[branch_ix].children
	
	return T

		
