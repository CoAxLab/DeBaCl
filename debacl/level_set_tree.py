##############################################################
## Brian P. Kent
## level_set_tree.py
## Created: 20130625
## Updated: 20130625
## Defines the main level set tree objects in the DeBaCl package.
##############################################################

##############
### SET UP ###
##############
import numpy as np


#####################
### BASIC CLASSES ###
#####################

class LevelSetTree(object):

	def __init__(self, model):
		self.nodes = {}
		self.subgraphs = {}
		self.model = model

		
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
		
		T = LevelSetTree()
		T.nodes[ix] = self.nodes[ix].copy()
		T.nodes[ix].parent = None
		queue = self.nodes[ix].children[:]
	
		while len(queue) > 0:
			branch_ix = queue.pop()
			T.nodes[branch_ix] = self.nodes[branch_ix]
			queue += self.nodes[branch_ix].children
	
		return T
		
		
	def constructMassMap(self, ix, start_pile, interval, width_mode):
		"""
		Map level set tree nodes to locations in a plot canvas. Finds the plot
		coordinates of vertical line segments corresponding to LST nodes and
		horizontal line segments corresponding to node splits. Also provides
		indices of vertical segments and splits for downstream use with
		interactive plot picker tools. This function is not meant to be called
		by the user; it is a helper function for the LevelSetTree.plot() method.
		This function is recursive: it calls itself to map the coordinates of
		children of the current node 'ix'. Differs from 'constructBranchMap' by
		setting the height of each vertical segment to be proportional to the
		number of points in the corresponding LST node.
	
		Parameters
		----------
		ix : int
			The tree node to map.
		
		start_pile: float
			The height of the branch on the plot at it's start (i.e. lower
			terminus).
			
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
		
		size = float(len(self.nodes[ix].members))
	
		## get children
		children = np.array(self.nodes[ix].children)
		n_child = len(children)
	
		
		## if there's no children, just one segment at the interval mean
		if n_child == 0:
			xpos = np.mean(interval)
			end_pile = start_pile + size/self.n
			segments = {}
			segmap = [ix]
			splits = {}		
			splitmap = []
			segments[ix] = ([xpos, start_pile], [xpos, end_pile])
			
		else:
			parent_range = interval[1] - interval[0]
			segments = {}
			segmap = [ix]
			splits = {}
			splitmap = []
		
			census = np.array([len(self.nodes[x].members) for x in children],
				dtype=np.float)
			weights = census / sum(census)
		
			seniority = np.argsort(weights)[::-1]
			children = children[seniority]
			weights = weights[seniority]
			
			## get relative branch intervals
			if width_mode == 'mass':
				child_intervals = np.cumsum(weights)
				child_intervals = np.insert(child_intervals, 0, 0.0)
			else:
				child_intervals = np.linspace(0.0, 1.0, n_child+1)
			
			
			## find height of the branch
			end_pile = start_pile + (size - sum(census))/self.n
		
			## loop over the children
			for j, child in enumerate(children):
		
				## translate local interval to absolute interval
				branch_interval = (interval[0] + \
					child_intervals[j] * parent_range, interval[0] + \
					child_intervals[j+1] * parent_range)

				## recurse on the child
				branch = self.constructMassMap(child, end_pile, branch_interval,
					width_mode)
				branch_segs, branch_splits, branch_segmap, \
					branch_splitmap = branch
				
				segmap += branch_segmap
				splitmap += branch_splitmap
				splits = dict(splits.items() + branch_splits.items())
				segments = dict(segments.items() + branch_segs.items())
				
				
			## find the middle of the children's x-position and make vertical
			## segment ix
			children_xpos = np.array([segments[k][0][0] for k in children])
			xpos = np.mean(children_xpos)				


			## add horizontal segments to the list
			for child in children:
				splitmap.append(child)
				child_xpos = segments[child][0][0]
				splits[child] = ([xpos, end_pile],[child_xpos, end_pile])

	
			## add vertical segment for current node
			segments[ix] = ([xpos, start_pile], [xpos, end_pile])
			
			
		return segments, splits, segmap, splitmap
		
			
