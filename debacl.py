##############################################################
## Brian P. Kent
## debacl.py
## Created: 20120821
## Updated: 20130314
##############################################################

##############
### SET UP ###
##############
"""
Main functions and classes for the DEnsity-BAsed CLustering (DeBaCl) toolbox. Includes
functions to construct and modify with level set tree objects, and tools for interactive
data analysis and clustering with level set trees.
"""

import numpy as np
import pandas as pd
import scipy.spatial.distance as spdist
import scipy.io as spio
import igraph as igr
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.widgets import Button

import gen_utils as utl
import plot_utils as plutl




#####################
### BASIC CLASSES ###
#####################

class ConnectedComponent(object):
	"""
	Defines a connected component for level set tree construction. A level set tree is
	really just a set of ConnectedComponents.
	"""
	
	def __init__(self, idnum, parent, children, start_level, end_level, start_mass,
		end_mass, members):
		
		self.idnum = idnum
		self.parent = parent
		self.children = children
		self.start_level = start_level
		self.end_level = end_level
		self.start_mass = start_mass
		self.end_mass = end_mass
		self.members = members
		
		
	def copy(self):
		"""
		Creates and returns a copy of a ConnetedComponent object.
		
		Parameters
		----------
		
		Returns
		-------
		component : ConnectedComponent	

		"""
		
		component = ConnectedComponent(self.idnum, self.parent, self.children,
			self.start_level, self.end_level, self.start_mass, self.end_mass,
			self.members)
			
		return component
		


class LevelSetTree(object):
	"""
	Defines methods and attributes for a level set tree, i.e. a collection of connected
	components organized hierarchically.
	
	Parameters
	----------
	bg_sets : list of lists
		The observations removed as background points at each successively higher
		density level.
	
	levels : array_like
		The probability density level associated with each element in 'bg_sets'.
	"""
	
	def __init__(self, bg_sets, levels):
		self.bg_sets = bg_sets
		self.levels = levels
		self.n = sum([len(x) for x in bg_sets])
		self.nodes = {}
		self.subgraphs = {}
		
		
	def collapseLeaves(self, active_nodes):
		"""
		Removes descendent nodes for the branches in 'active_nodes'.
		
		Parameters
		----------
		active_nodes : array-like
			List of nodes to use as the leaves in the collapsed tree.
		
		Returns
		-------
		"""
		
		for ix in active_nodes:
			subtree = makeSubtree(self, ix)
			
			max_end_level = max([v.end_level for v in subtree.nodes.values()])
			max_end_mass = max([v.end_mass for v in subtree.nodes.values()])
			
			self.nodes[ix].end_level = max_end_level
			self.nodes[ix].end_mass = max_end_mass
			self.nodes[ix].children = []
			
			for u in subtree.nodes.keys():
				if u != ix:
					del self.nodes[u]

		
	def mergeBySize(self, threshold):
		"""
		Prune splits from a tree based on size of child nodes. Merge members of child
		nodes rather than removing them.
		
		Parameters
		----------
		threshold : numeric
			Tree branches with fewer members than this will be merged into larger
			siblings or parents.
		
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
				parent.end_level = max([self.nodes[k].end_level for k in parent.children])
				parent.end_mass = max([self.nodes[k].end_mass for k in parent.children])

				# remove small kids from the tree
				for k in parent.children:
					del self.nodes[k]
				parent.children = []
				
			elif n_bigkid == 1:
				pass
				# identify the big kid
				ix_bigkid = [k for k, v in kid_size.iteritems() if v >= threshold][0]
				bigkid = self.nodes[ix_bigkid]
					
				# update k's end level and end mass
				parent.end_level = bigkid.end_level
				parent.end_mass = bigkid.end_mass
				
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
							
		
		
	def pruneBySize(self, delta, mode='proportion'):
		"""
		Prune a tree by removing all nodes with too few members.
		
		Parameters
		----------
		delta : numeric
			Specifies the pruning threshold.
		mode : {'proportion', 'number'}, optional
			Indicates if 'delta' is a proportion of the total data set size, or an
			integer number.
			
		Notes
		-----
		Operates in-place on the levelSetTree.
		"""
		
		if mode == 'proportion':
			thresh = round(delta * self.n)
		else:
			thresh = delta
			
		self.nodes = {k: v for k, v in self.nodes.iteritems()
			if len(v.members) > thresh}
		
		## remove pointers to children that no longer exist
		for v in self.nodes.itervalues():
			v.children = [ix for ix in v.children if ix in self.nodes.keys()]
			
		## if a node has only one child, collapse that node back into the parent
		one_child_keys = [k for k, v in self.nodes.iteritems() if len(v.children) == 1]
		one_child_keys = np.sort(one_child_keys)[::-1]  # do it from the leaves to roots

		for k in one_child_keys:
			v = self.nodes[k]
			k_child = v.children[0]
			child = self.nodes[k_child]

			# update the parent
			v.end_level = child.end_level
			v.end_mass = child.end_mass
			v.children = child.children
			
			# update the grandchildren's parent
			for c in v.children:
				self.nodes[c].parent = k
			
			# remove the child node
			del self.nodes[k_child]
			
			
	def pruneRootsByLevel(self, thresh):
		"""
		Prune a tree by removing root nodes that end below a set level.
		
		This only tends to be useful for LSTs based on epsilon-neighborhood
		graphs. It is not smart - if a root node ends below the threshold, the node and
		its children are removed even if the children end above the threshold.
		
		Parameters
		----------
		thresh : float
			The threshold for keeping or cutting a root node.
		
		Returns
		-------
		"""
	
		remove_queue = [k for k, v in self.nodes.iteritems() if v.parent==None and
			v.end_level <= thresh]
		
		while len(remove_queue) > 0:
			k = remove_queue.pop()
			remove_queue += self.nodes[k].children
			del self.nodes[k]
	
		
	def getSummary(self):
		"""
		Produce a tree summary table with Pandas. This can be printed to screen or saved
		easily to CSV. This is much simpler than manually formatting the strings in the
		CoAxLab version of DeBaCl, but it does require Pandas.
		
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
				'lambda1': v.start_level,
				'lambda2': v.end_level,
				'alpha1': v.start_mass,
				'alpha2': v.end_mass,
				'size': len(v.members),
				'parent': v.parent,
				'children': v.children
				}
			row = pd.DataFrame([row])
			summary = summary.append(row)
		
		summary.set_index('key', inplace=True)
		return summary
			
				
	def printSummary(self):
		"""
		Prints a table of summary statistics for all of the connected components in a
		level set tree.
		
		Parameters
		----------
		
		Returns
		-------
		"""
		
		names = (
			'index', 'lambda1', 'lambda2', 'alpha1',
			'alpha2', 'size', 'parent', 'children'
			)
		widths = {
			'ix':5, 
			'lambda1': max([len(str(x.start_level).split('.')[0])
				for x in self.nodes.values()]) + 9,
			'lambda2': max([len(str(x.end_level).split('.')[0])
				for x in self.nodes.values()]) + 9,
			'alpha1': max([len(str(x.start_mass).split('.')[0])
				for x in self.nodes.values()]) + 9,
			'alpha2': max([len(str(x.end_mass).split('.')[0])
				for x in self.nodes.values()]) + 9,
			'size': max(4, max([len(str(len(x.members)))
				for x in self.nodes.values()])) + 2,
			'parent': max(6, max([len(str(x.parent))
				for x in self.nodes.values()])) + 2,
			'child': max(7, max([len(str(x.children))
				for x in self.nodes.values()])) + 2
			}
		
		head = '{0:<{ix}}{1:>{lambda1}}{2:>{lambda2}}{3:>{alpha1}}'.format(
			*names, **widths) + \
			'{4:>{alpha2}}{5:>{size}}{6:>{parent}}{7:>{child}}'.format(*names, **widths)
		print '\n', head
		
		for k, v in self.nodes.iteritems():
			line = '{0:<{ix}}{1.start_level:>{lambda1}.6f}{1.end_level:{lambda2}.6f}'.format(
					k, v, len(v.members), **widths) + \
				'{1.start_mass:>{alpha1}.6f}{1.end_mass:>{alpha2}.6f}'.format(
					k, v, len(v.members), **widths) + \
				'{2:>{size}}{1.parent:>{parent}}{1.children:>{child}}'.format(
					k, v, len(v.members), **widths) 			
			print line
			
			
	def save(self, fname):
		"""
		Save a level set tree object to file.
		
		Saves a level set tree as a MATLAB struct using the scipy.io module. Ignore the
		warning about using oned_as default value ('column').
		
		Parameters
		----------
		fname : string
			File to save the tree to. The .mat extension is not necessary.
			
		Returns
		-------
		"""		

		tree_dict = {
			'bg_sets': self.bg_sets,
			'levels': self.levels,
			'idnums': [x.idnum for x in self.nodes.values()],
			'start_levels': [x.start_level for x in self.nodes.values()],
			'end_levels': [x.end_level for x in self.nodes.values()],
			'start_mass': [x.start_mass for x in self.nodes.values()],
			'end_mass': [x.end_mass for x in self.nodes.values()],
			'parents': [(-1 if x.parent is None else x.parent)
				for x in self.nodes.values()],
			'children': [x.children for x in self.nodes.values()],
			'members': [x.members for x in self.nodes.values()]
			}

		spio.savemat(fname, tree_dict)
		
		
	def plot(self, height_mode='mass', width_mode='uniform', xpos='middle', sort=True,
		gap=0.05, color=False, color_nodes=None):
		"""
		Create a static plot of a level set tree.
		
		Parameters
		----------
		height_mode : {'mass', 'levels'}, optional
			Determines if the dominant vertical axis is based on density level values or
			mass (i.e. probability content) values.
		
		width_mode : {'uniform', 'mass'}, optional
			Determines how much horzontal space each level set tree node is given. The
			default of "uniform" gives each child node an equal fraction of the parent
			node's horizontal space. If set to 'mass', then horizontal space is
			allocated proportional to the mass (i.e. fraction of points) of a node
			relative to its siblings.
		
		xpos : {'middle'}, optional
			Where the vertical line segments are located within their allocated
			horizontal space. The default (and only implement option) is for them to lie
			in the middle of their horizontal intervals.
		
		sort : bool
			If True, sort sibling nodes from most to least points and draw left to
			right. Also sorts root nodes in the same way.
			
		gap : float
			Fraction of vertical space to leave at the bottom. Default is 5%, and 0%
			also works well. Higher values are used for interactive tools to make room
			for buttons and messages.
			
		color : bool
			Indicates if tree nodes should be colored. If True and color_nodes is None,
			then nodes will be colored at the first split (including level 0 if there
			are multiple root nodes).
		
		color_nodes : list
			Each entry should be a valid index in the level set tree that will be
			colored uniquely.
		
		Returns
		-------
		fig : matplotlib figure
			Use fig.show() to view, fig.savefig() to save, etc.
			
		segments : dict
			A dictionary with values that contain the coordinates of vertical line
			segment endpoints. This is only useful to the interactive analysis tools.
		
		segmap : list
			Indicates the order of the vertical line segments as returned by the
			recursive coordinate mapping function, so they can be picked by the user in
			the interactive tools.
		
		splits : dict
			Dictionary values contain the coordinates of horizontal line segments (i.e.
			node splits).
			
		splitmap : list
			Indicates the order of horizontal line segments returned by recursive
			coordinate mapping function, for use with interactive tools.
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
		
		if sort is True:
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
			branch = constructBranchMap(self, ix, (intervals[i], intervals[i+1]),
					height_mode, width_mode, xpos, sort)
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
		level_ticks = np.sort(list(set(
			[v.start_level for v in self.nodes.itervalues()] + \
			[v.end_level for v in self.nodes.itervalues()])))
		level_tick_labels = [str(round(lvl, 2)) for lvl in level_ticks]
		
		mass_ticks = np.sort(list(set(
			[v.start_mass for v in self.nodes.itervalues()] + \
			[v.end_mass for v in self.nodes.itervalues()])))
		mass_tick_labels = [str(round(m, 2)) for m in mass_ticks]

	
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
			
		if color is True:
			palette = plutl.Palette(use='scatter')

			if color_nodes is None:
				if len(ix_root) > 1:
					active_nodes = ix_root
				else:
					active_nodes = self.nodes[ix_root[0]].children
			else:
				active_nodes = color_nodes
				
			if len(active_nodes) <= np.alen(palette.colorset):
				for i, ix in enumerate(active_nodes):
					c = palette.colorset[i, :]
					subtree = makeSubtree(self, ix)

					## set verical colors
					ix_replace = np.in1d(segmap, subtree.nodes.keys())
					segclr[ix_replace] = c

					## set horizontal colors
					ix_replace = np.in1d(splitmap, subtree.nodes.keys())
					splitclr[ix_replace] = c
					
			
		linecol = LineCollection(verts, linewidths=thickness, colors=segclr)
		ax.add_collection(linecol)
		linecol.set_picker(20)
		
		splitcol = LineCollection(lats, colors=splitclr)
		ax.add_collection(splitcol)


		## Make the plot
		if height_mode=='levels':
			ax.set_ylabel("Level")
			ymin = min([v.start_level for v in self.nodes.itervalues()])
			ymax = max([v.end_level for v in self.nodes.itervalues()])
			rng = ymax - ymin
			ax.set_ylim(ymin - gap*rng, ymax + 0.05*rng)
			ax.yaxis.grid(color='gray')

			ax.set_yticks(level_ticks)
			ax.set_yticklabels(level_tick_labels)
			
			ax2 = ax.twinx()
			ax2.set_xticks([])
			ax2.set_xticklabels([])
			ax2.set_ylabel("Mass", rotation=270)
			
			ax2.set_ylim(ax.get_ylim())
			ax2.set_yticks(level_ticks)
			ax2.set_yticklabels(mass_tick_labels)

		else:
			ax.set_ylabel("Mass", rotation=270)
			ax.yaxis.set_label_position('right')

			ymin = min([v.start_mass for v in self.nodes.itervalues()])
			ymax = max([v.end_mass for v in self.nodes.itervalues()])
			rng = ymax - ymin
			ax.set_ylim(ymin - gap*rng, ymax + 0.05*ymax)
			ax.yaxis.grid(color='gray')

			ax2 = ax.twinx()
			ax2.set_xticks([])
			ax2.set_xticklabels([])
			
			ax2.yaxis.set_label_position('left')
			ax2.set_ylabel("Level")

			ax.set_yticks(mass_ticks)
			ax.set_yticklabels(mass_tick_labels)
			ax.yaxis.tick_right()

			ax2.set_ylim(ax.get_ylim())
			ax2.set_yticks(mass_ticks)
			ax2.set_yticklabels(level_tick_labels)
			ax2.yaxis.tick_left()
				
		return fig, segments, segmap, splits, splitmap
		
		
	def allModeCluster(self):
		"""
		Set every leaf node as a foreground cluster.

		Parameters
		----------
		
		Returns
		-------
		labels : 2D numpy array
			Each row corresponds to a foreground data point. The first column contains
			the index of the point in the original data set, and the second column
			contains the cluster assignment. Cluster labels are increasing integers
			starting at 0.
		"""
	
		leaves = [k for k, v in self.nodes.items() if v.children == []]
		
		## find components in the leaves
		points = []
		cluster = []

		for i, k in enumerate(leaves):
			points.extend(self.nodes[k].members)
			cluster += ([i] * len(self.nodes[k].members))

		labels = np.array([points, cluster], dtype=np.int).T		
		return labels
		
		
	def firstKCluster(self, k):
		"""
		Returns foreground cluster labels for the 'k' modes with the lowest start
		levels. In principle, this is the 'k' leaf nodes with the smallest indices, but
		double check this by finding all leaf start values and ordering.
		
		Parameters
		----------
		k : integer
			The desired number of clusters.
		
		Returns
		-------
		labels : 2D numpy array
			Each row corresponds to a foreground data point. The first column contains
			the index of the point in the original data set, and the second column
			contains the cluster assignment. Cluster labels are increasing integers
			starting at 0.
		
		active_nodes : list
			Indices of tree nodes that are foreground clusters. Particularly useful for
			coloring a level set tree plot to match the data scatterplot.
		"""
		
		parents = np.array([u for u, v in self.nodes.items() if len(v.children) > 0])
		roots = [u for u, v in self.nodes.items() if v.parent is None]
		splits = [self.nodes[u].end_level for u in parents]
		order = np.argsort(splits)
		star_parents = parents[order[:(k-len(roots))]]
		
		children = [u for u, v in self.nodes.items() if v.parent is None]
		for u in star_parents:
			children += self.nodes[u].children

		active_nodes = [x for x in children if
			sum(np.in1d(self.nodes[x].children, children))==0] 
		
		
		points = []
		cluster = []
		
		for i, c in enumerate(active_nodes):
			cluster_pts = self.nodes[c].members
			points.extend(cluster_pts)
			cluster += ([i] * len(cluster_pts))

		labels = np.array([points, cluster], dtype=np.int).T
		return labels, active_nodes
		
		
		
	def upperSetCluster(self, cut, mode):
		"""
		Set foreground clusters by finding connected components at an upper level set or
		upper mass set.
		
		Parameters
		----------
		cut : float
			The level or mass value that defines the foreground set of points, depending
			on 'mode'.
		
		mode : {'level', 'mass'}
			Determines if the 'cut' threshold is a density level value or a mass value
			(i.e. fraction of data in the background set)
		
		Returns
		-------
		labels : 2D numpy array
			Each row corresponds to a foreground data point. The first column contains
			the index of the point in the original data set, and the second column
			contains the cluster assignment. Cluster labels are increasing integers
			starting at 0.
		
		active_nodes : list
			Indices of tree nodes that are foreground clusters. Particularly useful for
			coloring a level set tree plot to match the data scatterplot.
		"""
	
		## identify upper level points and the nodes active at the cut
		if mode == 'mass':
			n_bg = [len(x) for x in self.bg_sets]
			alphas = np.cumsum(n_bg) / (1.0 * self.n)
			upper_levels = np.where(alphas > cut)[0]
			active_nodes = [k for k, v in self.nodes.iteritems()
				if v.start_mass <= cut and v.end_mass > cut]

		else:
			upper_levels = np.where(np.array(self.levels) > cut)[0]
			active_nodes = [k for k, v in self.nodes.iteritems()
				if v.start_level <= cut and v.end_level > cut]

		upper_pts = np.array([y for x in upper_levels for y in self.bg_sets[x]])
		

		## find intersection between upper set points and each active component
		points = []
		cluster = []

		for i, c in enumerate(active_nodes):
			cluster_pts = upper_pts[np.in1d(upper_pts, self.nodes[c].members)]
			points.extend(cluster_pts)
			cluster += ([i] * len(cluster_pts))

		labels = np.array([points, cluster], dtype=np.int).T		
		return labels, active_nodes
		

	def firstKLevelCluster(self, k):
		"""
		Use the first K clusters to appear in the level set tree as foreground clusters.
		
		In general, K-1 clusters will appear at a lower level than the K'th cluster;
		this function returns all members from all K clusters (rather than only the
		members in the upper level set where the K'th cluster appears). There are not
		always K clusters available in a level set tree - see LevelSetTree.findKCut for
		details on default behavior in this case.
		
		Parameters
		----------
		k : int
			Desired number of clusters.
		
		Returns
		-------
		labels : 2D numpy array
			Each row corresponds to a foreground data point. The first column contains
			the index of the point in the original data set, and the second column
			contains the cluster assignment. Cluster labels are increasing integers
			starting at 0.
		
		active_nodes : list
			Indices of tree nodes that are foreground clusters. Particularly useful for
			coloring a level set tree plot to match the data scatterplot.
		"""
		
		cut = self.findKCut(k)
		active_nodes = [e for e, v in self.nodes.iteritems() \
			if v.start_level <= cut and v.end_level > cut]
			
		points = []
		cluster = []
		
		for i, c in enumerate(active_nodes):
		
			cluster_pts = self.nodes[c].members
			points.extend(cluster_pts)
			cluster += ([i] * len(cluster_pts))

		labels = np.array([points, cluster], dtype=np.int).T
		return labels, active_nodes


	def findKCut(self, k):
		"""
		Find the lowest level cut that has k connected components.
		
		If there are no levels that have k components, then find the lowest level that
		has at least k components. If no levels have > k components, find the lowest
		level that has the maximum number of components.
		
		Parameters
		----------
		k : int
			Desired number of clusters/nodes/components.
			
		Returns
		-------
		cut : float
			Lowest density level where there are k nodes.
		"""
		
		## Find the lowest level to cut at that has k or more clusters
		starts = [v.start_level for v in self.nodes.itervalues()]
		ends = [v.end_level for v in self.nodes.itervalues()]
		crits = np.unique(starts + ends)
		nclust = {}

		for c in crits:
			nclust[c] = len([e for e, v in self.nodes.iteritems() \
				if v.start_level <= c and v.end_level > c])
			
		width = np.max(nclust.values())
	
		if k in nclust.values():
			cut = np.min([e for e, v in nclust.iteritems() if v == k])
		else:
			if width < k:
				cut = np.min([e for e, v in nclust.iteritems() if v == width])
			else:
				ktemp = np.min([v for v in nclust.itervalues() if v > k])
				cut = np.min([e for e, v in nclust.iteritems() if v == ktemp])
				
		return cut
		
		
	def constructBranchMap(self, ix, interval, height_mode, width_mode, xpos, sort):
		"""
		Map level set tree nodes to locations in a plot canvas.
	
		Finds the plot coordinates of vertical line segments corresponding to LST nodes and
		horizontal line segments corresponding to node splits. Also provides indices of
		vertical segments and splits for downstream use with interactive plot picker tools.
		This function is not meant to be called by the user; it is a helper function for the
		LevelSetTree.plot() method. This function is recursive: it calls itself to map the
		coordinates of children of the current node 'ix'.
	
		Parameters
		----------
		ix : int
			The tree node to map.
		
		interval: length 2 tuple of floats
			Horizontal space allocated to node 'ix'.
	
		height_mode : {'mass', 'levels'}, optional
	
		width_mode : {'uniform', 'mass'}, optional
			Determines how much horzontal space each level set tree node is given. See
			LevelSetTree.plot() for more information.
	
		xpos : {middle'}, optional
	
		sort : bool
			If True, sort sibling nodes from most to least points and draw left to right.
			Also sorts root nodes in the same way.
	
		Returns
		-------
		segments : dict
			A dictionary with values that contain the coordinates of vertical line
			segment endpoints. This is only useful to the interactive analysis tools.
	
		segmap : list
			Indicates the order of the vertical line segments as returned by the
			recursive coordinate mapping function, so they can be picked by the user in
			the interactive tools.
	
		splits : dict
			Dictionary values contain the coordinates of horizontal line segments (i.e.
			node splits).
		
		splitmap : list
			Indicates the order of horizontal line segments returned by recursive
			coordinate mapping function, for use with interactive tools.
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

			if height_mode == 'levels':
				segments[ix] = (([xpos, self.nodes[ix].start_level],
					[xpos, self.nodes[ix].end_level]))
			else:
				segments[ix] = (([xpos, self.nodes[ix].start_mass],
					[xpos, self.nodes[ix].end_mass]))

		
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
		
			if sort is True:
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
				branch = constructBranchMap(self, child, branch_interval, height_mode,
					width_mode, xpos, sort)
				branch_segs, branch_splits, branch_segmap, branch_splitmap = branch
				
				segmap += branch_segmap
				splitmap += branch_splitmap
				splits = dict(splits.items() + branch_splits.items())
				segments = dict(segments.items() + branch_segs.items())
			
			
			## find the middle of the children's x-position and make vertical segment ix
			children_xpos = np.array([segments[k][0][0] for k in children])
		
			if xpos == 'middle':
				xpos = np.mean(children_xpos)
			else:
				xpos_order = np.argsort(children_xpos)
				ordered_xpos = children_xpos[xpos_order]
				ordered_weights = weights[xpos_order]
				xpos = sum([pos*w for pos, w in zip(ordered_xpos, ordered_weights[::-1])])
		
		
			## add horizontal segments to the list
			for child in children:
				splitmap.append(child)
				child_xpos = segments[child][0][0]
		
				if height_mode == 'levels':
					splits[child] = ([xpos, self.nodes[ix].end_level],
						[child_xpos, self.nodes[ix].end_level])
				else:
					splits[child] = ([xpos, self.nodes[ix].end_mass],
						[child_xpos, self.nodes[ix].end_mass])
				
	
			## add vertical segment for current node
			if height_mode == 'levels':
				segments[ix] = (([xpos, self.nodes[ix].start_level],
					[xpos, self.nodes[ix].end_level]))
			else:
				segments[ix] = (([xpos, self.nodes[ix].start_mass],
					[xpos, self.nodes[ix].end_mass]))
	
		return segments, splits, segmap, splitmap


	
	def massToLevel(self, alpha):
		"""
		Convert the specified mass value into a level location.
		
		Parameters 
		----------
		alpha : float
			Float in the interval [0.0, 1.0], with the desired fraction of all points.
			
		Returns
		-------
		cut_level: float
			Density level corresponding to the 'alpha' fraction of background points.
		"""
		
		n_bg = [len(bg_set) for bg_set in self.bg_sets]
		masses = np.cumsum(n_bg) / (1.0 * self.n)
		cut_level = self.levels[np.where(masses > alpha)[0][0]]
		
		return cut_level



	
#############################################
### LEVEL SET TREE CONSTRUCTION FUNCTIONS ###
#############################################

def makeLevelSetTree(W, levels, bg_sets, mode='general', verbose=False):
	"""
	Construct a level set tree.
	
	A level set tree is constructed by identifying connected components of observations
	at successively higher levels of a probability density estimate.
	
	Parameters
	----------
	W : 2D array
		An adjacency matrix for a similarity graph on the data.
			
	levels: array
		Defines the density levels where connected components will be computed.
		Typically this includes all unique values of a function evaluated on the data
		points.
	
	bg_sets: list of lists
		Specify which points to remove as background at each density level in 'levels'.
	
	mode: {'general', 'density'}, optional
		Establish if the values in 'levels' come from a probability density or
		pseudo-density estimate, in which case there is a natural floor at 0. The
		default is to model an arbitrary function, which requires a run-time choice of
		floor value.
	
	verbose: {False, True}, optional
		If set to True, then prints to the screen a progress indicator every 100 levels.
	
	Returns
	-------
	T : levelSetTree
		See debacl.levelSetTree for class and method definitions.
	"""
	
	n = np.alen(W)
	levels = [float(x) for x in levels]
	
	## Initialize the graph and cluster tree
	G = igr.Graph.Adjacency(W.tolist(), mode=igr.ADJ_MAX)
	G.vs['index'] = range(n)

	T = LevelSetTree(bg_sets, levels)
	cc0 = G.components()
	
	if mode == 'density':
		start_level = 0.0
	else:
		start_level = float(np.min(levels) - 1.0)
		
	for i, c in enumerate(cc0):
		T.subgraphs[i] = G.subgraph(c)
		T.nodes[i] = ConnectedComponent(i, parent=None, children=[],
			start_level=start_level, end_level=None, start_mass=0.0, end_mass=None,
			members=G.vs[c]['index'])
	
	
	# Loop through the removal grid
	for i, (level, bg) in enumerate(zip(levels, bg_sets)):
	
		if verbose and i % 100 == 0:
			print "iteration", i
		
		# compute mass and level
		mass = 1.0 - (sum([x.vcount() for
			x in T.subgraphs.itervalues()]) - len(bg)) / (1.0 * n)

		# loop through active components, i.e. subgraphs
		deactivate_keys = []
		activate_subgraphs = {}
		
		for (k, H) in T.subgraphs.iteritems():
			cutpoints = H.vs.select(index_in = bg)
			
			if len(cutpoints) > 0:
			
				# remove the cutpoints
				maxdeg = cutpoints.maxdegree()
				cutpoints.delete()
		
				# check if component has vanished
				if H.vcount() == 0:
					T.nodes[k].end_level = level
					T.nodes[k].end_mass = mass
					deactivate_keys.append(k)
					
				else:
					cc = H.components()
					
					# check if component splits
					if len(cc) > 1:
						
						# retire the parent component			
						deactivate_keys.append(k)
						T.nodes[k].end_level = level
						T.nodes[k].end_mass = mass
						
						# start a new component for each child
						for c in cc:
							new_key = max(T.nodes.keys()) + 1
							T.nodes[k].children.append(new_key)
							activate_subgraphs[new_key] = H.subgraph(c)
							T.nodes[new_key] = ConnectedComponent(new_key, parent=k,
								children=[], start_level=level, end_level=None,
								start_mass=mass, end_mass=None,
								members=H.vs[c]['index'])
											
		# update active components
		for k in deactivate_keys:
			del T.subgraphs[k]
			
		T.subgraphs.update(activate_subgraphs)
		
	return T
	


def constructBranchMap(T, ix, interval, height_mode, width_mode, xpos, sort):
	"""
	Map level set tree nodes to locations in a plot canvas.
	
	Finds the plot coordinates of vertical line segments corresponding to LST nodes and
	horizontal line segments corresponding to node splits. Also provides indices of
	vertical segments and splits for downstream use with interactive plot picker tools.
	This function is not meant to be called by the user; it is a helper function for the
	LevelSetTree.plot() method. This function is recursive: it calls itself to map the
	coordinates of children of the current node 'ix'.
	
	Parameters
	----------
	T : LevelSetTree
		The original level set tree being plotted
		
	ix : int
		The tree node to map.
		
	interval: length 2 tuple of floats
		Horizontal space allocated to node 'ix'.
	
	height_mode : {'mass', 'levels'}, optional
	
	width_mode : {'uniform', 'mass'}, optional
		Determines how much horzontal space each level set tree node is given. See
		LevelSetTree.plot() for more information.
	
	xpos : {middle'}, optional
	
	sort : bool
		If True, sort sibling nodes from most to least points and draw left to right.
		Also sorts root nodes in the same way.
	
	Returns
	-------
	segments : dict
		A dictionary with values that contain the coordinates of vertical line
		segment endpoints. This is only useful to the interactive analysis tools.
	
	segmap : list
		Indicates the order of the vertical line segments as returned by the
		recursive coordinate mapping function, so they can be picked by the user in
		the interactive tools.
	
	splits : dict
		Dictionary values contain the coordinates of horizontal line segments (i.e.
		node splits).
		
	splitmap : list
		Indicates the order of horizontal line segments returned by recursive
		coordinate mapping function, for use with interactive tools.
	"""

	## get children
	children = np.array(T.nodes[ix].children)
	n_child = len(children)
	
	
	## if there's no children, just one segment at the interval mean
	if n_child == 0:
		xpos = np.mean(interval)
		segments = {}
		segmap = [ix]
		splits = {}
		splitmap = []

		if height_mode == 'levels':
			segments[ix] = (([xpos, T.nodes[ix].start_level],
				[xpos, T.nodes[ix].end_level]))
		else:
			segments[ix] = (([xpos, T.nodes[ix].start_mass],
				[xpos, T.nodes[ix].end_mass]))

		
	## else, construct child branches then figure out parent's position				
	else:
		parent_range = interval[1] - interval[0]
		segments = {}
		segmap = [ix]
		splits = {}
		splitmap = []
		
		census = np.array([len(T.nodes[x].members) for x in children], dtype=np.float)
		weights = census / sum(census)
		
		if sort is True:
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
			branch = constructBranchMap(T, child, branch_interval, height_mode,
				width_mode, xpos, sort)
			branch_segs, branch_splits, branch_segmap, branch_splitmap = branch
				
			segmap += branch_segmap
			splitmap += branch_splitmap
			splits = dict(splits.items() + branch_splits.items())
			segments = dict(segments.items() + branch_segs.items())
			
			
		## find the middle of the children's x-position and make vertical segment ix
		children_xpos = np.array([segments[k][0][0] for k in children])
		
		if xpos == 'middle':
			xpos = np.mean(children_xpos)
		else:
			xpos_order = np.argsort(children_xpos)
			ordered_xpos = children_xpos[xpos_order]
			ordered_weights = weights[xpos_order]
			xpos = sum([pos*w for pos, w in zip(ordered_xpos, ordered_weights[::-1])])
		
		
		## add horizontal segments to the list
		for child in children:
			splitmap.append(child)
			child_xpos = segments[child][0][0]
		
			if height_mode == 'levels':
				splits[child] = ([xpos, T.nodes[ix].end_level],
					[child_xpos, T.nodes[ix].end_level])
			else:
				splits[child] = ([xpos, T.nodes[ix].end_mass],
					[child_xpos, T.nodes[ix].end_mass])
				
	
		## add vertical segment for current node
		if height_mode == 'levels':
			segments[ix] = (([xpos, T.nodes[ix].start_level],
				[xpos, T.nodes[ix].end_level]))
		else:
			segments[ix] = (([xpos, T.nodes[ix].start_mass],
				[xpos, T.nodes[ix].end_mass]))
	
	return segments, splits, segmap, splitmap
	
	
	
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
		
	T = LevelSetTree(bg_sets=[], levels=[])
	T.nodes[ix] = tree.nodes[ix].copy()
	T.nodes[ix].parent = None
	queue = tree.nodes[ix].children[:]
	
	while len(queue) > 0:
		branch_ix = queue.pop()
		T.nodes[branch_ix] = tree.nodes[branch_ix]
		queue += tree.nodes[branch_ix].children
	
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
	start_levels = indata['start_levels'].flatten()
	end_levels = indata['end_levels'].flatten()
	start_mass = indata['start_mass'].flatten()
	end_mass = indata['end_mass'].flatten()
	parents = [(None if x == -1 else x) for x in indata['parents'].flatten()]
	children = [list(x[0].flatten()) for x in indata['children']]
	members = [list(x[0].flatten()) for x in indata['members']]
	
	if len(children) == 0:	
		children = [[]]*len(idnums)
	
	## create tree
	T = LevelSetTree(bg_sets, levels)
	
	## add nodes to the tree
	nodes = {}
	for i, k in enumerate(idnums):
		nodes[k] = ConnectedComponent(k, parents[i], children[i], start_levels[i],
			end_levels[i], start_mass[i], end_mass[i], members[i])
		
	T.nodes = nodes
	return T




##################################################
### INTERACTIVE LEVEL SET TREE ANALYLSIS TOOLS ###
##################################################

class TreeComponentTool(object):
	"""
	Allow the user to interactively select level set tree nodes for tree coloring,
	subsetting, and scatter plots.
	
	Parameters
	----------
	tree : LevelSetTree
	
	pts : 2D numpy array
		Original data matrix. Rows are observations. Must have 3 or fewer columns if the
		'output' list includes 'scatter'.
	
	height_mode : {'mass', 'levels'}, optional
		Passed to LevelSetTree.plot(). Determines if the dominant vertical axis is
		based on density level values or mass (i.e. probability content) values.
	
	width_mode : {'uniform', 'mass'}, optional
		Passed to LevelSetTree.plot(). Determines how much horzontal space each
		level set tree node is given.
	
	output : list of strings, optional
		If the list includes 'tree', selecting a LST node will plot the subtree with
		the selected node as the root. If output includes 'scatter' and the data has
		3 or fewer dimensions, selecting a tree node produces a scatter plot showing
		the members of the selected node.
	
	s : int, optional
		If 'output' includes 'scatter', the size of the points in the scatterplot.
		
	f : 2D numpy array, optional
		Any function. Arguments in the first column and values in the second. Plotted
		independently of the data as a blue curve, so does not need to have the same
		number of rows as values in 'x'. Typically this is the generating probability
		density function for a 1D simulation.
	
	fhat : list of floats, optional
		Density estimate values for the data in 'pts'. Plotted as a black curve, with
		points colored according to the selected component.
	"""

	def __init__(self, tree, pts, height_mode='mass', width_mode='uniform',
		output=['tree', 'scatter'], s=20, f=None, fhat=None):

		self.T = tree
		self.X = pts
		self.height_mode = height_mode
		self.width_mode = width_mode
		self.output = output
		self.size = s
		self.f = f
		self.fhat = fhat
		self.fig, self.segments, self.segmap, self.splits, self.splitmap = self.T.plot(
			height_mode, width_mode)
		
		self.ax = self.fig.axes[0]
		self.ax.set_zorder(0.1)  # sets the first axes to have priority for picking
		segments = self.ax.collections[0]  # the line collection
		segments.set_picker(15)
		self.fig.canvas.mpl_connect('pick_event', self.handle_pick)

		
	def handle_pick(self, event):
		"""
		Return the members of the component that is picked out.
		"""
	
		## get the component members and subtree
		ix_seg = event.ind[0]
		self.node_ix = self.segmap[ix_seg]
		self.component = self.T.nodes[self.node_ix].members
		self.subtree = makeSubtree(self.T, self.node_ix)
							
		## recolor the original tree
		palette = plutl.Palette(use='scatter')
		segclr = np.array([[0.0, 0.0, 0.0]] * len(self.segmap))
		splitclr = np.array([[0.0, 0.0, 0.0]] * len(self.splitmap))

		# set new segment colors
		ix_replace = np.in1d(self.segmap, self.subtree.nodes.keys())
		segclr[ix_replace] = palette.colorset[0, :]
		
		ix_replace = np.in1d(self.splitmap, self.subtree.nodes.keys())
		splitclr[ix_replace] = palette.colorset[0, :]

		self.ax.collections[0].set_color(segclr)
		self.ax.collections[1].set_color(splitclr)
		self.fig.canvas.draw()

			
		# plot the component points in a new window
		if 'scatter' in self.output:
	
			# determine the data dimension
			if len(self.X.shape) == 1:
				n = len(self.X)
				p = 1
			else:
				n, p = self.X.shape
			
			if p == 1:
				uc = np.vstack((self.component, np.zeros((len(self.component),),
					dtype=np.int))).T
				hist_fig = plutl.clusterHistogram(self.X, uc, self.fhat, self.f)
				hist_fig.show()
			
			elif p == 2 or p == 3:
				base_clr = [217.0 / 255.0] * 3 # light gray
				comp_clr = [228/255.0, 26/255.0, 28/255.0] # red
				black = [0.0, 0.0, 0.0]

				clr_matrix = plutl.makeColorMatrix(n, bg_color=base_clr, bg_alpha=0.72,
					ix=list(self.component), fg_color=comp_clr, fg_alpha=0.68)
				edge_matrix = plutl.makeColorMatrix(n, bg_color=black, bg_alpha=0.38,
					ix=None)  # edges are black
				pts_fig = plutl.plotPoints(self.X, size=self.size, clr=clr_matrix,
					edgecolor=edge_matrix)
				pts_fig.show()

			else:
				print "Sorry, your data has too many dimensions to plot."
			
		
		# construct the new subtree and show it
		if 'tree' in self.output:
			subfig = self.subtree.plot(height_mode=self.height_mode,
				width_mode=self.width_mode)[0]
			subfig.show()
			
			
	def show(self):
		"""
		Show the instantiated TreeComponentTool (i.e. the interactive LevelSetTree
		plot).
		"""
		self.fig.show()
		
		
	def getComponent(self):
		"""
		Return the members of the currently selected level set tree node.
		"""
		return self.component


	def getSubtree(self):
		"""
		Return the subtree with the currently selected node as root.
		"""
		return self.subtree
		
		
	def getIndex(self):
		"""
		Return the index of the currently selected tree node.
		"""
		return self.node_ix
		
	


class TreeClusterTool(object):
	"""
	Allow the user to interactively select level or mass values in a level set tree and
	to see the clusters at that level.
	
	Parameters
	----------
	tree : LevelSetTree
	
	pts : 2D numpy array
		Original data matrix. Rows are observations.
	
	height_mode : {'mass', 'levels'}, optional
		Passed to LevelSetTree.plot(). Determines if the dominant vertical axis is
		based on density level values or mass (i.e. probability content) values.
	
	width_mode : {'uniform', 'mass'}, optional
		Passed to LevelSetTree.plot(). Determines how much horzontal space each
		level set tree node is given.
	
	output : list of strings, optional
		If output includes 'scatter' and the data has 3 or fewer dimensions, selecting a
		tree node produces a scatter plot showing the members of the selected node.
	
	s : int, optional
		If 'output' includes 'scatter', the size of the points in the scatterplot.
		
	f : 2D numpy array, optional
		Any function. Arguments in the first column and values in the second. Plotted
		independently of the data as a blue curve, so does not need to have the same
		number of rows as values in 'x'. Typically this is the generating probability
		density function for a 1D simulation.
	
	fhat : list of floats, optional
		Density estimate values for the data in 'pts'. Plotted as a black curve, with
		points colored according to the selected component.
	"""

	def __init__(self, tree, pts, height_mode='mass', width_mode='uniform',
		output=['scatter'], s=20, f=None, fhat=None):
		
		self.T = tree
		self.X = pts
		self.height_mode = height_mode
		self.width_mode = width_mode
		self.output = output
		self.size = s		
		self.f = f
		self.fhat = fhat
		self.clusters = None
		
		self.fig, self.segments, self.segmap, self.splits, self.splitmap = self.T.plot(
			height_mode, width_mode)
		self.ax = self.fig.axes[0]
		self.ax.set_zorder(0.1)  # sets the first axes to have priority for picking
		self.line = self.ax.axhline(y=0, color='blue', linewidth=1.0)
		self.fig.canvas.mpl_connect('button_press_event', self.handle_click)
		
		
	def handle_click(self, event):
		"""
		Deals with a user click on the interactive plot.
		"""

		## redraw line at the new click point
		self.line.set_ydata([event.ydata, event.ydata])

		## reset color of all line segments	to be black
		self.ax.collections[0].set_color('black')
		self.ax.collections[1].set_color('black')
		
		## get the clusters and the clusters attribute
		cut = self.line.get_ydata()[0]
		self.clusters, active_nodes = self.T.upperSetCluster(cut, mode='mass')
			
		# reset vertical segment colors
		palette = plutl.Palette(use='scatter')
		segclr = np.array([[0.0, 0.0, 0.0]] * len(self.segmap))
		splitclr = np.array([[0.0, 0.0, 0.0]] * len(self.splitmap))

		for i, node_ix in enumerate(active_nodes):
			subtree = makeSubtree(self.T, node_ix)

			# set new segment colors
			seg_replace = np.in1d(self.segmap, subtree.nodes.keys())
			segclr[seg_replace] = palette.colorset[i, :]

			split_replace = np.in1d(self.splitmap, subtree.nodes.keys())
			splitclr[split_replace] = palette.colorset[i, :]
	
		self.ax.collections[0].set_color(segclr)
		self.ax.collections[1].set_color(splitclr)
		self.fig.canvas.draw()


		## plot the clustered points in a new window
		if 'scatter' in self.output:
	
			# determine the data dimension
			if len(self.X.shape) == 1:
				n = len(self.X)
				p = 1
			else:
				n, p = self.X.shape
				
			if p == 1:
				cut_level = self.T.massToLevel(cut)
				hist_fig = plutl.clusterHistogram(self.X, self.clusters, self.fhat,
					self.f, levels=[cut_level])
				hist_fig.show()

			elif p == 2 or p == 3:
				base_clr = [217.0 / 255.0] * 3  ## light gray
				black = [0.0, 0.0, 0.0]
			
				clr_matrix = plutl.makeColorMatrix(n, bg_color=base_clr, bg_alpha=0.72,
					ix=self.clusters[:, 0], fg_color=self.clusters[:, 1], fg_alpha=0.68)
				edge_matrix = plutl.makeColorMatrix(n, bg_color=black, bg_alpha=0.38,
					ix=None)
				pts_fig = plutl.plotPoints(self.X, clr=clr_matrix,
					edgecolor=edge_matrix)
				pts_fig.show()

		
	def show(self):
		"""
		Show the interactive plot canvas.
		"""
		self.fig.show()
		
		
	def getClusters(self):
		"""
		Return the cluster memberships for the currently selected level or mass value.
		"""
		return self.clusters
		


