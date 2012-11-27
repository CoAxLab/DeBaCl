##############################################################
## Brian P. Kent
## densityCluster.py
## Created: 20120821
## Updated: 20121115
## Classes and functions for density-based clustering and visualization.
##############################################################


##############
### SET UP ###
##############
import sys
sys.path.append('../utils/')
import clustUtils as clutl
import plotUtils as plutl

import numpy as np
import scipy.spatial.distance as spdist
import igraph as igr
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.widgets import Button



#########################
### Class definitions ###
#########################
class TreeComponentTool:
	"""
	Allows the user to select and plot points in a particular component of the density
	tree, represented by vertical line segments in the tree.
	"""

	def __init__(self, tree, pts, height_mode='mass', width_mode='uniform',
		output=['tree', 'scatter'], s=20):

		self.T = tree
		self.X = pts
		self.height_mode = height_mode
		self.width_mode = width_mode
		self.output = output
		self.size = s
		self.fig, self.segments, self.segmap, self.splits, self.splitmap = self.T.plot(
			height_mode, width_mode, gap=0.15)
		self.fig.suptitle('Cluster Tree Component Selector', fontsize=14, weight='bold')
		
		self.ax = self.fig.axes[0]
		self.ax.set_zorder(0.1)  # sets the first axes to have priority for picking
		segments = self.ax.collections[0]  # the line collection
		segments.set_picker(15)

		self.fig.canvas.mpl_connect('pick_event', self.handle_pick)
		self.fig.canvas.mpl_connect('button_press_event', self.handle_click)
		
		self.confirm = []  # text box that confirms when a component has been selected
		self.remove_flag = False
		
		
	def handle_click(self, event):
		"""
		Gets rid of the confirmation text box if it's present.
		"""
		
		## reset segments to all be black
		self.ax.collections[0].set_color('black')
		self.ax.collections[1].set_color('black')

		## deal with confirmation box
		if len(self.confirm) == 0:
			self.remove_flag = False
			
		elif len(self.confirm) == 1:
			if self.remove_flag == True:
				box = self.confirm.pop()
				box.remove()
				self.remove_flag = False
				self.fig.canvas.draw()
			else:
				self.remove_flag = True
				
		else:
			box = self.confirm.pop()
			box.remove()
			
		
	def handle_pick(self, event):
		"""
		Return the members of the component that is picked out.
		"""
	
		## get the component members and subtree
		ix_seg = event.ind[0]
		node_ix = self.segmap[ix_seg]
		self.component = self.T.nodes[node_ix].members
		self.subtree = makeSubtree(self.T, node_ix)
		
		
		## draw confirmation text box
		props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
		textstr = "Component selected!"
		self.confirm.append(self.ax.text(0.37, 0.07, textstr, transform=self.ax.transAxes,
			fontsize=12, verticalalignment='top', bbox=props))
		self.fig.canvas.draw()
		
		
		## recolor the original tree
		c = plutl.Palette(use='scatter').colorset[0, :]

		# set vertical segment colors
		segclr = np.array([[0.0, 0.0, 0.0]] * len(self.segmap))
		ix_replace = np.in1d(self.segmap, self.subtree.nodes.keys())
		segclr[ix_replace] = c
		self.ax.collections[0].set_color(segclr)
		
		# set horizontal segment colors
		splitclr = np.array([[0.0, 0.0, 0.0]] * len(self.splitmap))
		ix_replace = np.in1d(self.splitmap, self.subtree.nodes.keys())
		splitclr[ix_replace] = c
		self.ax.collections[1].set_color(splitclr)
		
		self.fig.canvas.draw()

		
		# plot the component points in a new window (if output==True)
		if 'scatter' in self.output:
			n = self.X.shape[0]
			
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
		
		
		## construct the new subtree and show it
		if 'tree' in self.output:
			subfig = self.subtree.plot(height_mode=self.height_mode,
				width_mode=self.width_mode)[0]
			subfig.show()
			
			
	def show(self):
		self.fig.show()
		
		
	def getComponent(self):
		return self.component


	def getSubtree(self):
		return self.subtree




class TreeClusterTool:
	"""
	Allows the user to get a clustering of the upper level set points at any level of
	a density tree.
	"""

	def __init__(self, tree, pts, height_mode='mass', width_mode='uniform',
		output=['tree', 'scatter'], s=20):
		
		self.T = tree
		self.X = pts
		self.height_mode = height_mode
		self.width_mode = width_mode
		self.output = output
		self.size = s		
		self.clusters = None
		
		self.fig, self.segments, self.segmap, self.splits, self.splitmap = self.T.plot(
			height_mode, width_mode, gap=0.15)
		self.fig.suptitle('Cluster Tree Clustering Tool', fontsize=14, weight='bold')
		self.ax = self.fig.axes[0]
		self.ax.set_zorder(0.1)  # sets the first axes to have priority for picking
		
		self.line = self.ax.axhline(y=0, color='blue', linewidth=1.0)
		self.fig.canvas.mpl_connect('button_press_event', self.handle_click)
		
		ax2 = self.fig.add_axes([0.40, 0.01, 0.20, 0.04])
		self.button = Button(ax2, 'Set clusters')
		self.button.on_clicked(self.handle_button)
		
		self.confirm = None  # a text box that indicates when clusters have been computed
		
		
	
	def handle_click(self, event):

		## reset color of all line segments	to be black
		self.ax.collections[0].set_color('black')
		self.ax.collections[1].set_color('black')
	
		## get rid of the confirmation text box if it's there.
		if self.confirm != None:
			self.confirm.remove()
			self.confirm = None
		
		if event.inaxes == self.line.axes:
			self.line.set_ydata([event.ydata, event.ydata])
		
		self.fig.canvas.draw()
		
		
		
	def handle_button(self, event):

		## get the clusters and the clusters attribute
		cut = self.line.get_ydata()[0]
		self.clusters, active_nodes = self.T.clusterUpperSet(cut, mode=self.height_mode)
		

		## draw confirmation text box
		props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
		textstr = "Clusters computed!"
		self.confirm = self.ax.text(0.37, 0.07, textstr, transform=self.ax.transAxes,
			fontsize=12, verticalalignment='top', bbox=props)
		self.fig.canvas.draw()
		
		
		## recolor the existing plot
		palette = plutl.Palette(use='scatter')

		# set vertical segment colors
		segclr = np.array([[0.0, 0.0, 0.0]] * len(self.segmap))
		splitclr = np.array([[0.0, 0.0, 0.0]] * len(self.splitmap))

		for i, node_ix in enumerate(active_nodes):
			subtree = makeSubtree(self.T, node_ix)

			# set vertical segment color
			seg_replace = np.in1d(self.segmap, subtree.nodes.keys())
			segclr[seg_replace] = palette.colorset[i, :]

			# set horizontal segment color
			split_replace = np.in1d(self.splitmap, subtree.nodes.keys())
			splitclr[split_replace] = palette.colorset[i, :]
		
		self.ax.collections[0].set_color(segclr)
		self.ax.collections[1].set_color(splitclr)
		self.fig.canvas.draw()


		## plot the clustered points in a new window
		if 'scatter' in self.output:
			n = self.X.shape[0]
			base_clr = [217.0 / 255.0] * 3  ## light gray
			black = [0.0, 0.0, 0.0]
			
			clr_matrix = plutl.makeColorMatrix(n, bg_color=base_clr, bg_alpha=0.72,
				ix=self.clusters[:, 0], fg_color=self.clusters[:, 1], fg_alpha=0.68)

			edge_matrix = plutl.makeColorMatrix(n, bg_color=black, bg_alpha=0.38, ix=None)

			fig = plutl.plotPoints(self.X, clr=clr_matrix, edgecolor=edge_matrix)
			fig.show()

		
	def show(self):
		self.fig.show()
		
		
	def get_clusters(self):
		return self.clusters
		



class ConnectedComponent:
	"""
	Defines a connected component for level set clustering.
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
		component = ConnectedComponent(self.idnum, self.parent, self.children,
			self.start_level, self.end_level, self.start_mass, self.end_mass,
			self.members)
			
		return component
		



class ClusterTree:
	"""
	Defines a cluster tree, i.e. a collection of connected components organized
	hierarchically.
	"""
	
	def __init__(self, bg_sets, levels):
		self.bg_sets = bg_sets
		self.levels = levels
		self.n = sum([len(x) for x in bg_sets])
		self.nodes = {}
		self.subgraphs = {}
		
		
		
#	def pruneRootsByLevel(self, thresh):
#		"""
#		Prune a tree by removing root nodes that end below a set level.
#		"""
#	
#		remove_queue = [k for k, v in self.nodes.iteritems() if v.parent==None and
#			v.end_level <= thresh]
#		
#		while len(remove_queue) > 0:
#			k = remove_queue.pop()
#			remove_queue += self.nodes[k].children
#			del self.nodes[k]


		
	def pruneRootsBySize(self, delta):
		"""
		Prune a tree by removing root nodes that have too few members.
		"""
	
		thresh = round(delta * self.n)

		remove_queue = [k for k, v in self.nodes.iteritems() if v.parent==None and
			len(v.members) <= thresh]
		
		while len(remove_queue) > 0:
			k = remove_queue.pop()
			remove_queue += self.nodes[k].children
			del self.nodes[k]
			
			
			
	def pruneBySize(self, delta, mode='proportion'):
		"""
		Prune a tree by removing all nodes with too few members.
		"""
		
		if mode == 'proportion':
			thresh = round(delta * self.n)
		else:
			thresh = delta
			
		self.nodes = {k: v for k, v in self.nodes.iteritems() if len(v.members) > thresh}
		
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
	

		
		
	def summarize(self):
		
		names = ('index', 'lambda1', 'lambda2', 'alpha1', 'alpha2', 'size', 'parent',
			'children')
		widths = {
			'ix':5, 
			'lambda1': max([len(str(x.start_level).split('.')[0]) for x in self.nodes.values()]) + 9,
			'lambda2': max([len(str(x.end_level).split('.')[0]) for x in self.nodes.values()]) + 9,
			'alpha1': max([len(str(x.start_mass).split('.')[0]) for x in self.nodes.values()]) + 9,
			'alpha2': max([len(str(x.end_mass).split('.')[0]) for x in self.nodes.values()]) + 9,
			'size': max(4, max([len(str(len(x.members))) for x in self.nodes.values()])) + 2,
			'parent': max(6, max([len(str(x.parent)) for x in self.nodes.values()])) + 2,
			'child': max(7, max([len(str(x.children)) for x in self.nodes.values()])) + 2}
		
		head = '{0:<{ix}}{1:>{lambda1}}{2:>{lambda2}}{3:>{alpha1}}'.format(*names, **widths) + \
			'{4:>{alpha2}}{5:>{size}}{6:>{parent}}{7:>{child}}'.format(*names, **widths)
		print head
		
		for k, v in self.nodes.iteritems():
			line = '{0:<{ix}}{1.start_level:>{lambda1}.6f}{1.end_level:>{lambda2}.6f}'.format(
					k, v, len(v.members), **widths) + \
				'{1.start_mass:>{alpha1}.6f}{1.end_mass:>{alpha2}.6f}'.format(
					k, v, len(v.members), **widths) + \
				'{2:>{size}}{1.parent:>{parent}}{1.children:>{child}}'.format(
					k, v, len(v.members), **widths) 			
			print line
			
		
		
		
	def plot(self, height_mode='mass', width_mode='uniform', xpos='middle', sort=True,
		title='', gap=0.05, color=False, color_nodes=None):
		"""
		Make and return a plot of the cluster tree. For each root connected component,
		traverse the branches recursively by depth-first search.
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
			branch_segs, branch_splits, branch_segmap, branch_splitmap = constructBranchMap(
				self, ix, (intervals[i], intervals[i+1]), height_mode, width_mode, xpos, sort)
			segments = dict(segments.items() + branch_segs.items())
			splits = dict(splits.items() + branch_splits.items())
			segmap += branch_segmap
			splitmap += branch_splitmap
			
			
		## get the the vertical line segments in order of the segment map (segmap)
		verts = [segments[k] for k in segmap]
		lats = [splits[k] for k in splitmap]
			
		## Find the fraction of nodes in each segment (to use as linewidths)
		thickness = [max(0.45, 10.0 * len(self.nodes[x].members)/n) for x in segmap]
		
		
		## Find the right tick marks for the plot
		level_ticks = np.sort(list(set([v.start_level for v in self.nodes.itervalues()] + \
			[v.end_level for v in self.nodes.itervalues()])))
		level_tick_labels = [str(round(lvl, 2)) for lvl in level_ticks]
		
		mass_ticks = np.sort(list(set([v.start_mass for v in self.nodes.itervalues()] + \
			[v.end_mass for v in self.nodes.itervalues()])))
		mass_tick_labels = [str(round(m, 2)) for m in mass_ticks]

	
		## Set up the plot framework
		fig, ax = plt.subplots(1, figsize=(10, 10))
		fig.suptitle(title, size=14, weight='bold')
	
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
		
		
		
	def clusterUpperSet(self, cut, mode):
		"""
		Cut the tree at the desired level or probability content value to get the connected
		components of the upper level set.
		"""
		
		## identify upper level points and the nodes active at the cut
		if mode == 'mass':
			n_bg = [len(x) for x in self.bg_sets]
			alphas = np.cumsum(n_bg) / (1.0 * self.n)
			upper_levels = np.where(alphas > cut)[0]
			active_nodes = [k for k, v in self.nodes.iteritems() if v.start_mass <= cut and \
				v.end_mass > cut]

		else:
			upper_levels = np.where(self.levels > cut)[0]
			active_nodes = [k for k, v in self.nodes.iteritems() if v.start_level <= cut and \
				v.end_level > cut]

		upper_pts = np.array([y for x in upper_levels for y in self.bg_sets[x]])
		

		## find intersection between upper set points and each active component
		points = []
		cluster = []

		for i, c in enumerate(active_nodes):
			cluster_pts = upper_pts[np.in1d(upper_pts, self.nodes[c].members)]
			points.extend(cluster_pts)
			cluster += ([i] * len(cluster_pts))

		return np.array([points, cluster], dtype=np.int).T, active_nodes



	def getKCut(self, k):
		"""
		Find the lowest level cut that has k connected components. If there are no
		levels that have k components, then find the lowest level that has at least k 
		components. If no levels have > k components, find the lowest level that has the 
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
		
		
		
		
	def maxCluster(self):
		"""
		Cut the tree so that every leaf component (i.e. non-branching component) becomes a
		foreground cluster. This does not generally cut the tree at any particular level.
		"""
	
		leaves = [k for k, v in self.nodes.items() if v.children == []]
		
		## find components in the leaves
		points = []
		cluster = []

		for i, k in enumerate(leaves):
			points.extend(self.nodes[k].members)
			cluster += ([i] * len(self.nodes[k].members))

		return np.array([points, cluster], dtype=np.int).T
		
		


	def multilevelKClust(self, k):
		"""
		Cut the level set tree so there are k clusters, but return the points at the base
		of each foreground group rather than the points precisely at a given level.
		"""
		
		cut = self.getKCut(k)
		active_nodes = [e for e, v in self.nodes.iteritems() \
			if v.start_level <= cut and v.end_level > cut]
			
		points = []
		cluster = []
		
		for i, c in enumerate(active_nodes):
		
			cluster_pts = self.nodes[c].members
			points.extend(cluster_pts)
			cluster += ([i] * len(cluster_pts))

		return np.array([points, cluster], dtype=np.int).T


		
		
		
#######################################		
### CLUSTER TREE GENERATOR FUNCTION ###
#######################################
def generateChaudhuriTree():
	"""
	Generate a level set tree using the Chaudhuri & Dasgupta methodology.
	"""

	return 13
	
	
	

def generateTree(W, levels, bg_sets, mode='general'):
	"""
	Construct a level set tree from the upper level sets of a similarity matrix W.
	"""
	
	n = np.alen(W)
	levels = [float(x) for x in levels]
	
	## Initialize the graph and cluster tree
	G = igr.Graph.Adjacency(W.tolist(), mode=igr.ADJ_MAX)
	G.vs['index'] = range(n)

	T = ClusterTree(bg_sets, levels)
	cc0 = G.components()
	
	if mode == 'density':
		start_level = 0.0
	else:
		start_level = float(np.min(levels) - 1.0)
		
	for i, c in enumerate(cc0):
		T.subgraphs[i] = G.subgraph(c)
		T.nodes[i] = ConnectedComponent(i, parent=None, children=[], start_level=start_level,
			end_level=None, start_mass=0.0, end_mass=None, members=G.vs[c]['index'])
	
	
	# Loop through the removal grid
	for level, bg in zip(levels, bg_sets):
		
		# compute mass and level
		mass = 1.0 - (sum([x.vcount() for x in T.subgraphs.itervalues()]) - len(bg)) / (1.0 * n)

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
								start_mass=mass, end_mass=None, members=H.vs[c]['index'])
											
		# update active components
		for k in deactivate_keys:
			del T.subgraphs[k]
			
		T.subgraphs.update(activate_subgraphs)
		
	return T
	
	

	
#########################	
### UTILITY FUNCTIONS ###
#########################
def constructBranchMap(T, ix, interval, height_mode, width_mode, xpos, sort):

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
			segments[ix] = (([xpos, T.nodes[ix].start_level], [xpos, T.nodes[ix].end_level]))
		else:
			segments[ix] = (([xpos, T.nodes[ix].start_mass], [xpos, T.nodes[ix].end_mass]))

		
	
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
			branch_segs, branch_splits, branch_segmap, branch_splitmap = constructBranchMap(
				T, child, branch_interval, height_mode, width_mode, xpos, sort)
				
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
			segments[ix] = (([xpos, T.nodes[ix].start_level], [xpos, T.nodes[ix].end_level]))
		else:
			segments[ix] = (([xpos, T.nodes[ix].start_mass], [xpos, T.nodes[ix].end_mass]))
	
	
	return segments, splits, segmap, splitmap
	



def constructDensityGrid(density, mode='mass', n_grid=None):
	"""
	Create a list of lists of points to remove at each iteration of a level set or mass
	tree. Also create a list of the density level at each iteration.
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
			bg_sets = [list(np.where(density==uniq_dens[i])[0]) for i in range(len(uniq_dens))]
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
	
	
	
def makeSubtree(tree, ix):
	"""
	Return the subtree with node 'ix' as the root, and all ancestors of 'ix'.
	"""
	
	T = ClusterTree(bg_sets=[], levels=[])
	T.nodes[ix] = tree.nodes[ix].copy()
	T.nodes[ix].parent = None
	queue = tree.nodes[ix].children[:]
	
	while len(queue) > 0:
		branch_ix = queue.pop()
		T.nodes[branch_ix] = tree.nodes[branch_ix]
		queue += tree.nodes[branch_ix].children
	
	return T




	


