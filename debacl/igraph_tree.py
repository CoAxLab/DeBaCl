##############################################################
## Brian P. Kent
## igraph_tree.py
## Created: 20140707
## Updated: 20140707
##############################################################

"""
Tree construction methods that utilize the igraph package. igraph is faster than
networkx at computing connected components at each level of the tree, but is
typically more difficult to install. As such, the main tree construction tools
are in level_set_tree.py, which is based on networkx, while the igraph version
remains in this module for those who require more speed.
"""

import level_set_tree as lst  # main tree methods
import utils as utl  		  # DeBaCl utilities

try:
	import igraph as igr
except:
	raise ImportError("igraph failed to load. The igraph-based tree " + \
		"construction requires the igraph package.")


def construct_tree(adjacency_list, density_levels, background_sets,
	verbose=False):
	"""
	Construct a level set tree. A level set tree is constructed by identifying
	connected components of in a k-nearest neighbors graph at successively
	higher levels of a probability density estimate.
	
	Parameters
	----------
	adjacency_list : dict [list [int]]
		Adjacency list of the k-nearest neighbors graph on the data. Each key
		represents represents one of the 'n' observations, while the values are
		lists containing the indices of the k-nearest neighbors.

	levels: numpy array[float]
		Density levels at which connected components are computed. Typically
		this includes all unique values of a probability density estimate, but
		it can be a coarser grid for fast approximate tree estimates.
	
	bg_sets: list of lists
		Specify which points to remove as background at each density level in
		'levels'.
	
	verbose: {False, True}, optional
		If set to True, then prints to the screen a progress indicator every 100
		levels.
	
	Returns
	-------
	T : levelSetTree
		See the LevelSetTree class for attributes and method definitions.
	"""
	
	n = len(adjacency_list)
	_density_levels = [float(x) for x in density_levels]

	## Initialize the graph and cluster tree
	edge_list = utl.adjacency_to_edge_list(adjacency_list, self_edge=False)
	
	G = igr.Graph(n=n, edges=edge_list, directed=False,
		vertex_attrs={'index':range(n)})

	T = lst.LevelSetTree(background_sets, _density_levels)
	cc0 = G.components()
		
	for i, c in enumerate(cc0):
		T.subgraphs[i] = G.subgraph(c)
		T.nodes[i] = lst.ConnectedComponent(i, parent=None, children=[],
			start_level=0., end_level=None, start_mass=0.0,
			end_mass=None, members=G.vs[c]['index'])
	
	
	# Loop through the removal grid
	for i, (level, bg) in enumerate(zip(_density_levels, background_sets)):
	
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
						
							T.nodes[new_key] = lst.ConnectedComponent(new_key,
								parent=k, children=[], start_level=level,
								end_level=None, start_mass=mass, end_mass=None,
								members=H.vs[c]['index'])
											
		# update active components
		for k in deactivate_keys:
			del T.subgraphs[k]
			
		T.subgraphs.update(activate_subgraphs)
		
	return T


def construct_tree_from_matrix(W, levels, bg_sets, verbose=False):
	"""
	Construct a level set tree. A level set tree is constructed by identifying
	connected components of in a k-nearest neighbors graph at successively
	higher levels of a probability density estimate.
	
	Parameters
	----------
	W : numpy array
		Adjacency matrix

	levels: numpy array[float]
		Density levels at which connected components are computed. Typically
		this includes all unique values of a probability density estimate, but
		it can be a coarser grid for fast approximate tree estimates.
	
	bg_sets: list of lists
		Specify which points to remove as background at each density level in
		'levels'.
	
	verbose: {False, True}, optional
		If set to True, then prints to the screen a progress indicator every 100
		levels.
	
	Returns
	-------
	T : levelSetTree
		See the LevelSetTree class for attributes and method definitions.
	"""
	
	n = len(W)
	_levels = [float(x) for x in levels]
	
	## Initialize the graph and cluster tree
	G = igr.Graph.Adjacency(W.tolist(), mode=igr.ADJ_MAX)
	G.vs['index'] = range(n)

	T = lst.LevelSetTree(bg_sets, _levels)
	cc0 = G.components()
		
	for i, c in enumerate(cc0):
		T.subgraphs[i] = G.subgraph(c)
		T.nodes[i] = lst.ConnectedComponent(i, parent=None, children=[],
			start_level=0., end_level=None, start_mass=0.0,
			end_mass=None, members=G.vs[c]['index'])
	
	
	# Loop through the removal grid
	for i, (level, bg) in enumerate(zip(_levels, bg_sets)):
	
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
						
							T.nodes[new_key] = lst.ConnectedComponent(new_key,
								parent=k, children=[], start_level=level,
								end_level=None, start_mass=mass, end_mass=None,
								members=H.vs[c]['index'])
											
		# update active components
		for k in deactivate_keys:
			del T.subgraphs[k]
			
		T.subgraphs.update(activate_subgraphs)
		
	return T