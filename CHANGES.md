Master (not yet on PyPI)
------------------------
- The method `branch_partition` has been added to the `LevelSetTree`. This
  method assigns each point a label corresponding to the *highest density* node
  to which the point belongs in the level set tree.

- The tree table now prints when the tree is called by itself. Now both
  `print(tree)` and `tree` print the tree's summary table to the console.

v1.0, November 2015
-------------------
This release is a major overhaul of DeBaCl. The primary goal is to make the
level set trees (LSTs) easier to use, by removing much of the experimental and
quasi-analysis code from my dissertation work, adding unit tests to improve
code robustness, and simplifying the level set tree API. The experimental code
will not vanish; I will move it to separate branches or a new repository.
    - Brian (papayawarrior)

**Logistics**
- No more dependency on igraph. Graph computation is now done with Networkx.

- No more dependency on Pandas. Level set tree printing is now done with
  Prettytable.

- The dependencies on Scipy and Matplotlib are now recommended but optional.
  Scipy is now used only for constructing similarity graphs by brute force.

- Saving and loading now use cPickle instead of scipy.io’s `loadmat` function.

- The level set tree constructor functions and the `LevelSetTree` are now
  accessible directly from the `debacl` namespace.

**Level set tree construction**
- The main level set tree class `GeomTree` has been renamed to `LevelSetTree`.

- The similarity graph LST constructor `constructTree` has been renamed to
  `construct_tree_from_graph`.

- `construct_tree_from_graph` now takes the similarity graph in the form of an
  adjacency list, rather than an adjacency matrix.

- `construct_tree_from_graph` no longer requires the user to pre-compute
  density levels and "background sets" of instances. The function now requires only an adjacency list (to represent a similarity graph) and a density estimate for each data instance.

- `LevelSetTree` objects contain the density estimate for each input instance,
  rather than a collection of background sets.

- The similarity graph utilities `knn_graph` and `epsilon_graph` now return
  adjacency lists rather than adjacency matrices.

- The `constructDensityGrid` utility has been split into to two functions:
  `define_density_mass_grid` and `define_density_level_grid`. The LST
  constructors use the mass option, but the density level option is left for
  legacy purposes.

- The `gaussianGraph` utility has been removed.

**Level set tree printing and plotting**
- Changed tree table column names from 'lambda1', 'lambda2', 'alpha1', and
  'alpha2' to 'start_level', 'end_level', 'start_mass', 'end_mass'.

- The level set tree plot forms have been renamed from 'lambda', 'alpha', and
  'kappa' to 'density', 'mass', and 'branch-mass'.

- The 'width' parameter in the `LevelSetTree.plot` method has been renamed to
  'horizontal_spacing', and the 'mass' option for this parameter has been
  renamed to 'proportional'.

- Added a tree method ‘get_leaf_nodes’ which just returns the indices of the
  leaf nodes.

- Tree plotting now returns the color assigned to each node.

- Tree plotting no longer returns the ‘segmap’ and ‘splitmap’ objects.

- Tree plot objects ‘segments’ and ‘splits’ have been renamed to ‘node_coords’
  and ‘split_coords’.

- The interactive plotting tools `ComponentGUI` and `ClusterGUI` have been
  removed.

- Plotting utilities (`Palette`, `plot_foreground`, `make_color_matrix`, and
  `setPlotParams`) have been removed.

- The `clusterHistogram` utility for illustrating the level set tree method on
  1D data has been removed.

- The `plot` method of `LevelSetTree` objects no longer accept the 'gap'
  parameter for adding extra whitespace on the bottom of the plot.

- The 'old' form of level set tree plots has been removed.

- The `plot` method of `LevelSetTree` objects no longer accept the 'sort'
  parameter; the branches are always sorted now from highest to lowest mass.

**Level set tree pruning**
- Level set tree pruning can now be done directly in the tree constructors.
  There’s no need to call the `prune` method separately (although it's still a
  valid pattern).

- The `prune` method returns a new, pruned `LevelSetTree` object. This means
  pruning at various thresholds can be done from the same level set tree,
  without re-building the tree each time.

- The `prune` no longer takes a method parameter. It assumes the
  'merge-by-size' method.

- `LevelSetTree` objects now have a `prune_threshold` attribute.

**Level set tree clustering**
- Changed the name of `get_cluster_labels` to `get_clusters`.

- Changed the name 'all-mode' clustering to 'leaf' clustering.

- Added the ‘fill_backround’ flag to `get_clusters` to fill the background
  points with -1.

- Changed all clustering methods to return only cluster labels, not the list of
  active nodes.

- An instance's cluster label is now the index of the level set tree node that
  is "activated" by a given clustering method and to which the instance
  belongs. Previously cluster labels were consecutive integers.

- Added a utility function `reindex_cluster_labels` to re-index cluster labels
  to be consecutive integers.

- The `assignBackgroundPoints` utility function for assigning low-density
  points to clusters has bee removed. Any classifier (in scikit-learn, for
  example) can be used for this task.

**Bugfixes**
- External library imports are now hidden to avoid namespace pollution.
 
- The `num_levels` attribute is now correctly populated.

**Miscellaneous**
- Use Python built-in logging module instead of print statements.

- The `subgraphs` attribute of a `LevelSetTree` is now hidden from the user.

- Helper `LevelSetTree` methods are now hidden from the user.

- The *cd_tree.py* module containing the original level set tree algorithm
  (Chaudhuri & Dasgupta, 2010) tree has been removed.
  
- The `drawSample` utility has been removed. This can be done now with Numpy.
 
v0.2, July 2013
---------------
Initial release
