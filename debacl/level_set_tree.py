"""
Main functions and classes for the DEnsity-BAsed CLustering (DeBaCl) toolbox.
Includes functions to construct and modify level set trees produced by standard
geometric clustering on each level. Also defines tools for interactive data
analysis and clustering with level set trees.
"""

## Built-in packages
import logging as _logging
import copy as _copy
import cPickle as _cPickle
import utils as _utl

_logging.basicConfig(level=_logging.INFO, datefmt='%Y-%m-%d %I:%M:%S',
                     format='%(levelname)s (%(asctime)s): %(message)s')

## Required packages
try:
    import numpy as _np
    import networkx as _nx
    from prettytable import PrettyTable as _PrettyTable
except:
    raise ImportError("DeBaCl requires the numpy, networkx, and " +
                      "prettytable packages.")

## Soft dependencies
try:
    import matplotlib.pyplot as _plt
    from matplotlib.collections import LineCollection as _LineCollection
    _HAS_MPL = True
except:
    _HAS_MPL = False
    _logging.warning("Matplotlib could not be loaded, so DeBaCl plots will " +
                     "fail.")


class ConnectedComponent(object):
    """
    Defines a connected component for level set tree construction. A level set
    tree is really just a set of ConnectedComponents.
    """

    def __init__(self, idnum, parent, children, start_level, end_level,
                 start_mass, end_mass, members):

        self.idnum = idnum
        self.parent = parent
        self.children = children
        self.start_level = start_level
        self.end_level = end_level
        self.start_mass = start_mass
        self.end_mass = end_mass
        self.members = members


class LevelSetTree(object):
    """
    Level Set Tree attributes and methods. The level set tree is a collection
    of connected components organized hierarchically, based on a k-nearest
    neighbors density estimate and connectivity graph.

    .. warning::

        LevelSetTree objects should not generally be instantiated directly,
        because they will contain empty node hierarchies. Use the tree
        constructors :func:`construct_tree` or
        :func:`construct_tree_from_graph` to instantiate a LevelSetTree model.

    Parameters
    ----------
    density : list[float] or numpy array
        The observations removed as background points at each successively
        higher density level.

    levels : array_like
        Probability density levels at which to find clusters. Defines the
        vertical resolution of the tree.
    """

    def __init__(self, density=[], levels=[]):
        self.density = density
        self.levels = levels
        self.num_levels = len(levels)
        self.prune_threshold = None
        self.nodes = {}
        self._subgraphs = {}

    def __str__(self):
        """
        Print the tree summary table.
        """
        summary = _PrettyTable(["id", "start_level", "end_level", "start_mass",
                                "end_mass", "size", "parent", "children"])
        for node_id, v in self.nodes.items():
            summary.add_row([node_id,
                             v.start_level,
                             v.end_level,
                             v.start_mass,
                             v.end_mass,
                             len(v.members),
                             v.parent,
                             v.children])

        for col in ["start_level", "end_level", "start_mass", "end_mass"]:
            summary.float_format[col] = "5.3"

        return summary.get_string()

    def prune(self, threshold):
        """
        Prune the tree by recursively merging small leaf nodes into larger
        sibling nodes. The LevelSetTree is *immutable*, so pruning returns a
        new LevelSetTree whose nodes all contain more points than 'threshold'.

        Parameters
        ----------
        threshold : int
            Nodes smaller than this will be merged.

        Returns
        -------
        out : LevelSetTree
            A pruned level set tree. The original tree is unchanged.

        Examples
        --------
        >>> X = numpy.random.rand(100, 2)
        >>> tree = debacl.construct_tree(X, k=8, prune_threshold=2)
        >>> tree2 = tree.prune(threshold=12)
        """
        return self._merge_by_size(threshold)

    def save(self, filename):
        """
        Save a level set tree object to file. All members of the level set tree
        are serialized with the cPickle module and saved to file.

        Parameters
        ----------
        filename : str
            File to save the tree to. The filename extension does not matter
            for this method (although operating system requirements still
            apply).

        See Also
        --------
        load_tree

        Examples
        --------
        >>> X = numpy.random.rand(100, 2)
        >>> tree = debacl.construct_tree(X, k=8, prune_threshold=5)
        >>> tree.save('my_tree')
        """
        with open(filename, 'wb') as f:
            _cPickle.dump(self, f, _cPickle.HIGHEST_PROTOCOL)

    def plot(self, form='mass', horizontal_spacing='uniform', color_nodes=[],
             colormap='Dark2'):
        """
        Plot the level set tree as a dendrogram and return coordinates and
        colors of the branches.

        Parameters
        ----------
        form : {'mass', 'density', 'branch-mass'}, optional

            Main form of the plot.

            - 'density': the traditional form of the LST dendrogram where the
              vertical scale is density levels.

            - 'mass' (default): very similar to the 'density' form, but draws
              the dendrogram based on the mass of upper (density) level sets.

            - 'branch-mass': each node is drawn in the dendrogram so that its
              length is proportional to its mass, *excluding* the masses of the
              node's children. In this form, the lengths of the segments
              representing the tree nodes sum to 1.

        horizontal_spacing : {'uniform', 'proportional'}, optional
            Determines how much horizontal space each level set tree node is
            given. The default of "uniform" gives each child node an equal
            fraction of the parent node's horizontal space. If set to
            'proportional', then horizontal space is allocated proportionally
            to the mass of a node relative to its siblings.

        color_nodes : list, optional
            Nodes to color in the level set tree. For each node, the subtree
            for which that node is the root is drawn with a single color.

        colormap : str, optional
            Matplotlib colormap, used only if 'color_nodes' contains at least
            one node index. Default is the 'Dark2' colormap. "Qualitative"
            colormaps are highly recommended.

        Returns
        -------
        fig : matplotlib figure
            Use fig.show() to view, fig.savefig() to save, etc.

        node_colors : dict
            RGBA 4-tuple

        node_coords : dict
            Coordinates of vertical line segment endpoints representing LST
            nodes.

        split_coords : dict
            Coordinates of horizontal line segment endpoints, representing
            splits in the level set tree. There is a horizontal line segment
            for *each* child in a split, and the keys in the 'split_coords'
            dictionary indicate to which *child* the line segment belongs.

        Examples
        --------
        >>> X = numpy.random.rand(100, 2)
        >>> tree = debacl.construct_tree(X, k=8, prune_threshold=5)
        >>> plot = tree.plot(form='density')
        >>> fig = plot[0]
        >>> fig.show()
        """

        gap = 0.05
        min_node_width = 1.2

        ## Initialize the plot containers
        node_coords = {}
        split_coords = {}

        ## Find the root connected components and corresponding plot intervals
        ix_root = _np.array([k for k, v in self.nodes.iteritems()
                             if v.parent is None])
        n_root = len(ix_root)
        census = _np.array([len(self.nodes[x].members) for x in ix_root],
                           dtype=_np.float)
        n = sum(census)

        seniority = _np.argsort(census)[::-1]
        ix_root = ix_root[seniority]
        census = census[seniority]

        if horizontal_spacing == 'proportional':
            weights = census / n
            intervals = _np.cumsum(weights)
            intervals = _np.insert(intervals, 0, 0.0)
        else:
            intervals = _np.linspace(0.0, 1.0, n_root + 1)

        ## Do a depth-first search on each root to get segments for each branch
        for i, ix in enumerate(ix_root):
            if form == 'branch-mass':
                branch = self._construct_mass_map(
                    ix, 0.0, (intervals[i], intervals[i + 1]),
                    horizontal_spacing)
            else:
                branch = self._construct_branch_map(
                    ix, (intervals[i], intervals[i + 1]), form,
                    horizontal_spacing, sort=True)

            branch_node_coords, branch_split_coords, _, _ = branch
            node_coords.update(branch_node_coords)
            split_coords.update(branch_split_coords)

        ## Find the fraction of nodes in each segment (to use as line widths)
        node_widths = {k: max(min_node_width, 12.0 * len(node.members) / n)
                       for k, node in self.nodes.items()}

        ## Get the relevant vertical ticks
        primary_ticks = [(x[0][1], x[1][1]) for x in node_coords.values()]
        primary_ticks = _np.unique(_np.array(primary_ticks).flatten())
        primary_labels = [str(round(tick, 2)) for tick in primary_ticks]

        ## Set up the plot framework
        fig, ax = _plt.subplots()
        ax.set_position([0.11, 0.05, 0.78, 0.93])
        ax.set_xlim((-0.04, 1.04))
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.yaxis.grid(color='gray')
        ax.set_yticks(primary_ticks)
        ax.set_yticklabels(primary_labels)

        ## Form-specific details
        if form == 'branch-mass':
            kappa_max = max(primary_ticks)
            ax.set_ylim((-1.0 * gap * kappa_max, 1.04 * kappa_max))
            ax.set_ylabel("branch mass")

        elif form == 'density':
            ax.set_ylabel("density level")
            ymin = min([v.start_level for v in self.nodes.itervalues()])
            ymax = max([v.end_level for v in self.nodes.itervalues()])
            rng = ymax - ymin
            ax.set_ylim(ymin - gap * rng, ymax + 0.05 * rng)

        elif form == 'mass':
            ax.set_ylabel("mass level")
            ymin = min([v.start_mass for v in self.nodes.itervalues()])
            ymax = max([v.end_mass for v in self.nodes.itervalues()])
            rng = ymax - ymin
            ax.set_ylim(ymin - gap * rng, ymax + 0.05 * ymax)

        else:
            raise ValueError('Plot form not understood')

        ## Color the line segments.
        node_colors = {k: [0.0, 0.0, 0.0, 1.0] for k, v in self.nodes.items()}
        palette = _plt.get_cmap(colormap)
        colorset = palette(_np.linspace(0, 1, len(color_nodes)))

        for i, ix in enumerate(color_nodes):
            subtree = self._make_subtree(ix)

            for ix_sub in subtree.nodes.keys():
                node_colors[ix_sub] = list(colorset[i])

        ## Add the line segments to the figure.
        node_lines = _LineCollection(node_coords.values(),
                                     linewidths=node_widths.values(),
                                     colors=node_colors.values())
        ax.add_collection(node_lines)

        split_colors = {k: node_colors[k] for k in split_coords.keys()}
        split_lines = _LineCollection(split_coords.values(),
                                      colors=split_colors.values())
        ax.add_collection(split_lines)

        return fig, node_coords, split_coords, node_colors

    def get_clusters(self, method='leaf', fill_background=False, **kwargs):
        """
        Generic function for retrieving cluster labels from the level set tree.
        Dispatches a specific cluster labeling function.

        Parameters
        ----------
        method : {'leaf', 'first-k', 'upper-level-set', 'k-level'}, optional
            Method for obtaining cluster labels from the tree.

            - 'leaf': treat each leaf of the tree as a separate cluster.

            - 'first-k': find the first K non-overlapping clusters from the
              roots of the tree.

            - 'upper-level-set': cluster by cutting the tree at a specified
              density or mass level.

            - 'k-level': returns clusters at the lowest density level that has
              k nodes.

        fill_background : bool, optional
            If True, a label of -1 is assigned to background points, i.e. those
            instances not otherwise assigned to a high-density cluster.

        Other Parameters
        ----------------
        k : int
            If method is 'first-k' or 'k-level', this is the desired number of
            clusters.

        threshold : float
            If method is 'upper-level-set', this is the threshold at which to
            cut the tree.

        form : {'density', 'mass'}
            If method is 'upper-level-set', this is vertical scale which
            'threshold' refers to.

        Returns
        -------
        labels : 2-dimensional numpy array
            Each row corresponds to an observation. The first column indicates
            the index of the observation in the original data matrix, and the
            second column is the index of the LST node to which the observation
            belongs, *with respect to the clustering*. Note that the set of
            observations in this "foreground" set is typically smaller than the
            original dataset.

        See Also
        --------
        debacl.utils.reindex_cluster_labels, get_leaf_nodes

        Examples
        --------
        >>> X = numpy.random.rand(100, 2)
        >>> tree = debacl.construct_tree(X, k=8, prune_threshold=5)
        >>> labels = tree.get_clusters(method='leaf')
        """

        ## Retrieve foreground labels.
        if method == 'leaf':
            labels = self._leaf_cluster()

        elif method == 'first-k':
            required = set(['k'])
            if not set(kwargs.keys()).issuperset(required):
                raise ValueError("Incorrect arguments for the first-k " +
                                 "cluster labeling method.")
            else:
                k = kwargs.get('k')
                labels = self._first_K_cluster(k)

        elif method == 'upper-level-set':
            required = set(['threshold', 'form'])
            if not set(kwargs.keys()).issuperset(required):
                raise ValueError("Incorrect arguments for the upper-set " +
                                 "cluster labeling method.")
            else:
                threshold = kwargs.get('threshold')
                form = kwargs.get('form')
                labels = self._upper_set_cluster(threshold, form)

        elif method == 'k-level':
            required = set(['k'])
            if not set(kwargs.keys()).issuperset(required):
                raise ValueError("Incorrect arguments for the k-level " +
                                 "cluster labeling method.")
            else:
                k = kwargs.get('k')
                labels = self._first_K_level_cluster(k)

        else:
            raise ValueError("Cluster labeling method not understood.")

        ## If requested, fill in the background labels with -1
        if fill_background:
            n = len(self.density)
            full_labels = _np.vstack((_np.arange(n), [-1] * n)).T
            full_labels[labels[:, 0], 1] = labels[:, 1]
            labels = full_labels

        return labels

    def get_leaf_nodes(self):
        """
        Return the indices of leaf nodes in the level set tree.

        Returns
        -------
        leaves : list
            List of LST leaf node indices.

        See Also
        --------
        get_clusters

        Examples
        --------
        >>> X = numpy.random.rand(100, 2)
        >>> tree = debacl.construct_tree(X, k=8, prune_threshold=5)
        >>> leaves = tree.get_leaf_nodes()
        >>> print leaves
        [1, 5, 6]
        """
        return [k for k, v in self.nodes.items() if v.children == []]

    def branch_partition(self):
        """
        Partition the input data. Each instance's assigned label is the index
        of the *highest density* node to which the point belongs. This is
        similar to retrieving high-density clusters with background points
        labeled, except here the background points are labeled according to
        their internal node membership as well (not just that they're
        background points).

        Returns
        -------
        labels : 2-dimensional numpy array
            Each row corresponds to an observation. The first column indicates
            the index of the observation in the original data matrix, and the
            second column is the index of the LST node to which the observation
            belongs.

        See Also
        --------
        get_clusters

        Examples
        --------
        >>> X = numpy.random.rand(100, 2)
        >>> tree = debacl.construct_tree(X, k=8, prune_threshold=5)
        >>> labels = tree.branch_partition(method='leaf')
        """
        points = []
        labels = []

        for ix, node in self.nodes.items():
            branch_members = node.members.copy()

            for ix_child in node.children:
                child_node = self.nodes[ix_child]
                branch_members.difference_update(child_node.members)

            points.extend(branch_members)
            labels += ([ix] * len(branch_members))

        partition = _np.array([points, labels], dtype=_np.int).T
        return partition

    def _make_subtree(self, ix):
        """
        Return the subtree with node 'ix' as the root, and all ancestors of
        'ix'.

        Parameters
        ----------
        ix : int
            Node to use at the root of the new tree.

        Returns
        -------
        T : LevelSetTree
            A completely independent level set tree, with 'ix' as the root
            node.
        """

        T = LevelSetTree()
        T.nodes[ix] = _copy.deepcopy(self.nodes[ix])
        T.nodes[ix].parent = None
        queue = self.nodes[ix].children[:]

        while len(queue) > 0:
            branch_ix = queue.pop()
            T.nodes[branch_ix] = self.nodes[branch_ix]
            queue += self.nodes[branch_ix].children

        return T

    def _merge_by_size(self, threshold):
        """
        Prune splits from a tree based on size of child nodes. Merge members of
        child nodes rather than removing them.

        Parameters
        ----------
        threshold : numeric
            Tree branches with fewer members than this will be merged into
            larger siblings or parents.

        Returns
        -------
        tree : LevelSetTree
            A pruned level set tree. The original tree is unchanged.
        """

        tree = _copy.deepcopy(self)
        tree.prune_threshold = threshold

        ## remove small root branches
        small_roots = [k for k, v in tree.nodes.iteritems()
                       if v.parent is None and len(v.members) <= threshold]

        for root in small_roots:
            root_tree = tree._make_subtree(root)
            for ix in root_tree.nodes.iterkeys():
                del tree.nodes[ix]

        ## main pruning
        parents = [k for k, v in tree.nodes.iteritems()
                   if len(v.children) >= 1]
        parents = _np.sort(parents)[::-1]

        for ix_parent in parents:
            parent = tree.nodes[ix_parent]

            # get size of each child
            kid_size = {k: len(tree.nodes[k].members) for k in parent.children}

            # count children larger than 'threshold'
            n_bigkid = sum(_np.array(kid_size.values()) >= threshold)

            if n_bigkid == 0:
                # update parent's end level and end mass
                parent.end_level = max([tree.nodes[k].end_level
                                        for k in parent.children])
                parent.end_mass = max([tree.nodes[k].end_mass
                                       for k in parent.children])

                # remove small kids from the tree
                for k in parent.children:
                    del tree.nodes[k]
                parent.children = []

            elif n_bigkid == 1:
                pass
                # identify the big kid
                ix_bigkid = [k for k, v in kid_size.iteritems()
                             if v >= threshold][0]
                bigkid = tree.nodes[ix_bigkid]

                # update k's end level and end mass
                parent.end_level = bigkid.end_level
                parent.end_mass = bigkid.end_mass

                # set grandkids' parent to k
                for c in bigkid.children:
                    tree.nodes[c].parent = ix_parent

                # delete small kids
                for k in parent.children:
                    if k != ix_bigkid:
                        del tree.nodes[k]

                # set k's children to grandkids
                parent.children = bigkid.children

                # delete the single big kid
                del tree.nodes[ix_bigkid]

            else:
                pass  # do nothing here

        return tree

    def _leaf_cluster(self):
        """
        Set every leaf node as a foreground cluster.

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

        leaves = self.get_leaf_nodes()

        ## find components in the leaves
        points = []
        cluster = []

        for leaf in leaves:
            points.extend(self.nodes[leaf].members)
            cluster += ([leaf] * len(self.nodes[leaf].members))

        labels = _np.array([points, cluster], dtype=_np.int).T
        return labels

    def _first_K_cluster(self, k):
        """
        Returns foreground cluster labels for the 'k' modes with the lowest
        start levels. In principle, this is the 'k' leaf nodes with the
        smallest indices, but this function double checks by finding and
        ordering all leaf start values and ordering.

        Parameters
        ----------
        k : int
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

        parents = _np.array([u for u, v in self.nodes.items()
                             if len(v.children) > 0])
        roots = [u for u, v in self.nodes.items() if v.parent is None]
        splits = [self.nodes[u].end_level for u in parents]
        order = _np.argsort(splits)
        star_parents = parents[order[:(k - len(roots))]]

        children = [u for u, v in self.nodes.items() if v.parent is None]
        for u in star_parents:
            children += self.nodes[u].children

        nodes = [x for x in children if
                 sum(_np.in1d(self.nodes[x].children, children)) == 0]

        points = []
        cluster = []

        for c in nodes:
            cluster_pts = self.nodes[c].members
            points.extend(cluster_pts)
            cluster += ([c] * len(cluster_pts))

        labels = _np.array([points, cluster], dtype=_np.int).T
        return labels

    def _upper_set_cluster(self, threshold, form='mass'):
        """
        Set foreground clusters by finding connected components at an upper
        level set or upper mass set.

        Parameters
        ----------
        threshold : float
            The level or mass value that defines the foreground set of points,
            depending on 'form'.

        form : {'mass', 'density'}
            Determines if the 'cut' threshold is a density level value or a
            mass value (i.e. fraction of data in the background set)

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
        if form == 'mass':
            density_level = self._mass_to_density(mass=threshold)
            return self._upper_set_cluster(threshold=density_level,
                                           form='density')

        else:
            upper_level_set = _np.where(_np.array(self.density) > threshold)[0]
            active_nodes = [k for k, v in self.nodes.iteritems()
                            if (v.start_level <= threshold and
                                v.end_level > threshold)]

            ## find intersection between upper set points and each active
            #  component
            points = []
            cluster = []

            for c in active_nodes:
                cluster_mask = _np.in1d(upper_level_set,
                                        list(self.nodes[c].members))
                cluster_pts = upper_level_set[cluster_mask]
                points.extend(cluster_pts)
                cluster += ([c] * len(cluster_pts))

            labels = _np.array([points, cluster], dtype=_np.int).T
            return labels

    def _first_K_level_cluster(self, k):
        """
        Use the first K clusters to appear in the level set tree as foreground
        clusters. In general, K-1 clusters will appear at a lower level than
        the K'th cluster; this function returns all members from all K clusters
        (rather than only the members in the upper level set where the K'th
        cluster appears). There are not always K clusters available in a level
        set tree - see LevelSetTree.findKCut for details on default behavior in
        this case.

        Parameters
        ----------
        k : int
            Desired number of clusters.

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

        cut = self._find_K_cut(k)
        nodes = [e for e, v in self.nodes.iteritems()
                 if v.start_level <= cut and v.end_level > cut]

        points = []
        cluster = []

        for c in nodes:

            cluster_pts = self.nodes[c].members
            points.extend(cluster_pts)
            cluster += ([c] * len(cluster_pts))

        labels = _np.array([points, cluster], dtype=_np.int).T
        return labels

    def _collapse_leaves(self, active_nodes):
        """
        Removes descendant nodes for the branches in 'active_nodes'.

        Parameters
        ----------
        active_nodes : array-like
            List of nodes to use as the leaves in the collapsed tree.

        Returns
        -------
        """

        for ix in active_nodes:
            subtree = self._make_subtree(ix)

            max_end_level = max([v.end_level for v in subtree.nodes.values()])
            max_end_mass = max([v.end_mass for v in subtree.nodes.values()])

            self.nodes[ix].end_level = max_end_level
            self.nodes[ix].end_mass = max_end_mass
            self.nodes[ix].children = []

            for u in subtree.nodes.keys():
                if u != ix:
                    del self.nodes[u]

    def _find_K_cut(self, k):
        """
        Find the lowest level cut that has k connected components. If there are
        no levels that have k components, then find the lowest level that has
        at least k components. If no levels have > k components, find the
        lowest level that has the maximum number of components.

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
        crits = _np.unique(starts + ends)
        nclust = {}

        for c in crits:
            nclust[c] = len([e for e, v in self.nodes.iteritems()
                             if v.start_level <= c and v.end_level > c])

        width = _np.max(nclust.values())

        if k in nclust.values():
            cut = _np.min([e for e, v in nclust.iteritems() if v == k])
        else:
            if width < k:
                cut = _np.min([e for e, v in nclust.iteritems() if v == width])
            else:
                ktemp = _np.min([v for v in nclust.itervalues() if v > k])
                cut = _np.min([e for e, v in nclust.iteritems() if v == ktemp])

        return cut

    def _construct_branch_map(self, ix, interval, form, horizontal_spacing,
                              sort):
        """
        Map level set tree nodes to locations in a plot canvas. Finds the plot
        coordinates of vertical line segments corresponding to LST nodes and
        horizontal line segments corresponding to node splits. Also provides
        indices of vertical segments and splits for downstream use with
        interactive plot picker tools. This function is not meant to be called
        by the user; it is a helper function for the LevelSetTree.plot()
        method. This function is recursive: it calls itself to map the
        coordinates of children of the current node 'ix'.

        Parameters
        ----------
        ix : int
            The tree node to map.

        interval: length 2 tuple of floats
            Horizontal space allocated to node 'ix'.

        form : {'density', 'mass'}, optional

        horizontal_spacing : {'uniform', 'proportional'}, optional
            Determines how much horizontal space each level set tree node is
            given. See LevelSetTree.plot() for more information.

        sort : bool
            If True, sort sibling nodes from most to least points and draw left
            to right. Also sorts root nodes in the same way.

        Returns
        -------
        segments : dict
            A dictionary with values that contain the coordinates of vertical
            line segment endpoints. This is only useful to the interactive
            analysis tools.

        segmap : list
            Indicates the order of the vertical line segments as returned by
            the recursive coordinate mapping function, so they can be picked by
            the user in the interactive tools.

        splits : dict
            Dictionary values contain the coordinates of horizontal line
            segments (i.e. node splits).

        splitmap : list
            Indicates the order of horizontal line segments returned by
            recursive coordinate mapping function, for use with interactive
            tools.
        """

        ## get children
        children = _np.array(self.nodes[ix].children)
        n_child = len(children)

        ## if there's no children, just one segment at the interval mean
        if n_child == 0:
            xpos = _np.mean(interval)
            segments = {}
            segmap = [ix]
            splits = {}
            splitmap = []

            if form == 'density':
                segments[ix] = (
                    ([xpos, self.nodes[ix].start_level],
                     [xpos, self.nodes[ix].end_level]))
            else:
                segments[ix] = (
                    ([xpos, self.nodes[ix].start_mass],
                     [xpos, self.nodes[ix].end_mass]))

        ## else, construct child branches then figure out parent's
        ## position
        else:
            parent_range = interval[1] - interval[0]
            segments = {}
            segmap = [ix]
            splits = {}
            splitmap = []

            census = _np.array([len(self.nodes[x].members) for x in children],
                               dtype=_np.float)
            weights = census / sum(census)

            if sort is True:
                seniority = _np.argsort(weights)[::-1]
                children = children[seniority]
                weights = weights[seniority]

            ## get relative branch intervals
            if horizontal_spacing == 'uniform':
                child_intervals = _np.cumsum(weights)
                child_intervals = _np.insert(child_intervals, 0, 0.0)

            elif horizontal_spacing == 'proportional':
                child_intervals = _np.linspace(0.0, 1.0, n_child + 1)

            else:
                raise ValueError("'horizontal_spacing' argument not " +
                                 "understood. 'horizontal_spacing' " +
                                 "must be either 'uniform' or 'proportional'.")

            ## loop over the children
            for j, child in enumerate(children):

                ## translate local interval to absolute interval
                branch_interval = (
                    interval[0] + child_intervals[j] * parent_range,
                    interval[0] + child_intervals[j + 1] * parent_range)

                ## recurse on the child
                branch = self._construct_branch_map(child, branch_interval,
                                                    form, horizontal_spacing,
                                                    sort)
                branch_segs, branch_splits, branch_segmap, \
                    branch_splitmap = branch

                segmap += branch_segmap
                splitmap += branch_splitmap
                splits = dict(splits.items() + branch_splits.items())
                segments = dict(segments.items() + branch_segs.items())

            ## find the middle of the children's x-position and make vertical
            #  segment ix
            children_xpos = _np.array([segments[k][0][0] for k in children])
            xpos = _np.mean(children_xpos)

            ## add horizontal segments to the list
            for child in children:
                splitmap.append(child)
                child_xpos = segments[child][0][0]

                if form == 'density':
                    splits[child] = (
                        [xpos, self.nodes[ix].end_level],
                        [child_xpos, self.nodes[ix].end_level])
                else:
                    splits[child] = (
                        [xpos, self.nodes[ix].end_mass],
                        [child_xpos, self.nodes[ix].end_mass])

            ## add vertical segment for current node
            if form == 'density':
                segments[ix] = (
                    ([xpos, self.nodes[ix].start_level],
                     [xpos, self.nodes[ix].end_level]))
            else:
                segments[ix] = (
                    ([xpos, self.nodes[ix].start_mass],
                     [xpos, self.nodes[ix].end_mass]))

        return segments, splits, segmap, splitmap

    def _construct_mass_map(self, ix, start_pile, interval,
                            horizontal_spacing):
        """
        Map level set tree nodes to locations in a plot canvas. Finds the plot
        coordinates of vertical line segments corresponding to LST nodes and
        horizontal line segments corresponding to node splits. Also provides
        indices of vertical segments and splits for downstream use with
        interactive plot picker tools. This function is not meant to be called
        by the user; it is a helper function for the LevelSetTree.plot()
        method. This function is recursive: it calls itself to map the
        coordinates of children of the current node 'ix'. Differs from
        'constructBranchMap' by setting the height of each vertical segment to
        be proportional to the number of points in the corresponding LST node.

        Parameters
        ----------
        ix : int
            The tree node to map.

        start_pile: float
            The height of the branch on the plot at it's start (i.e. lower
            terminus).

        interval: length 2 tuple of floats
            Horizontal space allocated to node 'ix'.

        horizontal_spacing : {'uniform', 'proportional'}, optional
            Determines how much horizontal space each level set tree node is
            given. See LevelSetTree.plot() for more information.

        Returns
        -------
        segments : dict
            A dictionary with values that contain the coordinates of vertical
            line segment endpoints. This is only useful to the interactive
            analysis tools.

        segmap : list
            Indicates the order of the vertical line segments as returned by
            the recursive coordinate mapping function, so they can be picked by
            the user in the interactive tools.

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
        children = _np.array(self.nodes[ix].children)
        n_child = len(children)

        ## if there's no children, just one segment at the interval mean
        if n_child == 0:
            xpos = _np.mean(interval)
            end_pile = start_pile + size / len(self.density)
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

            census = _np.array([len(self.nodes[x].members) for x in children],
                               dtype=_np.float)
            weights = census / sum(census)

            seniority = _np.argsort(weights)[::-1]
            children = children[seniority]
            weights = weights[seniority]

            ## get relative branch intervals
            if horizontal_spacing == 'proportional':
                child_intervals = _np.cumsum(weights)
                child_intervals = _np.insert(child_intervals, 0, 0.0)
            else:
                child_intervals = _np.linspace(0.0, 1.0, n_child + 1)

            ## find height of the branch
            end_pile = start_pile + (size - sum(census)) / len(self.density)

            ## loop over the children
            for j, child in enumerate(children):

                ## translate local interval to absolute interval
                branch_interval = (
                    interval[0] + child_intervals[j] * parent_range,
                    interval[0] + child_intervals[j + 1] * parent_range)

                ## recurse on the child
                branch = self._construct_mass_map(child, end_pile,
                                                  branch_interval,
                                                  horizontal_spacing)
                branch_segs, branch_splits, branch_segmap, \
                    branch_splitmap = branch

                segmap += branch_segmap
                splitmap += branch_splitmap
                splits = dict(splits.items() + branch_splits.items())
                segments = dict(segments.items() + branch_segs.items())

            ## find the middle of the children's x-position and make vertical
            ## segment ix
            children_xpos = _np.array([segments[k][0][0] for k in children])
            xpos = _np.mean(children_xpos)

            ## add horizontal segments to the list
            for child in children:
                splitmap.append(child)
                child_xpos = segments[child][0][0]
                splits[child] = ([xpos, end_pile], [child_xpos, end_pile])

            ## add vertical segment for current node
            segments[ix] = ([xpos, start_pile], [xpos, end_pile])

        return segments, splits, segmap, splitmap

    def _mass_to_density(self, mass):
        """
        Convert the specified mass level value into a density level value.

        Parameters
        ----------
        mass : float
            Float in the interval [0.0, 1.0], with the desired fraction of all
            points.

        Returns
        -------
        level: float
            Density level corresponding to the 'mass' fraction of background
            points.
        """
        density_order = _np.argsort(self.density)
        n = len(self.density)
        mass_fraction = max(0, int(round(mass * n)) - 1)
        level_index = density_order[mass_fraction]
        level = self.density[level_index]

        return level


#############################################
### LEVEL SET TREE CONSTRUCTION FUNCTIONS ###
#############################################
def construct_tree(X, k, prune_threshold=None, num_levels=None, verbose=False):
    """
    Construct a level set tree from tabular data.

    Parameters
    ----------
    X : 2-dimensional numpy array
        Numeric dataset, where each row represents one observation.

    k : int
        Number of observations to consider as neighbors to a given point.

    prune_threshold : int, optional
        Leaf nodes with fewer than this number of members are recursively
        merged into larger nodes. If 'None' (the default), then no pruning
        is performed.

    num_levels : int, optional
        Number of density levels in the constructed tree. If None (default),
        `num_levels` is internally set to be the number of rows in `X`.

    verbose : bool, optional
        If True, a progress indicator is printed at every 100th level of tree
        construction.

    Returns
    -------
    T : LevelSetTree
        A pruned level set tree.

    See Also
    --------
    construct_tree_from_graph, LevelSetTree

    Examples
    --------
    >>> X = numpy.random.rand(100, 2)
    >>> tree = debacl.construct_tree(X, k=8, prune_threshold=5)
    >>> print tree
    +----+-------------+-----------+------------+----------+------+--------+----------+
    | id | start_level | end_level | start_mass | end_mass | size | parent | children |
    +----+-------------+-----------+------------+----------+------+--------+----------+
    | 0  |    0.000    |   0.870   |   0.000    |  0.450   | 100  |  None  |  [3, 4]  |
    | 3  |    0.870    |   3.364   |   0.450    |  0.990   |  17  |   0    |    []    |
    | 4  |    0.870    |   1.027   |   0.450    |  0.520   |  35  |   0    |  [7, 8]  |
    | 7  |    1.027    |   1.755   |   0.520    |  0.870   |  8   |   4    |    []    |
    | 8  |    1.027    |   3.392   |   0.520    |  1.000   |  23  |   4    |    []    |
    +----+-------------+-----------+------------+----------+------+--------+----------+
    """

    sim_graph, radii = _utl.knn_graph(X, k, method='brute_force')

    n, p = X.shape
    density = _utl.knn_density(radii, n, p, k)

    tree = construct_tree_from_graph(adjacency_list=sim_graph, density=density,
                                     prune_threshold=prune_threshold,
                                     num_levels=num_levels, verbose=verbose)

    return tree


def construct_tree_from_graph(adjacency_list, density, prune_threshold=None,
                              num_levels=None, verbose=False):
    """
    Construct a level set tree from a similarity graph and a density estimate.

    Parameters
    ----------
    adjacency_list : list [list]
        Adjacency list of the k-nearest neighbors graph on the data. Each entry
        contains the indices of the `k` closest neighbors to the data point at
        the same row index.

    density : list [float]
        Estimate of the density function, evaluated at the data points
        represented by the keys in `adjacency_list`.

    prune_threshold : int, optional
        Leaf nodes with fewer than this number of members are recursively
        merged into larger nodes. If 'None' (the default), then no pruning
        is performed.

    num_levels : list int, optional
        Number of density levels in the constructed tree. If None (default),
        `num_levels` is internally set to be the number of rows in `X`.

    verbose : bool, optional
        If True, a progress indicator is printed at every 100th level of tree
        construction.

    Returns
    -------
    T : levelSetTree
        See the LevelSetTree class for attributes and method definitions.

    See Also
    --------
    construct_tree, LevelSetTree

    Examples
    --------
    >>> X = numpy.random.rand(100, 2)
    >>> knn_graph, radii = debacl.utils.knn_graph(X, k=8)
    >>> density = debacl.utils.knn_density(radii, n=100, p=2, k=8)
    >>> tree = debacl.construct_tree_from_graph(knn_graph, density,
    ...                                         prune_threshold=5)
    >>> print tree
    +----+-------------+-----------+------------+----------+------+--------+----------+
    | id | start_level | end_level | start_mass | end_mass | size | parent | children |
    +----+-------------+-----------+------------+----------+------+--------+----------+
    | 0  |    0.000    |   0.768   |   0.000    |  0.390   | 100  |  None  |  [1, 2]  |
    | 1  |    0.768    |   1.494   |   0.390    |  0.790   |  30  |   0    |  [7, 8]  |
    | 2  |    0.768    |   4.812   |   0.390    |  1.000   |  31  |   0    |    []    |
    | 7  |    1.494    |   2.375   |   0.790    |  0.950   |  6   |   1    |    []    |
    | 8  |    1.494    |   2.308   |   0.790    |  0.940   |  5   |   1    |    []    |
    +----+-------------+-----------+------------+----------+------+--------+----------+
    """

    ## Initialize the graph and cluster tree
    levels = _utl.define_density_mass_grid(density, num_levels=num_levels)

    G = _nx.from_dict_of_lists(
        {i: neighbors for i, neighbors in enumerate(adjacency_list)})

    T = LevelSetTree(density, levels)

    ## Figure out roots of the tree
    cc0 = _nx.connected_components(G)

    for i, c in enumerate(cc0):  # c is only the vertex list, not the subgraph
        T._subgraphs[i] = G.subgraph(c)
        T.nodes[i] = ConnectedComponent(
            i, parent=None, children=[], start_level=0., end_level=None,
            start_mass=0., end_mass=None, members=c)

    # Loop through the removal grid
    previous_level = 0.
    n = float(len(adjacency_list))

    for i, level in enumerate(levels):
        if verbose and i % 100 == 0:
            _logging.info("iteration {}".format(i))

        ## figure out which points to remove, i.e. the background set.
        bg = _np.where((density > previous_level) & (density <= level))[0]
        previous_level = level

        ## compute the mass after the current bg set is removed
        old_vcount = sum([x.number_of_nodes()
                          for x in T._subgraphs.itervalues()])
        current_mass = 1. - ((old_vcount - len(bg)) / n)

        # loop through active components, i.e. subgraphs
        deactivate_keys = []     # subgraphs to deactivate at the iter end
        activate_subgraphs = {}  # new subgraphs to add at the end of the iter

        for (k, H) in T._subgraphs.iteritems():

            ## remove nodes at the current level
            H.remove_nodes_from(bg)

            ## check if subgraph has vanished
            if H.number_of_nodes() == 0:
                T.nodes[k].end_level = level
                T.nodes[k].end_mass = current_mass
                deactivate_keys.append(k)

            else:  # subgraph hasn't vanished

                ## check if subgraph now has multiple connected components
                # NOTE: this is *the* bottleneck
                if not _nx.is_connected(H):

                    ## deactivate the parent subgraph
                    T.nodes[k].end_level = level
                    T.nodes[k].end_mass = current_mass
                    deactivate_keys.append(k)

                    ## start a new subgraph & node for each child component
                    cc = _nx.connected_components(H)

                    for c in cc:
                        new_key = max(T.nodes.keys()) + 1
                        T.nodes[k].children.append(new_key)
                        activate_subgraphs[new_key] = H.subgraph(c)

                        T.nodes[new_key] = ConnectedComponent(
                            new_key, parent=k, children=[], start_level=level,
                            end_level=None, start_mass=current_mass,
                            end_mass=None, members=c)

        # update active components
        for k in deactivate_keys:
            del T._subgraphs[k]

        T._subgraphs.update(activate_subgraphs)

    ## Prune the tree
    if prune_threshold is not None:
        T = T.prune(threshold=prune_threshold)

    return T


def load_tree(filename):
    """
    Load a saved tree from file.

    Parameters
    ----------
    filename : str
        Filename to load.

    Returns
    -------
    T : LevelSetTree
        The loaded and reconstituted level set tree object.

    See Also
    --------
    LevelSetTree.save

    Examples
    --------
    >>> X = numpy.random.rand(100, 2)
    >>> tree = debacl.construct_tree(X, k=8, prune_threshold=5)
    >>> tree.save('my_tree')
    >>> tree2 = debacl.load_tree('my_tree')
    """
    with open(filename, 'rb') as f:
        T = _cPickle.load(f)

    return T
