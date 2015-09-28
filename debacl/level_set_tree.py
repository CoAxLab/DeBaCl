"""
Main functions and classes for the DEnsity-BAsed CLustering (DeBaCl) toolbox.
Includes functions to construct and modify level set trees produced by standard
geometric clustering on each level. Also defines tools for interactive data
analysis and clustering with level set trees.
"""

import cPickle
import utils as utl

try:
    import numpy as np
    import networkx as nx
    from prettytable import PrettyTable
except:
    raise ImportError("DeBaCl requires the numpy, networkx, and " +
        "prettytable packages for level set tree estimation and printing.")

try:
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.widgets import Button
    _HAS_MPL = True
except:
    _HAS_MPL = False
    print "Matplotlib could not be loaded, so level set tree plots will fail."


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

    def copy(self):
        """
        Create and return a copy of a ConnetedComponent object.

        Returns
        -------
        component : ConnectedComponent
        """

        return ConnectedComponent(self.idnum, self.parent, self.children,
            self.start_level, self.end_level, self.start_mass, self.end_mass,
            self.members)


class LevelSetTree(object):
    """
    The level set tree. The level set tree is a collection of connected
    components organized hierarchically, based on a k-nearest neighbors density
    estimate and connectivity graph.

    Parameters
    ----------
    bg_sets : list of lists
        The observations removed as background points at each successively
        higher density level.

    levels : array_like
        The probability density level associated with each element in 'bg_sets'.
    """

    def __init__(self, density=[], level_grid=[]):
        self.density = density
        self.level_grid = level_grid
        self.num_levels = 0
        self.nodes = {}
        self.subgraphs = {}

    def __str__(self):
        """
        Print the tree summary table.
        """
        summary = PrettyTable(["id", "lambda1", "lambda2", "alpha1",
                              "alpha2", "size", "parent", "children"])
        for node_id, v in self.nodes.items():
            summary.add_row([node_id,
                             v.start_level,
                             v.end_level,
                             v.start_mass,
                             v.end_mass,
                             len(v.members),
                             v.parent,
                             v.children])

        for col in ['lambda1', 'lambda2', 'alpha1', 'alpha2']:
            summary.float_format[col] = "5.3"

        return summary.get_string()

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
                self._merge_by_size(gamma)

        else:
            print "Pruning method not understood. 'size-merge' is the only " +\
            "pruning method currently implemented. No changes were made to " + \
            "the tree."

    def save(self, filename):
        """
        Save a level set tree object to file.

        Serialize all members of a level set tree with the cPickle module and
        save to file.

        Parameters
        ----------
        filename : string
            File to save the tree to. The filename extension does not matter for
            this method (although operating system requirements still apply).
        """
        with open(filename, 'wb') as f:
            cPickle.dump(self, f, cPickle.HIGHEST_PROTOCOL)

    def plot(self, form, width='uniform', sort=True, gap=0.05, color_nodes=None):
        """
        Create a level set tree plot in Matplotlib.

        Parameters
        ----------
        form : {'lambda', 'alpha', 'kappa', 'old'}
            Determines main form of the plot. 'lambda' is the traditional plot
            where the vertical scale is density levels, but plot improvements
            such as mass sorting of the nodes and colored nodes are allowed and
            the secondary 'alpha' scale is visible (but not controlling). The
            'old' form uses density levels for vertical scale but does not allow
            plot tweaks and does not show the secondary 'alpha' scale. The
            'alpha' setting makes the uppper level set mass the primary vertical
            scale, leaving the 'lambda' scale in place for reference. 'kappa'
            makes node mass the vertical scale, so that each node's vertical
            height is proportional to its mass excluding the mass of the node's
            children.

        width : {'uniform', 'mass'}, optional
            Determines how much horzontal space each level set tree node is
            given. The default of "uniform" gives each child node an equal
            fraction of the parent node's horizontal space. If set to 'mass',
            then horizontal space is allocated proportional to the mass (i.e.
            fraction of points) of a node relative to its siblings.

        sort : bool, optional
            If True, sort sibling nodes from most to least points and draw left
            to right. Also sorts root nodes in the same way.

        gap : float, optional
            Fraction of vertical space to leave at the bottom. Default is 5%,
            and 0% also works well. Higher values are used for interactive tools
            to make room for buttons and messages.

        color_nodes : list, optional
            Each entry should be a valid index in the level set tree that will
            be colored uniquely.

        Returns
        -------
        fig : matplotlib figure
            Use fig.show() to view, fig.savefig() to save, etc.

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

        ## Validate input
        if form == 'old':
            sort = False
            color_nodes = None
            width = 'uniform'


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

        if sort is True:
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
            if form == 'kappa':
                branch = self._construct_mass_map(ix, 0.0, (intervals[i],
                    intervals[i+1]), width)
            elif form == 'old':
                branch = self._construct_branch_map(ix, (intervals[i],
                    intervals[i+1]), 'lambda', width, sort)
            else:
                branch = self._construct_branch_map(ix, (intervals[i],
                    intervals[i+1]), form, width, sort)

            branch_segs, branch_splits, branch_segmap, branch_splitmap = branch
            segments = dict(segments.items() + branch_segs.items())
            splits = dict(splits.items() + branch_splits.items())
            segmap += branch_segmap
            splitmap += branch_splitmap


        ## get the the vertical line segments in order of the segment map (segmap)
        verts = [segments[k] for k in segmap]
        lats = [splits[k] for k in splitmap]


        ## Find the fraction of nodes in each segment (to use as linewidths)
        thickness = [max(1.0, 12.0 * len(self.nodes[x].members)/n)
            for x in segmap]


        ## Get the relevant vertical ticks
        primary_ticks = [(x[0][1], x[1][1]) for x in segments.values()]
        primary_ticks = np.unique(np.array(primary_ticks).flatten())
        primary_labels = [str(round(tick, 2)) for tick in primary_ticks]


        ## Set up the plot framework
        fig, ax = plt.subplots()
        ax.set_position([0.11, 0.05, 0.78, 0.93])
        ax.set_xlim((-0.04, 1.04))
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.yaxis.grid(color='gray')
        ax.set_yticks(primary_ticks)
        ax.set_yticklabels(primary_labels)


        ## Form-specific details
        if form == 'kappa':
            kappa_max = max(primary_ticks)
            ax.set_ylim((-1.0 * gap * kappa_max, 1.04*kappa_max))
            ax.set_ylabel("mass")

        elif form == 'old':
            ax.set_ylabel("lambda")
            ymin = min([v.start_level for v in self.nodes.itervalues()])
            ymax = max([v.end_level for v in self.nodes.itervalues()])
            rng = ymax - ymin
            ax.set_ylim(ymin - gap*rng, ymax + 0.05*rng)

        elif form == 'lambda':
            ax.set_ylabel("lambda")
            ymin = min([v.start_level for v in self.nodes.itervalues()])
            ymax = max([v.end_level for v in self.nodes.itervalues()])
            rng = ymax - ymin
            ax.set_ylim(ymin - gap*rng, ymax + 0.05*rng)

            ax2 = ax.twinx()
            ax2.set_position([0.11, 0.05, 0.78, 0.93])
            ax2.set_ylabel("alpha", rotation=270)

            alpha_ticks = np.sort(list(set(
                [v.start_mass for v in self.nodes.itervalues()] + \
                [v.end_mass for v in self.nodes.itervalues()])))
            alpha_labels = [str(round(m, 2)) for m in alpha_ticks]

            ax2.set_yticks(primary_ticks)
            ax2.set_yticklabels(alpha_labels)
            ax2.set_ylim(ax.get_ylim())

        elif form == 'alpha':
            ax.set_ylabel("alpha")
            ymin = min([v.start_mass for v in self.nodes.itervalues()])
            ymax = max([v.end_mass for v in self.nodes.itervalues()])
            rng = ymax - ymin
            ax.set_ylim(ymin - gap*rng, ymax + 0.05*ymax)

            ax2 = ax.twinx()
            ax2.set_position([0.11, 0.05, 0.78, 0.93])
            ax2.set_ylabel("lambda", rotation=270)

            lambda_ticks = np.sort(list(set(
                [v.start_level for v in self.nodes.itervalues()] + \
                [v.end_level for v in self.nodes.itervalues()])))
            lambda_labels = [str(round(lvl, 2)) for lvl in lambda_ticks]

            ax2.set_ylim(ax.get_ylim())
            ax2.set_yticks(primary_ticks)
            ax2.set_yticklabels(lambda_labels)

        else:
            raise ValueError('Plot form not understood')


        ## Add the line segments
        segclr = np.array([[0.0, 0.0, 0.0]] * len(segmap))
        splitclr = np.array([[0.0, 0.0, 0.0]] * len(splitmap))

        palette = dbc_plot.Palette()
        if color_nodes is not None:
            for i, ix in enumerate(color_nodes):
                n_clr = np.alen(palette.colorset)
                c = palette.colorset[i % n_clr, :]
                subtree = self.make_subtree(ix)

                ## set verical colors
                ix_replace = np.in1d(segmap, subtree.nodes.keys())
                segclr[ix_replace] = c

                ## set horizontal colors
                if splitmap:
                    ix_replace = np.in1d(splitmap, subtree.nodes.keys())
                    splitclr[ix_replace] = c

        linecol = LineCollection(verts, linewidths=thickness, colors=segclr)
        ax.add_collection(linecol)
        linecol.set_picker(20)

        splitcol = LineCollection(lats, colors=splitclr)
        ax.add_collection(splitcol)

        return fig, segments, segmap, splits, splitmap

    def get_cluster_labels(self, method='all-mode', **kwargs):
        """
        Generic function for retrieving custer labels from the level set tree.
        Dispatches a specific cluster labeling function.

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

        scale : {'lambda', 'alpha'}
            If method is 'upper-set', this is vertical scale which 'threshold'
            refers to. 'lambda' corresponds to a density level, 'alpha'
            corresponds to a mass level.

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
            labels, nodes = self._all_mode_cluster()

        elif method == 'first-k':
            required = set(['k'])
            if not set(kwargs.keys()).issuperset(required):
                raise ValueError("Incorrect arguments for the first-k " + \
                "cluster labeling method.")
            else:
                k = kwargs.get('k')
                labels, nodes = self._first_K_cluster(k)

        elif method == 'upper-set':
            required = set(['threshold', 'scale'])
            if not set(kwargs.keys()).issuperset(required):
                raise ValueError("Incorrect arguments for the upper-set " + \
                "cluster labeling method.")
            else:
                threshold = kwargs.get('threshold')
                scale = kwargs.get('scale')
                labels, nodes = self._upper_set_cluster(threshold, scale)

        elif method == 'k-level':
            required = set(['k'])
            if not set(kwargs.keys()).issuperset(required):
                raise ValueError("Incorrect arguments for the k-level " + \
                "cluster labeling method.")
            else:
                k = kwargs.get('k')
                labels, nodes = self._first_K_level_cluster(k)

        else:
            print 'method not understood'
            labels = np.array([])
            nodes = []

        return labels, nodes

    def make_subtree(self, ix):
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

    def _merge_by_size(self, threshold):
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
            root_tree = self.make_subtree(root)
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
                parent.end_level = max([self.nodes[k].end_level
                    for k in parent.children])
                parent.end_mass = max([self.nodes[k].end_mass
                    for k in parent.children])

                # remove small kids from the tree
                for k in parent.children:
                    del self.nodes[k]
                parent.children = []

            elif n_bigkid == 1:
                pass
                # identify the big kid
                ix_bigkid = [k for k, v in kid_size.iteritems()
                    if v >= threshold][0]
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

    def _all_mode_cluster(self):
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

    def _first_K_cluster(self, k):
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
        splits = [self.nodes[u].end_level for u in parents]
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

    def _upper_set_cluster(self, threshold, scale='alpha'):
        """
        Set foreground clusters by finding connected components at an upper
        level set or upper mass set.

        Parameters
        ----------
        threshold : float
            The level or mass value that defines the foreground set of points,
            depending on 'scale'.

        scale : {'alpha', 'lambda'}
            Determines if the 'cut' threshold is a density level value or a mass
            value (i.e. fraction of data in the background set)

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
        if scale == 'alpha':
            n_bg = [len(x) for x in self.bg_sets]
            alphas = np.cumsum(n_bg) / (1.0 * self.n)
            upper_levels = np.where(alphas > threshold)[0]
            nodes = [k for k, v in self.nodes.iteritems()
                if v.start_mass <= threshold and v.end_mass > threshold]

        else:
            upper_levels = np.where(np.array(self.levels) > threshold)[0]
            nodes = [k for k, v in self.nodes.iteritems()
                if v.start_level <= threshold and v.end_level > threshold]

        upper_pts = np.array([y for x in upper_levels for y in self.bg_sets[x]])


        ## find intersection between upper set points and each active component
        points = []
        cluster = []

        for i, c in enumerate(nodes):
            cluster_pts = upper_pts[np.in1d(upper_pts, self.nodes[c].members)]
            points.extend(cluster_pts)
            cluster += ([i] * len(cluster_pts))

        labels = np.array([points, cluster], dtype=np.int).T
        return labels, nodes

    def _first_K_level_cluster(self, k):
        """
        Use the first K clusters to appear in the level set tree as foreground
        clusters. In general, K-1 clusters will appear at a lower level than the
        K'th cluster; this function returns all members from all K clusters
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

        cut = self.findKCut(k)
        nodes = [e for e, v in self.nodes.iteritems() \
            if v.start_level <= cut and v.end_level > cut]

        points = []
        cluster = []

        for i, c in enumerate(nodes):

            cluster_pts = self.nodes[c].members
            points.extend(cluster_pts)
            cluster += ([i] * len(cluster_pts))

        labels = np.array([points, cluster], dtype=np.int).T
        return labels, nodes

    def _collapse_leaves(self, active_nodes):
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
            subtree = self.make_subtree(ix)

            max_end_level = max([v.end_level for v in subtree.nodes.values()])
            max_end_mass = max([v.end_mass for v in subtree.nodes.values()])

            self.nodes[ix].end_level = max_end_level
            self.nodes[ix].end_mass = max_end_mass
            self.nodes[ix].children = []

            for u in subtree.nodes.keys():
                if u != ix:
                    del self.nodes[u]

    def find_K_cut(self, k):
        """
        Find the lowest level cut that has k connected components. If there are
        no levels that have k components, then find the lowest level that has at
        least k components. If no levels have > k components, find the lowest
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

    def _construct_branch_map(self, ix, interval, scale, width, sort):
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

        scale : {'lambda', 'alpha'}, optional

        width : {'uniform', 'mass'}, optional
            Determines how much horzontal space each level set tree node is
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

            if scale == 'lambda':
                segments[ix] = (([xpos, self.nodes[ix].start_level],
                    [xpos, self.nodes[ix].end_level]))
            else:
                segments[ix] = (([xpos, self.nodes[ix].start_mass],
                    [xpos, self.nodes[ix].end_mass]))


        ## else, construct child branches then figure out parent's
        ## position
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
            if width == 'mass':
                child_intervals = np.cumsum(weights)
                child_intervals = np.insert(child_intervals, 0, 0.0)

            elif width == 'uniform':
                child_intervals = np.linspace(0.0, 1.0, n_child+1)

            else:
                raise ValueError("'width' argument not understood. 'width' " +
                                 "must be either 'mass' or 'uniform'.")

            ## loop over the children
            for j, child in enumerate(children):

                ## translate local interval to absolute interval
                branch_interval = (interval[0] + \
                    child_intervals[j] * parent_range,
                    interval[0] + child_intervals[j+1] * parent_range)

                ## recurse on the child
                branch = self._construct_branch_map(child, branch_interval,
                    scale, width, sort)
                branch_segs, branch_splits, branch_segmap, \
                    branch_splitmap = branch

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

                if scale == 'lambda':
                    splits[child] = ([xpos, self.nodes[ix].end_level],
                        [child_xpos, self.nodes[ix].end_level])
                else:
                    splits[child] = ([xpos, self.nodes[ix].end_mass],
                        [child_xpos, self.nodes[ix].end_mass])


            ## add vertical segment for current node
            if scale == 'lambda':
                segments[ix] = (([xpos, self.nodes[ix].start_level],
                    [xpos, self.nodes[ix].end_level]))
            else:
                segments[ix] = (([xpos, self.nodes[ix].start_mass],
                    [xpos, self.nodes[ix].end_mass]))

        return segments, splits, segmap, splitmap

    def _construct_mass_map(self, ix, start_pile, interval, width_mode):
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
            end_pile = start_pile + size/len(self.density)
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
            end_pile = start_pile + (size - sum(census))/len(self.density)

            ## loop over the children
            for j, child in enumerate(children):

                ## translate local interval to absolute interval
                branch_interval = (interval[0] +
                    child_intervals[j] * parent_range, interval[0] +
                    child_intervals[j+1] * parent_range)

                ## recurse on the child
                branch = self._construct_mass_map(child, end_pile,
                                                  branch_interval, width_mode)
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

    def _mass_to_level(self, alpha):
        """
        Convert the specified mass value into a level location.

        Parameters
        ----------
        alpha : float
            Float in the interval [0.0, 1.0], with the desired fraction of all
            points.

        Returns
        -------
        cut_level: float
            Density level corresponding to the 'alpha' fraction of background
            points.
        """

        n_bg = [len(bg_set) for bg_set in self.bg_sets]
        masses = np.cumsum(n_bg) / (1.0 * len(self.density))
        cut_level = self.levels[np.where(masses > alpha)[0][0]]

        return cut_level



#############################################
### LEVEL SET TREE CONSTRUCTION FUNCTIONS ###
#############################################

def construct_tree(adjacency_list, density, level_grid=None,
                   prune_threshold=None, verbose=False):
    """
    Construct a level set tree. A level set tree is constructed by identifying
    connected components of in a k-nearest neighbors graph at successively
    higher levels of a probability density estimate.

    Parameters
    ----------
    adjacency_list : list [list]
        Adjacency list of the k-nearest neighbors graph on the data. Each entry
        contains the indices of the `k` closest neighbors to the data point at
        the same row index.

    density : list [float]
        Estimate of the density function, evaluated at the data points
        represented by the keys in `adjacency_list`.

    level_grid : list [float], optional
        Density levels at which connected components are computed. If not
        specified, this is all unique values of a probability density estimate,
        but it can be a coarser grid for fast approximate tree estimates. The
        utility function `define_density_grid` can be used to construct a custom
        `level_grid`.

    prune_threshold : int, optional
        Leaf nodes with fewer than this number of members are recursively merged
        into larger siblings. If 'None' (the default), then no pruning is
        performed.

    verbose : bool, optional
        If set to True, then prints to the screen a progress indicator every 100
        levels.

    Returns
    -------
    T : levelSetTree
        See the LevelSetTree class for attributes and method definitions.
    """

    ## Determine density levels (if not provided) and background sets.
    if level_grid is None:
        level_grid = utl.define_density_grid(density, mode='levels',
                                             num_levels=None)


    ## Initialize the graph and cluster tree
    # num_levels = len(level_grid)
    # levels = [float(x) for x in density_levels]
    G = nx.from_dict_of_lists({i: neighbors for i, neighbors in enumerate(adjacency_list)})
    T = LevelSetTree(density, level_grid)


    ## Figure out roots of the tree
    cc0 = nx.connected_components(G)

    for i, c in enumerate(cc0):  # c is only the vertex list, not the subgraph
        T.subgraphs[i] = G.subgraph(c)
        T.nodes[i] = ConnectedComponent(i, parent=None, children=[],
            start_level=0., end_level=None, start_mass=0., end_mass=None,
            members=c)


    # Loop through the removal grid
    previous_level = 0.
    n = float(len(adjacency_list))

    for i, level in enumerate(level_grid):
        if verbose and i % 100 == 0:
            print "iteration", i

        ## figure out which points to remove, i.e. the background set.
        bg = np.where((density > previous_level) & (density <= level))[0]
        previous_level = level

        ## compute the mass after the current bg set is removed
        old_vcount = sum([x.number_of_nodes() for x in T.subgraphs.itervalues()])
        current_mass = 1. - ((old_vcount - len(bg)) / n)

        # loop through active components, i.e. subgraphs
        deactivate_keys = []     # subgraphs to deactivate at the end of the iter
        activate_subgraphs = {}  # new subgraphs to add at the end of the iter

        for (k, H) in T.subgraphs.iteritems():

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
                if not nx.is_connected(H):

                    ## deactivate the parent subgraph
                    T.nodes[k].end_level = level
                    T.nodes[k].end_mass = current_mass
                    deactivate_keys.append(k)

                    ## start a new subgraph & node for each child component
                    cc = nx.connected_components(H)

                    for c in cc:
                        new_key = max(T.nodes.keys()) + 1
                        T.nodes[k].children.append(new_key)
                        activate_subgraphs[new_key] = H.subgraph(c)

                        T.nodes[new_key] = ConnectedComponent(new_key,
                            parent=k, children=[], start_level=level,
                            end_level=None, start_mass=current_mass,
                            end_mass=None, members=c)

        # update active components
        for k in deactivate_keys:
            del T.subgraphs[k]

        T.subgraphs.update(activate_subgraphs)

    ## Prune the tree
    if prune_threshold is not None:
        T.prune(gamma=prune_threshold)

    return T

def load_tree(filename):
    """
    Load a saved tree from file.

    Parameters
    ----------
    filename : string
        Filename to load.

    Returns
    -------
    T : LevelSetTree
        The loaded and reconstituted level set tree object.
    """
    with open(filename, 'rb') as f:
        T = cPickle.load(f)

    return T
