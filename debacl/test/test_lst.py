
from __future__ import print_function as _print_function
from __future__ import absolute_import as _absolute_import

import os
import unittest
import tempfile
import numpy as np
import debacl as dcl
from numpy.testing import assert_array_equal


class TestLSTConstructors(unittest.TestCase):
    """
    Test that the various level set tree constructors correctly create or load
    a LST.
    """

    def setUp(self):
        """
        Create a dataset from a mixture of three univariate Gaussian
        distributions.
        """
        ## Data parameters
        np.random.seed(451)
        self.n = 1000
        mix = (0.3, 0.5, 0.2)
        mean = (-1, 0, 1)
        stdev = (0.3, 0.2, 0.1)

        ## Tree parameters
        self.k = 50
        self.gamma = 5

        ## Simulate data
        membership = np.random.multinomial(self.n, pvals=mix)
        self.dataset = np.array([], dtype=np.float)

        for (p, mu, sigma) in zip(membership, mean, stdev):
            draw = np.random.normal(loc=mu, scale=sigma, size=p)
            self.dataset = np.append(self.dataset, draw)

        self.dataset = np.sort(self.dataset).reshape((self.n, 1))

        ## Compute a similarity graph and density estimate
        self.knn_graph, radius = dcl.utils.knn_graph(self.dataset, self.k,
                                                     'kd-tree')
        self.density = dcl.utils.knn_density(radius, self.n, p=1, k=self.k)

    def _check_tree_viability(self, tree):
        """
        Generic function for testing the viability of any tree. Does not check
        correctness with respect to any particular dataset, similarity graph,
        or density function.
        """

        ## Check that the type of the tree is LevelSetTree
        self.assertTrue(isinstance(tree, dcl.level_set_tree.LevelSetTree))

        ## Check min and max density levels
        start_levels = [x.start_level for x in tree.nodes.values()]
        self.assertAlmostEqual(min(start_levels), 0.)

        end_levels = [x.end_level for x in tree.nodes.values()]
        self.assertAlmostEqual(max(end_levels), max(self.density))

        ## Check min and max masses
        start_masses = [x.start_mass for x in tree.nodes.values()]
        self.assertAlmostEqual(min(start_masses), 0.)

        end_masses = [x.end_mass for x in tree.nodes.values()]
        self.assertAlmostEqual(max(end_masses), 1.)

        ## Sum of root sizes should be the size of the input dataset.
        root_sizes = [len(node.members) for node in tree.nodes.values()
                      if node.parent is None]
        self.assertEqual(sum(root_sizes), self.n)

        ## Check per-node plausibility
        for idx, node in tree.nodes.items():

            if idx == 0:
                self.assertTrue(node.parent is None)

            else:
                if node.parent is not None:  # there *can* be multiple roots
                    self.assertTrue(isinstance(node.parent, int))
                    self.assertTrue(node.parent >= 0)

            ## any node with 1 member has no children
            if len(node.members) == 1:
                self.assertEqual(len(node.children), 0)

            # start and end levels and masses should be in the right order,
            # relative to each other
            self.assertLess(node.start_level, node.end_level)
            self.assertLess(node.start_mass, node.end_mass)

            ## no node should have size no smaller than the pruning threshold.
            self.assertLessEqual(self.gamma, len(node.members))

            if len(node.children) > 0:

                ## children sizes sum to less than parent size
                kid_sizes = [len(tree.nodes[x].members) for x in node.children]
                self.assertLess(sum(kid_sizes), len(node.members))

                for child in node.children:

                    ## child nodes have higher indices than parent
                    self.assertGreater(child, idx)

                    ## child nodes start where parent nodes end
                    self.assertEqual(node.end_level,
                                     tree.nodes[child].start_level)

                    self.assertEqual(node.end_mass,
                                     tree.nodes[child].start_mass)

    def _check_tree_correctness(self, tree):
        """
        Check correctness of the specific LST estimated according to the
        simulated dataset and LST parameters in the setup of this class.
        """

        ## Density levels
        start_levels = [round(node.start_level, 3)
                        for node in tree.nodes.values()]
        end_levels = [round(node.end_level, 3)
                      for node in tree.nodes.values()]

        ans_start_levels = [0.0, 0.104, 0.104, 0.189, 0.189, 0.345, 0.345,
                            0.741, 0.741, 0.862, 0.862]
        ans_end_levels = [0.104, 0.189, 1.001, 0.345, 0.741, 0.508, 0.381,
                          0.862, 0.804, 1.349, 1.004]

        self.assertItemsEqual(start_levels, ans_start_levels)
        self.assertItemsEqual(end_levels, ans_end_levels)

        ## Masses
        start_masses = [round(node.start_mass, 3)
                        for node in tree.nodes.values()]
        end_masses = [round(node.end_mass, 3)
                      for node in tree.nodes.values()]

        ans_start_masses = [0.0, 0.018, 0.018, 0.079, 0.079, 0.293, 0.293,
                            0.667, 0.667, 0.816, 0.816]
        ans_end_masses = [0.018, 0.079, 0.947, 0.293, 0.667, 0.473, 0.359,
                          0.816, 0.734, 1.0, 0.949]

        self.assertItemsEqual(start_masses, ans_start_masses)
        self.assertItemsEqual(end_masses, ans_end_masses)

        ## Sizes and parents
        sizes = [len(node.members) for node in tree.nodes.values()]
        parents = [node.parent for node in tree.nodes.values()]

        ans_sizes = [1000, 767, 215, 238, 472, 76, 23, 231, 12, 103, 48]
        ans_parents = [None, 0, 0, 1, 1, 3, 3, 4, 4, 7, 7]

        self.assertItemsEqual(sizes, ans_sizes)
        self.assertItemsEqual(parents, ans_parents)

    def test_construct_from_graph(self):
        """
        Check viability and correctness of LSTs constructed from a similarity
        graph.
        """
        tree = dcl.construct_tree_from_graph(self.knn_graph, self.density,
                                             prune_threshold=self.gamma)

        self._check_tree_viability(tree)
        self._check_tree_correctness(tree)

    def test_construct_from_data(self):
        """
        Check viability and correctness of an LST constructed directly from a
        tabular dataset.
        """
        tree = dcl.construct_tree(self.dataset, self.k,
                                  prune_threshold=self.gamma)

        self._check_tree_viability(tree)
        self._check_tree_correctness(tree)

    def test_load(self):
        """
        Check viability and correctness of an LST saved then loaded from file.
        """
        tree = dcl.construct_tree(self.dataset, self.k,
                                  prune_threshold=self.gamma)

        with tempfile.NamedTemporaryFile() as f:
            tree.save(f.name)
            tree2 = dcl.load_tree(f.name)

        self._check_tree_viability(tree2)
        self._check_tree_correctness(tree)


class TestBackwardCompatibility(unittest.TestCase):
    """
    Make sure models from previous versions of DeBaCl still load in the
    current version.
    """

    def test_old_trees(self):
        """
        Make sure models from previous versions of DeBaCl still load in the
        current version.
        """
        old_tree_dir = os.path.join(os.path.dirname(__file__), 'saved_trees')
        print("old tree dir: {}".format(old_tree_dir))

        for f in os.listdir(old_tree_dir):

            ## Load the tree.
            tree = dcl.load_tree(os.path.join(old_tree_dir, f))

            ## Is it a level set tree?
            self.assertIsInstance(tree, dcl.LevelSetTree)

            ## Make sure all methods run ok.
            # printing
            print(tree)
            for c in ('start_level', 'end_level', 'start_mass', 'end_mass',
                      'size', 'parent', 'children'):
                self.assertTrue(c in tree.__str__())

            # pruning
            old_gamma = tree.prune_threshold
            new_gamma = 2 * tree.prune_threshold
            pruned_tree = tree.prune(new_gamma)
            self.assertEqual(tree.prune_threshold, old_gamma)
            self.assertEqual(pruned_tree.prune_threshold, new_gamma)

            # plotting
            # plot = tree.plot()  # Travis CI doesn't like this.

            # retrieve clusters.
            labels = tree.get_clusters()
            n = len(tree.density)

            self.assertTrue(labels.dtype is np.dtype('int64'))
            self.assertLessEqual(len(labels), n)
            self.assertEqual(len(labels[:, 0]), len(np.unique(labels[:, 0])))
            self.assertGreaterEqual(np.min(labels), 0)

            max_row_index = np.max(labels[:, 0])
            self.assertLessEqual(max_row_index, n)

            # retrieve leaf indices
            leaves = tree.get_leaf_nodes()
            self.assertIsInstance(leaves, list)
            self.assertGreater(len(leaves), 0)

            # full partition
            partition = tree.branch_partition()

            self.assertTrue(partition.dtype is np.dtype('int64'))
            self.assertEqual(len(partition), n)
            self.assertEqual(len(partition[:, 0]),
                             len(np.unique(partition[:, 0])))
            self.assertEqual(np.min(partition[:, 0]), 0)
            self.assertItemsEqual(np.unique(partition[:, 1]),
                                  tree.nodes.keys())

            # save and load
            with tempfile.NamedTemporaryFile() as t:
                tree.save(t.name)
                tree2 = dcl.load_tree(t.name)

                self.assertItemsEqual(
                    [x.start_level for x in tree.nodes.values()],
                    [y.start_level for y in tree2.nodes.values()])


class TestLevelSetTree(unittest.TestCase):
    """
    Test the functions of a level set tree, *after* it's been created.

    Note that `LevelSetTree.save` is already tested by the `load_tree`
    constructor test above.
    """

    def setUp(self):
        ## Data parameters
        np.random.seed(451)
        self.n = 1000
        mix = (0.3, 0.5, 0.2)
        mean = (-1, 0, 1)
        stdev = (0.3, 0.2, 0.1)

        ## Tree parameters
        k = 50
        self.gamma = 5

        ## Simulate data
        membership = np.random.multinomial(self.n, pvals=mix)
        dataset = np.array([], dtype=np.float)

        for (p, mu, sigma) in zip(membership, mean, stdev):
            draw = np.random.normal(loc=mu, scale=sigma, size=p)
            dataset = np.append(dataset, draw)

        dataset = np.sort(dataset).reshape((self.n, 1))
        self.tree = dcl.construct_tree(dataset, k, self.gamma)

    def _check_tree_viability(self, tree):
        """
        Generic function for testing the viability of any tree. Does not check
        correctness with respect to any particular dataset, similarity graph,
        or density function.
        """

        ## Check that the type of the tree is LevelSetTree
        self.assertTrue(isinstance(tree, dcl.level_set_tree.LevelSetTree))

        ## Check min and max density levels
        start_levels = [x.start_level for x in tree.nodes.values()]
        self.assertAlmostEqual(min(start_levels), 0.)

        ## Check min and max masses
        start_masses = [x.start_mass for x in tree.nodes.values()]
        self.assertAlmostEqual(min(start_masses), 0.)

        end_masses = [x.end_mass for x in tree.nodes.values()]
        self.assertAlmostEqual(max(end_masses), 1.)

        ## Sum of root sizes should be the size of the input dataset.
        root_sizes = [len(node.members) for node in tree.nodes.values()
                      if node.parent is None]
        self.assertEqual(sum(root_sizes), self.n)

        ## Check per-node plausibility
        for idx, node in tree.nodes.items():

            if idx == 0:
                self.assertTrue(node.parent is None)

            else:
                if node.parent is not None:  # there *can* be multiple roots
                    self.assertTrue(isinstance(node.parent, int))
                    self.assertTrue(node.parent >= 0)

            ## any node with 1 member has no children
            if len(node.members) == 1:
                self.assertEqual(len(node.children), 0)

            # start and end levels and masses should be in the right order,
            # relative to each other
            self.assertLess(node.start_level, node.end_level)
            self.assertLess(node.start_mass, node.end_mass)

            ## no node should have size no smaller than the pruning threshold.
            self.assertLessEqual(self.gamma, len(node.members))

            if len(node.children) > 0:

                ## children sizes sum to less than parent size
                kid_sizes = [len(tree.nodes[x].members) for x in node.children]
                self.assertLess(sum(kid_sizes), len(node.members))

                for child in node.children:

                    ## child nodes have higher indices than parent
                    self.assertGreater(child, idx)

                    ## child nodes start where parent nodes end
                    self.assertEqual(node.end_level,
                                     tree.nodes[child].start_level)

                    self.assertEqual(node.end_mass,
                                     tree.nodes[child].start_mass)

    def test_summaries(self):
        """
        Check that tree printing in table form is working properly.
        """
        print

        try:
            self.tree
        except:
            self.assertTrue(False, "LevelSetTree __repr__ failed.")

        try:
            print(self.tree)
        except:
            self.assertTrue(False, "LevelSetTree failed to print.")

        print_string = self.tree.__str__()

        self.assertFalse('fossa' in print_string)

        column_names = ['start_level', 'end_level', 'start_mass', 'end_mass',
                        'size', 'parent', 'children']
        for col_name in column_names:
            self.assertTrue(col_name in print_string)

    def test_prune(self):
        """
        Test LevelSetTree pruning.
        """
        ## Double check prune threshold for the original tree.
        self.assertEqual(self.tree.prune_threshold, self.gamma)

        ## Check the prune threshold for a new tree, and make sure the
        #  threshold didn't change for the original tree.
        new_gamma = 50
        pruned_tree = self.tree.prune(threshold=new_gamma)
        self.assertEqual(pruned_tree.prune_threshold, new_gamma)
        self.assertEqual(self.tree.prune_threshold, self.gamma)

        ## Make sure the new tree is still a viable tree.
        self._check_tree_viability(self.tree)
        self._check_tree_viability(pruned_tree)

    def _check_cluster_label_plausibility(self, labels, background=False):
        """
        Utility for checking whether cluster labels conform to minimal
        standards of plausibility. Does *not* check correctness of cluster
        labels relative to any particular dataset, similarity graph, or density
        function.
        """
        # all values should be integers
        self.assertTrue(labels.dtype is np.dtype('int64'))

        # fewer labels than data instances
        self.assertLessEqual(len(labels), self.n)

        # all row indices are less than the number of indices
        max_row_index = np.max(labels[:, 0])
        self.assertLessEqual(max_row_index, self.n)

        # no duplicate row indices
        self.assertEqual(len(labels[:, 0]), len(np.unique(labels[:, 0])))

        # row indices and cluster labels are integers greater than or equal to
        # zero, or -1 if background is True
        if background:
            self.assertGreaterEqual(np.min(labels), -1)
        else:
            self.assertGreaterEqual(np.min(labels), 0)

    def test_clusters(self):
        """
        Test the various ways of getting clusters from a LevelSetTree.
        """

        ## Bogus input
        with self.assertRaises(ValueError):
            labels = self.tree.get_clusters(method='fossa')

        ## First-K clusters
        k = 3
        labels = self.tree.get_clusters(method='first-k', k=k)
        self._check_cluster_label_plausibility(labels)
        self.assertTrue(len(np.unique(labels[:, 1])), k)

        ## First level with K clusters
        labels = self.tree.get_clusters(method='k-level', k=k)
        self._check_cluster_label_plausibility(labels)
        self.assertTrue(len(np.unique(labels[:, 1])), k)

        ## Upper set clustering
        labels = self.tree.get_clusters(method='upper-level-set',
                                        threshold=0.4,
                                        form='density')
        self._check_cluster_label_plausibility(labels)

        labels = self.tree.get_clusters(method='upper-level-set',
                                        threshold=0.6,
                                        form='mass')
        self._check_cluster_label_plausibility(labels)

        ## Leaf clustering
        leaf_labels = self.tree.get_clusters(method='leaf')
        self._check_cluster_label_plausibility(leaf_labels)

        leaves = [idx for idx, node in self.tree.nodes.items()
                  if len(node.children) == 0]

        self.assertItemsEqual(np.unique(leaf_labels[:, 1]), leaves)

        ## Check that background filling works correctly.
        full_labels = self.tree.get_clusters(method='leaf',
                                             fill_background=True)
        self._check_cluster_label_plausibility(full_labels, background=True)
        assert_array_equal(leaf_labels[:, 1],
                           full_labels[leaf_labels[:, 0], 1])

    def test_leaf_node_getter(self):
        """
        Test that the nodes returned by the leaf node getter are actually
        leaves and that all leaves are returned correctly.
        """
        leaves = self.tree.get_leaf_nodes()
        answer = [2, 5, 6, 8, 9, 10]
        self.assertEqual(leaves, answer)

    def test_branch_partition(self):
        """
        Test that the full data partition based on branch membership works
        correctly.
        """
        partition = self.tree.branch_partition()

        # all values should be integers
        self.assertTrue(partition.dtype is np.dtype('int64'))

        # a label for each data instance
        self.assertEqual(len(partition), self.n)

        # no duplicate row indices
        self.assertEqual(len(partition[:, 0]), len(np.unique(partition[:, 0])))

        # row indices should start at 0
        self.assertEqual(np.min(partition[:, 0]), 0)

        # Labels should match tree nodes exactly.
        self.assertItemsEqual(np.unique(partition[:, 1]),
                              self.tree.nodes.keys())
