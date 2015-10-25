
import unittest
import tempfile
import numpy as np
import debacl as dcl


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
        root_sizes = [len(node.members) for node in tree.nodes.values() if node.parent is None]
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
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(self.dataset, self.density)

        fig2 = tree.plot()[0]


        ## Single root
        roots = [idx for idx, node in tree.nodes.items() if node.parent is None]
        self.assertEqual(len(roots), 1)

        ## After pruning, three leaf nodes (for three Gaussian)
        pruned_tree = tree.prune(threshold=50)
        leaves = [idx for idx, node in pruned_tree.nodes.items() 
                  if len(node.children) == 0]
        self.assertEqual(len(leaves), 3)

        import ipdb
        ipdb.set_trace()


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


class TestLevelSetTree(unittest.TestCase):
    """
    Test the functions of a level set tree, *after* it's been created.
    """

    def setUp(self):
        pass

    def test_summaries(self):
        """
        """
        pass

    def test_save(self):
        """
        """
        pass

    def test_prune(self):
        """
        """
        pass

    def test_plot(self):
        """
        """
        pass

    def test_leaf_clusters(self):
        """
        """
        pass

    def test_first_k_clusters(self):
        """
        """
        pass

    def test_upper_set_clusters(self):
        """
        """
        pass

    def test_first_k_level_clusters(self):
        """
        """
        pass

    def test_cluster_dispatch(self):
        """
        """
        pass

    def test_k_cut_search(self):
        """
        """
        pass

    def test_subtree(self):
        """
        """
        pass
