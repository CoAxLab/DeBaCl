
from __future__ import print_function as _print_function
from __future__ import absolute_import as _absolute_import

import unittest
import scipy.special as spspec
import numpy as np
from numpy.testing import assert_array_equal

import debacl.utils as utl


class TestDensityEstimates(unittest.TestCase):
    """
    Unit test class for density estimate functions in DeBaCl utilities.
    """

    def test_knn_density(self):
        """
        Test correctness of the knn density estimator in the DeBaCl utilities.
        Tests the estimated value at a single arbitrary point.
        """

        # Input parameters
        r_k = np.array([1.])
        n = 100
        p = 2
        k = 5.

        # Correct density estimate
        unit_ball_volume = np.pi**(p / 2.) / spspec.gamma(1 + p / 2.0)
        normalizer = k / (n * unit_ball_volume)
        answer = normalizer / (r_k**p)

        # DeBaCl knn density utility
        fhat = utl.knn_density(r_k, n, p, k)
        self.assertEqual(fhat, answer)

        ## Check that undefined density estimates raise an error.
        with self.assertRaises(ArithmeticError):
            r_k = np.array([10., 10.])
            fhat = utl.knn_density(r_k, n=1000, p=350, k=10)


class TestSimilarityGraphs(unittest.TestCase):
    """
    Unit test class for neighbor graphs. Use very simple stylized data so the
    correct similarity graph is known.
    """

    def setUp(self):

        ## Make data, evenly spaced in a single dimensionself.
        n = 5
        self.X = np.arange(n).reshape((n, 1))

    def test_knn_graph(self):
        """
        Test construction of the k-nearest neighbor graph.
        """
        k = 3

        ## Correct knn similarity graph
        ans_radii = np.array([2., 1., 1., 1., 2.])
        ans_graph = np.array([[0, 1, 2],
                              [1, 0, 2],
                              [2, 1, 3],
                              [3, 2, 4],
                              [4, 3, 2]])

        ## DeBaCl knn similarity graph
        for method in ['brute_force', 'kd_tree', 'ball_tree']:

            knn, radii = utl.knn_graph(self.X, k=k, method=method)

            ## Test
            assert_array_equal(radii, ans_radii)

            for neighbors, ans_neighbors in zip(knn, ans_graph):
                self.assertItemsEqual(neighbors, ans_neighbors)

    def test_epsilon_graph(self):
        """
        Test construction of the epsilon-nearest neighbor graph.
        """
        epsilon = 1.01

        ## Correct similarity graph
        ans_graph = [np.array([0, 1]),
                     np.array([1, 0, 2]),
                     np.array([2, 1, 3]),
                     np.array([3, 2, 4]),
                     np.array([4, 3])]

        enn = utl.epsilon_graph(self.X, epsilon)

        for neighbors, ans_neighbors in zip(enn, ans_graph):
            self.assertItemsEqual(neighbors, ans_neighbors)


class TestDensityGrids(unittest.TestCase):
    """
    Test class for the utility functions that define the 1D grid of density
    levels, upon which a level set tree is estimated.
    """

    def setUp(self):
        self.n = 10

        self.unique_density = np.arange(self.n) + 1
        np.random.shuffle(self.unique_density)

        self.generic_array = np.hstack((self.unique_density, [-1., -2.]))
        self.uniform_density = np.array([1.] * self.n)

    def _check_bogus_input(self, grid_func):
        """
        Check that inputs are validated correctly for a given density grid
        function.

        Parameters
        ----------
        grid_func : function
            Density grid function.
        """

        ## Check form of the density input.
        with self.assertRaises(ValueError):
            grid_func([])

        with self.assertRaises(TypeError):
            grid_func(density='fossa')

        ## Check the 'num_levels' parameter.
        with self.assertRaises(ValueError):
            grid_func(self.unique_density, num_levels=-1)

        with self.assertRaises(ValueError):
            grid_func(self.unique_density, num_levels=1)

        with self.assertRaises(TypeError):
            grid_func(self.unique_density, num_levels=2.17)

        with self.assertRaises(TypeError):
            grid_func(self.unique_density, num_levels='fossa')

    def test_bogus_input(self):
        """
        Check that each of the density grid functions trap input errors.
        """
        self._check_bogus_input(utl.define_density_level_grid)
        self._check_bogus_input(utl.define_density_mass_grid)

    def test_mass_grid(self):
        """
        Check that the mass-based grid is constructed correctly.
        """
        ## Test typical input - should be sorted
        levels = utl.define_density_mass_grid(self.unique_density)
        answer = np.sort(self.unique_density)
        assert_array_equal(answer, levels)

        ## Test more levels than density values (answer is the same as typical
        #  input).
        levels = utl.define_density_mass_grid(self.unique_density,
                                              num_levels=self.n * 2)
        assert_array_equal(answer, levels)

        ## Test fewer levels than density values.
        levels = utl.define_density_mass_grid(self.unique_density,
                                              num_levels=2)
        answer = np.array([1, 10])
        assert_array_equal(answer, levels)

        ## Test negative values.
        levels = utl.define_density_mass_grid(self.generic_array)
        answer = np.sort(self.generic_array)
        assert_array_equal(answer, levels)

        ## Test uniform input.
        levels = utl.define_density_mass_grid(self.uniform_density)
        self.assertItemsEqual(levels, [1.])

    def _check_level_grid_answer(self, density, levels):
        """
        Utility to check correctness of a num_levels=n density level grid.

        Parameters
        ----------
        density : numpy array
            Input density values.

        levels : numpy array
            Values to check.
        """
        self.assertTrue(np.min(density) in levels)
        self.assertTrue(np.max(density) in levels)
        self.assertEqual(len(levels), len(density))
        assert_array_equal(levels, np.sort(levels))

    def test_level_grid(self):
        """
        Check that the level-based grid is constructed correctly.
        """
        ## Typical input should include the right number of values, sorted,
        #  between min and max.
        levels = utl.define_density_level_grid(self.unique_density)
        self._check_level_grid_answer(self.unique_density, levels)

        ## More levels than density values should yield the same answer.
        levels = utl.define_density_level_grid(self.unique_density,
                                               num_levels=self.n * 2)
        self._check_level_grid_answer(self.unique_density, levels)

        ## 2 density levels should just be min and max
        levels = utl.define_density_level_grid(self.unique_density,
                                               num_levels=2)
        answer = np.array([1, 10])
        assert_array_equal(answer, levels)

        ## Negative values should still range from min and max, sorted, with
        #  the right number of values.
        levels = utl.define_density_level_grid(self.generic_array,
                                               num_levels=self.n * 2)
        self._check_level_grid_answer(self.generic_array, levels)

        ## Uniform input should have a single value.
        levels = utl.define_density_level_grid(self.uniform_density)
        self.assertItemsEqual(levels, [1.])


class TestClusterReindexing(unittest.TestCase):
    """
    Make sure the cluster label reindexing works properly.
    """

    def setUp(self):
        self.cluster_labels = np.array([[7, 6, 8, 11, 62],
                                       [3, 3, 7, 7, 4]]).T

    def test_label_reindexing(self):
        """
        Test cluster label reindexing.
        """
        new_labels = utl.reindex_cluster_labels(self.cluster_labels)
        assert_array_equal(self.cluster_labels[:, 0], new_labels[:, 0])
        assert_array_equal(new_labels[:, 1], np.array([0, 0, 2, 2, 1]))
