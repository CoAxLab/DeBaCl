
import unittest
import scipy.special as spspec
import numpy as np
from numpy.testing import assert_array_equal

from debacl import utils as utl


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
        r_k = 1.
        n = 100
        p = 2
        k = 5.

        # Correct density estimate
        unit_ball_volume = np.pi**(p/2.) / spspec.gamma(1 + p/2.0)
        normalizer = k / (n * unit_ball_volume)
        answer = normalizer / (r_k**p)
        
        # DeBaCl knn density utility
        fhat = utl.knn_density(r_k, n, p, k)

        self.assertEqual(fhat, answer)


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


class TestDensityGrid(unittest.TestCase):
    """
    Test class for the utility function that defines the 1D grid of density
    levels, upon which a level set tree is estimated.
    """

    def setUp(self):
        self.n = 10
        
        self.unique_density = np.arange(self.n) + 1
        np.random.shuffle(self.unique_density)

        self.uniform_density = np.array([1.] * self.n)

    def test_bogus_input(self):
        """
        Check that inputs are validated correctly.
        """

        ## Check the 'mode' parameter.
        with self.assertRaises(ValueError):
            levels = utl.define_density_grid(self.unique_density, mode='fossa')

        ## Check form of the density input.
        with self.assertRaises(ValueError):
            levels = utl.define_density_grid([])            

        with self.assertRaises(TypeError):
            levels = utl.define_density_grid(density='fossa')

        ## Check the 'num_levels' parameter.
        with self.assertRaises(ValueError):
            levels = utl.define_density_grid(self.unique_density,
                                             num_levels=-1)

        with self.assertRaises(ValueError):
            levels = utl.define_density_grid(self.unique_density,
                                             num_levels=0)

        with self.assertRaises(TypeError):
            levels = utl.define_density_grid(self.unique_density,
                                             num_levels=2.17)

        with self.assertRaises(TypeError):
            levels = utl.define_density_grid(self.unique_density,
                                             num_levels='fossa')

    def test_mass_grid(self):
        """
        Check that the mass-based grid is constructed correctly for a toy
        density function.
        """
        ## Test typical input - should be sorted
        levels = utl.define_density_grid(self.unique_density, mode='mass')
        answer = np.sort(self.unique_density)
        assert_array_equal(answer, levels)

        ## Test more levels than density values (answer is the same as typical
        #  input).
        levels = utl.define_density_grid(self.unique_density, mode='mass', 
                                         num_levels=self.n * 2)
        assert_array_equal(answer, levels)

        ## Test fewer levels than density values.
        levels = utl.define_density_grid(self.unique_density, mode='mass',
                                         num_levels=2)
        answer = np.array([5, 10])
        assert_array_equal(answer, levels)

        ## Test negative values.
        neg_function = np.hstack((self.unique_density, [-1., -2.]))
        levels = utl.define_density_grid(neg_function, mode='mass')
        answer = np.sort(neg_function)
        assert_array_equal(answer, levels)

        ## Test uniform input.
        levels = utl.define_density_grid(self.uniform_density, mode='mass')

        import ipdb
        ipdb.set_trace()


        ## Test for off-by-one errors - make sure the min and the max values
        #  are covered.


    def test_level_grid(self):
        """
        Check that the level-based grid is constructed correctly for a toy
        density function.
        """
        pass


class TestBackgroundAssignments(unittest.TestCase):
    """
    """

    def setUp(self):
        pass

    def test_zero_method(self):
        """
        """
        pass

    def test_k_plus_one(self):
        """
        """
        pass

    def test_knn_classifier(self):
        """
        """
        pass

    def test_nearest_center(self):
        """
        """
        pass