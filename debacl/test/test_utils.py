
import unittest
import numpy as np
import scipy.special as spspec

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
    Unit test class for neighbor graphs.
    """

    def setUp(self):

        ## Make data
        n = 5
        self.X = np.arange(5).reshape((n, 1))

        ## Graph parameters
        self.k = 3
        self.epsilon = 1.01

        ## Answers
        self.knn = {
            0: set([0, 1, 2]),
            1: set([1, 0, 2]),
            2: set([2, 1, 3]),
            3: set([3, 2, 4]),
            4: set([4, 3, 2])}
        self.r_k = np.array([2., 1., 1., 1., 2.])

        self.eps_nn = {
            0: set([0, 1]),
            1: set([1, 0, 2]),
            2: set([2, 1, 3]),
            3: set([3, 2, 4]),
            4: set([4, 3])}

        self.edge_list = [(0, 1), (1, 2), (2, 3), (3, 4)]

    def test_knn_graph(self):
        """
        Test construction of the k-nearest neighbor graph.
        """

        import ipdb
        ipdb.set_trace()

        knn, r_k = utl.knn_graph(self.X, k=self.k, method='brute-force')
        np.testing.assert_array_equal(r_k, self.r_k)

        for idx, neighbors in knn.iteritems():
            self.assertSetEqual(self.knn[idx], set(neighbors))

    def test_epsilon_graph(self):
        """
        Test construction of the epsilon-nearest neighbor graph.
        """
        eps_nn = utl.epsilon_graph(self.X, self.epsilon)

        for idx, neighbors in eps_nn.iteritems():
            self.assertSetEqual(self.eps_nn[idx], set(neighbors))


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


class TestDensityGrid(unittest.TestCase):
    """
    """

    def setUp(self):
        """
        """
        pass

    def test_density_grid(self):
        """
        """
        pass