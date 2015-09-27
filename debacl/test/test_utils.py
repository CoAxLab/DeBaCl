
import unittest
import numpy as np
import scipy.special as spspec

import sys
from debacl import utils as utl


class TestDensityEstimates(unittest.TestCase):
	"""
	Unit test class for density estimate functions in DeBaCl utilities.
	"""

	def setUp(self):

		# Input parameters
		self.r_k = 1.
		self.n = 100
		self.p = 2
		self.k = 5.

		# Correct density estimate
		unit_ball_volume = np.pi**(self.p/2.) / spspec.gamma(1 + self.p/2.0)
		normalizer = self.k / (self.n * unit_ball_volume)
		self.fhat = normalizer / (self.r_k**self.p)

	def test_knn_density(self):
		fhat = utl.knn_density(self.r_k, self.n, self.p, self.k)
		self.assertEqual(self.fhat, fhat)


class TestNeighborGraphs(unittest.TestCase):
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

	def test_type_conversions(self):
		"""
		Test conversion between graph representations.
		"""
		edge_list = utl.adjacency_to_edge_list(self.eps_nn, self_edge=False)
		edge_list = sorted([tuple(sorted(x)) for x in edge_list])

		for e, ans in zip(edge_list, self.edge_list):
			self.assertTupleEqual(e, ans)
