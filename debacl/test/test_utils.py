#####################################
## Brian P. Kent
## test_utils.py
## created: 20140529
## updated: 20140529
## Test the DeBaCl utility functions.
#####################################

import unittest
import numpy as np
import scipy.special as spspec

import sys
sys.path.insert(0, '/home/brian/Projects/debacl/DeBaCl/')
from debacl import utils as utl


## Example from the unittest introduction
# class TestSequenceFunctions(unittest.TestCase):

# 	def setUp(self):
# 		self.seq = range(10)

# 	def test_choice(self):
# 		element = random.choice(self.seq)
		# self.assertTrue(element in self.seq)



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
		fhat = utl.knnDensity(self.r_k, self.n, self.p, self.k)
		self.assertEqual(self.fhat, fhat)


class TestNeighborGraphs(unittest.TestCase):
	"""
	Unit test class for neighbor graphs.
	"""

	def setUp(self):
		pass

	def test_knn_graph(self):
		pass

	def test_epsilon_graph(self):
		pass

	def test_gaussian_graph(self):
		pass


class TestTreeConstructionUtils(unittest.TestCase):
	"""
	Unit test class for stages of level set tree construction.
	"""

	def setUp(self):
		pass

	def test_density_grid(self):
		pass

	def test_background_assignment(self):
		pass