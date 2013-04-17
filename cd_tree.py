##############################################################
## Brian P. Kent
## cd_tree.py
## Created: 20130417
## Updated: 20130417
##############################################################

##############
### SET UP ###
##############
"""
Main functions and classes for construction and use of Chaudhuri-Dasgupta level set
trees. A companion to debacl.py, which has a more developed set of tools for working
with generic level set trees.
"""

import numpy as np
import pandas as pd
import scipy.spatial.distance as spdist
import scipy.io as spio
import igraph as igr
import matplotlib.pyplot as plt




#####################
### BASIC CLASSES ###
#####################
class CD_Component(object):
	"""
	Defines a connected component for level set tree construction. A level set tree is
	really just a set of ConnectedComponents.
	"""
	
	def __init__(self, idnum, parent, children, start_radius, end_radius, members):
		self.idnum = idnum
		self.parent = parent
		self.children = children
		self.start_radius = start_radius
		self.end_radius = end_radius
		self.members = members
		
		
		
		
class CD_Tree(object):
	"""
	Defines methods and attributes for a Chaudhuri-Dasgupta level set tree.
	"""
	
	def __init__(self):
		self.nodes = {}
		self.subgraphs = {}
		
		
		
#############################################
### LEVEL SET TREE CONSTRUCTION FUNCTIONS ###
#############################################

def makeCDTree(X, verbose=False):
	"""
	Construct a Chaudhuri-Dasgupta level set tree.
	
	A level set tree is constructed by identifying connected components of observations
	as edges are removed from the geometric graph in descending order of pairwise
	distance.
	
	Parameters
	----------
#	W : 2D array
#		An adjacency matrix for a similarity graph on the data.
#			
#	levels: array
#		Defines the density levels where connected components will be computed.
#		Typically this includes all unique values of a function evaluated on the data
#		points.
#	
#	bg_sets: list of lists
#		Specify which points to remove as background at each density level in 'levels'.
#	
#	mode: {'general', 'density'}, optional
#		Establish if the values in 'levels' come from a probability density or
#		pseudo-density estimate, in which case there is a natural floor at 0. The
#		default is to model an arbitrary function, which requires a run-time choice of
#		floor value.
#	
#	verbose: {False, True}, optional
#		If set to True, then prints to the screen a progress indicator every 100 levels.
#	
#	Returns
#	-------
#	T : levelSetTree
#		See debacl.levelSetTree for class and method definitions.
	"""
	
		
