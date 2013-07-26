##############################################################
## Brian P. Kent
## gauss_demo.py
## Created: 20130207
## Updated: 20130725
## Demo key DeBaCl functions for 1D and 2D Gaussian mixture simulations.
##############################################################

##############
### SET UP ###
##############

### Import libraries
import sys
sys.path.append('..')  # shouldn't need this once installed from PyPI

from debacl import geom_tree as gtree
from debacl import utils as utl

import numpy as np
import matplotlib.pyplot as plt



###############################
### 1-D GAUSSIAN SIMULATION ###
###############################
### This example uses the polished function geomTree to build a level set tree
### in as straightforward a fashion as possible. The steps of density
### estimation, tree construction, and pruning are all wrapped into one
### function.


### General parameters
#n = 1500
#p_k = 0.02
#p_gamma = 0.05
#mix = (0.5, 0.3, 0.2)

#k = int(p_k * n)
#gamma = int(p_gamma * n)


### Data parameters
#ctr = ((1,), (6,), (11,))
#sdev = (np.eye(1),) * 3


### Generate data
#membership = np.random.multinomial(n, pvals=mix)
#p = len(ctr[0])
#X = np.zeros((n, p), dtype=np.float)
#g = np.zeros((n, ), dtype=np.int)
#b = np.cumsum((0,) + tuple(membership))

#for i, (size, mu, sigma) in enumerate(zip(membership, ctr, sdev)):
#	ix = range(b[i], b[i+1])
#	X[ix, :] = np.random.multivariate_normal(mu, sigma, size)
#	g[ix] = i

#X = np.sort(X, axis=0)  # sort the points for prettier downstream plotting


### Plot a histogram of the data to show the simulation worked
#fig, ax = plt.subplots()
#ax.hist(X, bins=n/20, normed=1, alpha=0.5)
#ax.set_xlabel('X')
#ax.set_ylabel('density')
#fig.show()


### Estimate the level set tree - the easy way
#tree = gtree.geomTree(X, k, gamma, n_grid=None, verbose=True)
#print tree


### Retrieve cluster assignments from the tree
#uc, leaves = tree.allModeCluster()
#print "cluster counts:", np.bincount(uc[:, 1])
#print "leaf indices:", leaves


### Plot the level set tree with colored leaves
### note the plot function returns a tuple with 5 objects. The first member of 
### tuple is the figure, which for most users is the only interesting part.
#fig = tree.plot(form='lambda', width='uniform', color_nodes=leaves)[0]
#fig.show()


### Assign background points
#fc = utl.assignBackgroundPoints(X.reshape((n, -1)), uc, method='knn', k=9)
#print "final cluster counts:", np.bincount(fc[:, 1])




###############################
### 2-D GAUSSIAN SIMULATION ###
###############################
### This example illustrates more advanced functions of the DeBaCl library. In
### this case we want to estimate the density with something other than the kNN
### estimator used in the geomTree convenience function, so we have to do each
### step manually. This example also illustrates tree saving and loading, as
### well as DeBaCl's interactive tools.


## General parameters
n = 1500
p_k = 0.005
p_gamma = 0.01
mix = (0.5, 0.3, 0.2)

k = int(p_k * n)
gamma = int(p_gamma * n)


## Data parameters
ctr = ((1, 5), (4, 5), (5, 1))
sdev = (0.5*np.eye(2),) * 3


## Generate data
membership = np.random.multinomial(n, pvals=mix)
p = len(ctr[0])
X = np.zeros((n, p), dtype=np.float)
g = np.zeros((n, ), dtype=np.int)
b = np.cumsum((0,) + tuple(membership))

for i, (size, mu, sigma) in enumerate(zip(membership, ctr, sdev)):
	ix = range(b[i], b[i+1])
	X[ix, :] = np.random.multivariate_normal(mu, sigma, size)
	g[ix] = i


## Scatterplot, to show the simulation worked
fig, ax = plt.subplots()
ax.scatter(X[:,0], X[:,1], s=50, c=g, alpha=0.4)
#fig.show()


## Construct the similarity graph and density estimate
W, k_radius = utl.knnGraph(X, k, self_edge=False)

import scipy.stats as spstat
kernel = spstat.gaussian_kde(X.T)
fhat = kernel(X.T)


## Construct the level set tree - note the speed-up by approximating the tree on
## a grid of 300 points (less than the )
bg_sets, levels = utl.constructDensityGrid(fhat, mode='mass', n_grid=300)
tree = gtree.constructTree(W, levels, bg_sets, mode='density', verbose=True)
print tree


## Save and/or load a tree (obviously redundant in this tutorial)
tree.save('test_tree')
tree = gtree.loadTree('test_tree')


## Prune the tree
tree.mergeBySize(gamma)
print tree



## Interactive tools
# Only uncomment and use one these at a time.
#tool = gtree.ComponentGUI(tree, X, form='alpha', width='mass', output=['scatter'])
tool = gtree.ClusterGUI(tree, X, form='kappa', width='mass', output=['scatter'])
tool.show()



#### Retrieve cluster assignments from the interactive tools manually
### This is especially useful for data with more than 3 dimensions, where the clusters
### (or single component) cannot be plotted directly by the interactive tools.
### Uncomment and paste the relevant code into the command line. Don't run the script
### straight through with these lines uncommented.
##uc = tool.getComponent()  # for TreeComponentTool
##uc = tool.getClusters()  # for TreeClusterTool




#### Plot upper level set clusters manually
### This does not work for 1D data.
#base = [217.0/255] * 3
#black = [0.0] * 3
#edge = plutl.makeColorMatrix(n, bg_color=black, bg_alpha=0.35, ix=None)
#fill = plutl.makeColorMatrix(n, bg_color=base, bg_alpha=0.72, ix=uc[:,0],
#	fg_color=uc[:,1], fg_alpha=0.68)
#fig = plutl.plotPoints(x, clr=fill, edgecolor=edge)
#fig.show()
