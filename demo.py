##############################################################
## Brian P. Kent
## 1D_demo.py
## Created: 20130207
## Updated: 20130207
## Demo key DeBaCl functions for a 1D Gaussian mixture simulation.
##############################################################

##############
### SET UP ###
##############

### Import libraries
import debacl as dcl
import gen_utils as utl
import plot_utils as plutl

import numpy as np
np.set_printoptions(precision=4, edgeitems=5, suppress=True, linewidth=140)


### Set general parameters
n = 1500
p_k = 0.005
k = int(p_k * n)
p_prune = 0.03
mix = (0.5, 0.3, 0.2)


### Data parameters
## 1-dimensional
#ctr = ((1,), (6,), (11,))
#sdev = (np.eye(1),) * 3

## 2-dimensional
ctr = ((1, 5), (4, 5), (5, 1))
sdev = (0.5*np.eye(2),) * 3

## 3-dimensional
#ctr = ((1, 5, 1), (3, 5, 5), (5, 1, 3))
#sdev = (0.5*np.eye(3),) * 3


### Generate data
membership = np.random.multinomial(n, pvals=mix)
p = len(ctr[0])
x = np.zeros((n, p), dtype=np.float)
g = np.zeros((n, ), dtype=np.int)
b = np.cumsum((0,) + tuple(membership))

for i, (size, mu, sigma) in enumerate(zip(membership, ctr, sdev)):
	ix = range(b[i], b[i+1])
	x[ix, :] = np.random.multivariate_normal(mu, sigma, size)
	g[ix] = i


if p == 1:
	x = np.sort(x, axis=0)  # sort the points for prettier downstream plotting


### Plot the raw data, colored by group label
## Only works for 2D and 3D data. Otherwise, plot a histogram.
#fig = plutl.plotPoints(x, clr=g, alpha=0.7)
#fig.show()



#####################################
#### CONSTRUCT THE LEVEL SET TREE ###
#####################################

### Construct similarity graph
W, k_radius = utl.knnGraph(x, k)
#W, eps = utl.epsilonGraph(x, q=0.05)
np.fill_diagonal(W, False)


### Compute density estimate
## k-nearest neighbor density 
#fhat = utl.knnDensity(k_radius, n, p=1, k=k)

### any density estimator works - here's a kernel estimator with a gaussian kernel
## this one only works for 1D data
#from statsmodels.nonparametric.kde import KDE
#fhat_func = KDE(x.flatten())
#fhat_func.fit(bw=1.0)
#fhat = fhat_func.evaluate(x.flatten())

### here's another kernel density estimator that works with multivariate data. Sadly,
## you can't change the bandwidth with scipy 0.9.0.'
import scipy.stats as spstat
kernel = spstat.gaussian_kde(x.T)
fhat = kernel(x.T)


### Construct the level set tree or load a saved tree
bg_sets, levels = utl.constructDensityGrid(fhat, mode='mass', n_grid=None)
#bg_sets, levels = utl.constructDensityGrid(fhat, mode='levels', n_grid=250)

tree = dcl.makeLevelSetTree(W, levels, bg_sets, mode='density', verbose=True)
#dcl.loadTree('testsave')


### Prune the tree
tree.pruneBySize(p_prune, mode='proportion')
tree.summarize()


### Save the tree for future use
tree.save('testsave.mat')



################################################
### USE THE LEVEL SET TREE FOR DATA ANALYSIS ###
################################################

### Interactive tools
## Only uncomment and use one these at a time.
#tool = dcl.TreeComponentTool(tree, x, height_mode='mass', width_mode='uniform',
#	output=['scatter', 'tree'], fhat=fhat)
#tool = dcl.TreeClusterTool(tree, x, height_mode='mass', width_mode='uniform',
#	output=['scatter'], fhat=fhat)
#tool.show()


### Retrieve cluster assignments from the interactive tools manually
## This is especially useful for data with more than 3 dimensions, where the clusters
## (or single component) cannot be plotted directly by the interactive tools.
## Uncomment and paste the relevant code into the command line. Don't run the script
## straight through with these lines uncommented.
#uc = tool.getComponent()  # for TreeComponentTool
#uc = tool.getClusters()  # for TreeClusterTool


### Other script-based clustering tools
uc = tree.allModeCluster()
#uc = tree.upperSetCluster(cut=0.1, mode='mass')[0]
#uc = tree.firstKCluster(k=3)[0]
print np.bincount(uc[:, 1])


### Plot the level set tree with colored leaves
## For "all mode" clustering only, this and the following plot match tree nodes to data
## clusters.
leaves = [k for k, v in tree.nodes.items() if v.children == []]
fig = tree.plot(color=True, color_nodes=leaves)[0]
fig.show()


### Plot upper level set clusters manually
## This does not work for 1D data.
base = [217.0/255] * 3
black = [0.0] * 3
edge = plutl.makeColorMatrix(n, bg_color=black, bg_alpha=0.35, ix=None)
fill = plutl.makeColorMatrix(n, bg_color=base, bg_alpha=0.72, ix=uc[:,0],
	fg_color=uc[:,1], fg_alpha=0.68)
fig = plutl.plotPoints(x, clr=fill, edgecolor=edge)
fig.show()


### Assign background points
fc = utl.assignBackgroundPoints(x.reshape((n, -1)), uc, method='knn', k=9)
print np.bincount(fc[:, 1])








