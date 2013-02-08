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
import sys
sys.path.append('..')
import debacl as dcl
import gen_utils as utl
import plot_utils as plutl

import numpy as np
np.set_printoptions(precision=4, edgeitems=5, suppress=True, linewidth=160)
import scipy.stats as spstat


### Set Parameters
n = 1000
mix = (0.5, 0.3, 0.2)
ctr = (1, 6, 11)
sdev = (1, 1, 1)
p_k = 0.02
k = int(p_k * n)
p_prune = 0.05


### Generate data
## 'x' is the data
membership = np.random.multinomial(n, pvals=mix)
x = np.array([], dtype=np.float)

for (p, mu, sigma) in zip(membership, ctr, sdev):
	draw = spstat.norm.rvs(size=p, loc=mu, scale=sigma)
	x = np.append(x, draw)

x = np.sort(x)



####################################
### CONSTRUCT THE LEVEL SET TREE ###
####################################

### Construct similarity graph
W, k_radius = utl.knnGraph(x.reshape((n, -1)), k)
#W, eps = utl.epsilonGraph(x.reshape((n, -1)), q=0.05)
np.fill_diagonal(W, False)


### Compute density estimate
## k-nearest neighbor density 
fhat = utl.knnDensity(k_radius, n, p=1, k=k)

## any density estimator works - here's a kernel estimator with a gaussian kernel
# from statsmodels.nonparametric.kde import KDE
#fhat_func = KDE(z)
#fhat_func.fit(bw=1.0)
#fhat = fhat_func.evaluate(z)


### Construct the level set tree or load a saved tree
bg_sets, levels = utl.constructDensityGrid(fhat, mode='mass', n_grid=None)
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

### Static level set tree plot
fig = tree.plot()[0]
fig.show()


### Clustering tools
uc = tree.allModeCluster()
#uc = tree.upperSetCluster(cut=0.1, mode='mass')[0]
#uc = tree.firstKCluster(k=3)[0]
print np.bincount(uc[:, 1])


### Assign background points
fc = utl.assignBackgroundPoints(x.reshape((n, -1)), uc, method='knn', k=9)
print np.bincount(fc[:, 1])


### Interactive tools
## Only uncomment and use one these at a time.
tool = dcl.TreeComponentTool(tree, x, height_mode='mass', width_mode='uniform',
	output=['scatter', 'tree'], fhat=fhat)
#tool = dcl.TreeClusterTool(tree, x, height_mode='mass', width_mode='uniform',
#	output=['scatter'], fhat=fhat)
tool.show()






