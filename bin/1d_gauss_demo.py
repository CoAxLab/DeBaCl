"""
Illustrate how a level set tree works for a mixture of Gaussian distributions
in one dimension. The tree is estimated with the `construct_tree` constructor,
which takes the raw data as input.
"""

import numpy as np
import matplotlib.pyplot as plt
import debacl as dcl


## Generate data.
n = 1500
centers = ((1,), (6,), (11,))
sdev = (np.eye(1),) * 3
mix = (0.5, 0.3, 0.2)

membership = np.random.multinomial(n, pvals=mix)
p = len(centers[0])
X = np.zeros((n, p), dtype=np.float)
g = np.zeros((n, ), dtype=np.int)
b = np.cumsum((0,) + tuple(membership))

for i, (size, mu, sigma) in enumerate(zip(membership, centers, sdev)):
  ix = range(b[i], b[i+1])
  X[ix, :] = np.random.multivariate_normal(mu, sigma, size)
  g[ix] = i

X = np.sort(X, axis=0)


## Estimate the level set tree.
k = int(0.02 * n)
gamma = int(0.05 * n)

tree = dcl.construct_tree(X, k, prune_threshold=gamma, verbose=True)
print tree


## Retrieve cluster assignments from the tree.
labels = tree.get_clusters(method='leaf')


## Labels returned from the `get_clusters` method match the index of the
#  highest density node to which an observation belongs. Because these labels
#  are usually non-consecutive, we can reindex to make many post-processing
#  steps more natural.
new_labels = dcl.utils.reindex_cluster_labels(labels)
print "cluster counts:", np.bincount(new_labels[:, 1])


## Plot the level set tree as a dendrogram. The plot function returns a tuple
#  containing 4 objects. The first item is a matplotlib figure, which can be
#  shown and saved.
plot = tree.plot()
fig = plot[0]
fig.show()