"""
This example illustrates the power of density-based clustering, by correctly
clustering points from a "crater" distribution with a high density core and a
low density ring.

This demo also illustrates several more advanced features of DeBaCl:
  - How to build a level set tree from a similarity graph and density estimate,
    instead of tabular data.
  - How to save and load a level set tree.
  - Coloring dendrogram nodes to match feature space cluster colors.
  - Assigning "background" points to clusters with a scikit-learn classifier.
"""

import numpy as np
import matplotlib.pyplot as plt
import debacl as dcl
from sklearn.neighbors import KNeighborsClassifier


## Generate "crater" data.
n, p = 2000, 2
center = (0,) * p
sdev = 0.5 * np.eye(p)
radius = 15
mix = (0.3, 0.7)

n_ball, n_circle = np.random.multinomial(n, pvals=mix)

ball = np.random.multivariate_normal(center, sdev, n_ball)

gauss = np.random.multivariate_normal(center, sdev, n_circle)
norm = np.apply_along_axis(np.linalg.norm, axis=1, arr=gauss)
angle = gauss / norm[:, np.newaxis]
length = radius * np.random.uniform(low=0.2, high=0.35, size=n_circle)
circle = angle * length[:, np.newaxis]

X = np.vstack((ball, circle))


## Plot the data for intuition.
fig, ax = plt.subplots()
ax.set_title("Crater simulation")
ax.set_xlabel("x0")
ax.set_ylabel("x1", rotation=0)
ax.scatter(X[:,0], X[:,1], color='black', s=50, alpha=0.4)
fig.show()


## Construct a similarity graph and density estimate by hand. For this demo we
#  use the k-nearest neighbors similarity graph and density estimator that
#  DeBaCl uses by default, but this is the place to plug in custom similarity
#  graphs and density estimators. In particular, this functionality means the
#  user is not limited to tabular data or pre-set distance functions. If the
#  user can make a similarity graph and estimate a density (or pseudo-density),
#  then a level set tree can be estimated with DeBaCl.
k = int(0.05 * n)
knn_graph, radii = dcl.utils.knn_graph(X, k, method='kd-tree')
density = dcl.utils.knn_density(radii, n, p, k)


## Build the level set tree.
gamma = int(0.1 * n)
tree = dcl.construct_tree_from_graph(knn_graph, density, prune_threshold=gamma,
                                     verbose=True)
print tree


## Save the tree to disk and load it back (just for demo purposes).
tree.save('crater_tree.lst')

tree2 = dcl.load_tree('crater_tree.lst')
print tree2


## Get leaf clusters and leaf node indices.
leaves = tree.get_leaf_nodes()
labels = tree.get_clusters(method='leaf')


## Plot the tree, coloring the nodes used to cluster.
fig = tree.plot(form='mass', color_nodes=leaves, colormap=plt.cm.Spectral)[0]
fig.show()


## Plot the clusters in feature space.
X_clusters = X[labels[:, 0], :]

fig, ax = plt.subplots()
ax.set_title("High-density cluster labels")
ax.set_xlabel('x0')
ax.set_ylabel('x1', rotation=0)
ax.scatter(X[:, 0], X[:, 1], s=50, color='black', alpha=0.25)
ax.scatter(X_clusters[:, 0], X_clusters[:, 1], c=labels[:, 1], s=80, alpha=0.8,
           cmap=plt.cm.Spectral)
fig.show()


## Assign the background points to clusters.

# Previously we obtained labels just for the instances in the high-density
# clusters. We can also use the 'fill_background' method of the `get_clusters`
# method to identify *unlabeled* points with a label of -1.
labels = tree.get_clusters(method='leaf', fill_background=True)

cluster = labels[labels[:, 1] != -1]
background = labels[labels[:, 1] == -1]

X_cluster = X[cluster[:, 0]]
y_cluster  = cluster[:, 1]

# We can use any classifier to assign the background points to a cluster. For
# the purpose of illustration, here we use the K-nearest neighbors classifier
# from scikit-learn.
knn_class = KNeighborsClassifier(n_neighbors=5)
knn_class.fit(X_cluster, y_cluster)

X_background = X[background[:, 0]]
y_background = knn_class.predict(X_background)


## Re-draw all the points in feature space.
fig, ax = plt.subplots()
ax.set_title("Final cluster labels.")
ax.set_xlabel('x0')
ax.set_ylabel('x1', rotation=0)
ax.scatter(X_cluster[:, 0], X_cluster[:, 1], s=80, c=y_cluster, alpha=0.75,
           cmap=plt.cm.Spectral)
ax.scatter(X_background[:, 0], X_background[:, 1], s=80, c=y_background,
           alpha=0.75, cmap=plt.cm.Spectral)
fig.show()
