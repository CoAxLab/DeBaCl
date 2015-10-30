"""
This example illustrates the power of density-based clustering, by correctly
clustering points from a "crater" distribution with a high density core and a
low density ring.

This demo also illustrates several more advanced features of DeBaCl:

  - How to save and load a level set tree.

  - Different level set tree plot forms and coloring dendrogram nodes to match
    feature space cluster colors.

  - Assigning "background" points to clusters with a scikit-learn classifier.
"""

## Generate "crater" data.


## Plot the data for intuition.
fig, ax = plt.subplots()
ax.scatter(X[:,0], X[:,1], s=50, c=g, alpha=0.4)
fig.show()


## Estimate the level set tree.
tree = dcl.construct_tree(X, k, prune_threshold=gamma, verbose=True)
print tree


## Save the tree to disk.
tree.save('crater_tree.lst')


## Load the tree back into memory.
tree2 = dcl.load_tree('crater_tree.lst')
print tree2


## Get leaf clusters and leaf node indices.

## Plot the tree, coloring the nodes used to cluster.

## Plot the clusters in feature space.

## Assign the background points to clusters.

## Re-draw all the points in feature space.




 # The last item indicates the color used to draw each node,
#  which is useful for connecting nodes in the dendrogram to instances plotted
#  in feature space.