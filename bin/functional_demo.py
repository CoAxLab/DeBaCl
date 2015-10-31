"""
DeBaCl can estimate a level set tree for complex data types. The basic LST
constructor `construct_tree` takes tabular data and assumes euclidean distance,
but the `construct_tree_from_graph` function takes a similarity graph and a
density estimate as inputs instead. This example shows how to build a level set
tree for functional data by manually constructing a similarity graph and
pseudo-density estimate, then using the `construct_tree_from_graph` function.
"""

import debacl as dcl


## Download or generate data.
# - Phoneme data is good to show but evenly sampled, so effectively euclidean.
# - Fiber tracks, but hard to explain
# - Hurricane data?
#   - plotting is hard.
#   - unevenly sampled.
#   - very intuitive.
#   - license for basemap?
#   - getting basemap to work on conda?
#   - subsample to get down to a reasonable data size (need to save the data in
#     the repo).
#   - license for saving the data in the repo?

## Construct the similarity graph.

## Estimate the pseudo-density function.

## Construct the level set tree.

## Plot the level set tree.

## Get cluster labels from the level set tree.

## Draw the labeled curves in feature space.