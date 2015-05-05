
##############
### SET UP ###
##############
"""
Level set tree distance functions.
"""

import emd


def paint_mover_distance(tree1, tree2, ground_distance, form='kappa'):
    """
    Compute paint mover distance (PMD) between two level set trees.

    Parameters
    ----------
    tree1, tree2 : level set tree

    ground_distance : function
        Function to measure distance between two points on the level set tree
        canvas.

    form : string
        Vertical scale of the dendrograms.

    Returns
    -------
    d : float
        The paint mover distance.
    """
    signature1 = tree1.get_signature(form=form)
    signature2 = tree2.get_signature(form=form)
    d = emd.emd(signature1, signature2, ground_distance)
    return d

