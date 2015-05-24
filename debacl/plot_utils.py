
import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
except:
    print "Matplotlib could not be loaded. DeBaCl plot functions will not work."



### Orchard plots
### -------------

def plot_orchard(orchard, color=(0.596, 0.306, 0.639), opacity=0.7, 
                 width='mass'):
    """
    Plot a collection of level set trees on the same canvas. For now, only draws
    the 'alpha' form of the trees.

    Parameters
    ----------
    orchard : list[LevelSetTree]
        A collection of level set trees.

    color : list or tuple [float]
        Color of the trees, in RGB format. The default is purple.

    opacity : Opacity of the trees. Setting this to be less than 1 can alleviate
        some overplotting effects.

    width : {'uniform', 'mass'}, optional
        Procedure for assigning horizontal white space for each branch. `See
        LevelSetTree.plot` for more detail.

    Returns
    -------
    fig : matplotlib.figure
    """
    fig, ax = plt.subplots()
    ax.set_xlim((-0.04, 1.04))
    ax.set_ylim(0., 1.02)
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_ylabel("Alpha", rotation=90)
    ax.yaxis.grid(color='gray')

    n = len(orchard)

    ## loop over trees
    for tree in orchard:

        # get roots and root masses
        ix_root = np.array([u for u, v in tree.nodes.iteritems()
            if v.parent is None])
        n_root = len(ix_root)
        census = np.array([len(tree.nodes[x].members) for x in ix_root],
            dtype=np.float)


        # order the roots by mass
        seniority = np.argsort(census)[::-1]
        ix_root = ix_root[seniority]
        census = census[seniority]
        n_pt = sum(census)


        # set up root silos
        weights = census / n_pt
        silos = np.cumsum(weights)
        silos = np.insert(silos, 0, 0.0)


        # set line segments for each root
        segments = {}
        splits = {}
        segmap = []
        splitmap = []

        for i, ix in enumerate(ix_root):
            branch = tree._construct_branch_map(ix, (silos[i],
                silos[i+1]), scale='alpha', width=width, sort=True)

            branch_segs, branch_splits, branch_segmap, branch_splitmap = branch
            segments = dict(segments.items() + branch_segs.items())
            splits = dict(splits.items() + branch_splits.items())
            segmap += branch_segmap
            splitmap += branch_splitmap

            verts = [segments[k] for k in segmap]
            lats = [splits[k] for k in splitmap]

            # draw the line segments
            ax.add_collection(LineCollection(verts, colors=np.tile(color,
                (len(segmap), 1))))
            ax.add_collection(LineCollection(lats, colors=np.tile(color,
                (len(splitmap), 1))))

    return fig



######################################
### PLOTTING FUNCITONS AND CLASSES ###
######################################
class Palette(object):
    """
    Define some good RGB sscolors manually to simplify plotting upper level sets
    and foreground clusters.

    Parameters
    ----------
    use : {'scatter', 'lines', 'neuroimg'}, optional
        Application for the palette. Different palettes work better in different
        settings.
    """

    def __init__(self, use='scatter'):
        self.black = np.array([0.0, 0.0, 0.0])
        if use == 'lines':
            self.colorset = np.array([
                (228, 26, 28), #red
                (55, 126, 184), #blue
                (77, 175, 74), #green
                (152, 78, 163), #purple
                (255, 127, 0), #orange
                (166, 86, 40), #brown
                (0, 206, 209), #turqoise
                (82, 82, 82), #dark gray
                (247, 129, 191), #pink
                (184, 134, 11), #goldenrod
                ]) / 255.0

        elif use == 'neuroimg':
            self.colorset = np.array([
                (170, 0, 0), # dark red
                (255, 0, 0), # red
                (0, 255, 0), # green
                (0, 0, 255), # blue
                (0, 255, 255), # cyan
                (255, 0, 255), # violet
                (255, 255, 0), # yellow
                ]) / 255.0

        else:
            self.colorset = np.array([
                    (228, 26, 28), #red
                    (55, 126, 184), #blue
                    (77, 175, 74), #green
                    (152, 78, 163), #purple
                    (255, 127, 0), #orange
                    (247, 129, 191), #pink
                    (166, 86, 40), #brown
                    (0, 206, 209), #turqoise
                    (85, 107, 47), #olive green
                    (127, 255, 0), #chartreuse
                    (205, 92, 92), #light red
                    (0, 0, 128), #navy
                    (255, 20, 147), #hot pink
                    (184, 134, 11), #goldenrod
                    (176, 224, 230), #light blue
                    (255, 255, 51), #yellow
                    (0, 250, 192),
                    (13, 102, 113),
                    (83, 19, 67),
                    (162, 38, 132),
                    (171, 15, 88),
                    (204, 77, 51),
                    (118, 207, 23), #lime green
                    (207, 203, 23), #pea green
                    (238, 213, 183), #bisque
                    (82, 82, 82), #dark gray
                    (150, 150, 150), #gray
                    (240, 240, 240) # super light gray
                    ]) / 255.0


    def apply_colorset(self, ix):
        """
        Turn a numpy array of group labels (integers) into RGBA colors.
        """
        n_clr = np.alen(self.colorset)
        return self.colorset[ix % n_clr]


def make_color_matrix(n, bg_color, bg_alpha, ix=None,
    fg_color=[228/255.0, 26/255.0, 28/255.0], fg_alpha=1.0):
    """
    Construct the RGBA color parameter for a matplotlib plot.

    This function is intended to allow for a set of "foreground" points to be
    colored according to integer labels (e.g. according to clustering output),
    while "background" points are all colored something else (e.g. light gray).
    It is used primarily in the interactive plot tools for DeBaCl but can also
    be used directly by a user to build a scatterplot from scratch using more
    complicated DeBaCl output. Note this function can be used to build an RGBA
    color matrix for any aspect of a plot, including point face color, edge
    color, and line color, despite use of the term "points" in the descriptions
    below.

    Parameters
    ----------
    n : int
        Number of data points.

    bg_color : list of floats
        A list with three entries, specifying a color in RGB format.

    bg_alpha : float
        Specifies background point opacity.

    ix : list of ints, optional
        Identifies foreground points by index. Default is None, which does not
        distinguish between foreground and background points.

    fg_color : list of ints or list of floats, optional
        Only relevant if 'ix' is specified. If 'fg_color' is a list of integers
        then each entry in 'fg_color' indicates the color of the corresponding
        foreground point. If 'fg_color' is a list of 3 floats, then all
        foreground points will be that RGB color. The default is to color all
        foreground points red.

    fg_alpha : float, optional
        Opacity of the foreground points.

    Returns
    -------
    rgba : 2D numpy array
        An 'n' x 4 RGBA array, where each row corresponds to a plot point.
    """

    rgba = np.zeros((n, 4), dtype=np.float)
    rgba[:, 0:3] = bg_color
    rgba[:, 3] = bg_alpha

    if ix is not None:
        if np.array(fg_color).dtype.kind == 'i':
            palette = Palette()
            fg_color = palette.applyColorset(fg_color)

        rgba[ix, 0:3] = fg_color
        rgba[ix, 3] = fg_alpha

    return rgba


def cluster_histogram(x, cluster, fhat=None, f=None, levels=None):
    """
    Plot a histogram and illustrate the location of selected cluster points.

    The primary plot axis is a histogram. Under this plot is a second axis that
    shows the location of the points in 'cluster', colored according to cluster
    label. If specified, also plot a density estimate, density function (or any
    function), and horizontal guidelines. This is the workhorse of the DeBaCl
    interactive tools for 1D data.

    Parameters
    ----------
    x : 1D numpy array of floats
        The data.

    cluster : 2D numpy array
        A cluster matrix: rows represent points in 'x', with first entry as the
        index and second entry as the cluster label. The output of all
        LevelSetTree clustering methods are in this format.

    fhat : list of floats, optional
        Density estimate values for the data in 'x'. Plotted as a black curve,
        with points colored according to 'cluster'.

    f : 2D numpy array, optional
        Any function. Arguments in the first column and values in the second.
        Plotted independently of the data as a blue curve, so does not need to
        have the same number of rows as values in 'x'. Typically this is the
        generating probability density function for a 1D simulation.

    levels : list of floats, optional
        Each entry in 'levels' causes a horizontal dashed red line to appear at
        that value.

    Returns
    -------
    fig : matplotlib figure
        Use fig.show() to show the plot, fig.savefig() to save it, etc.
    """

    n = len(x)
    palette = Palette()

    ## set up the figure and plot the data histogram
    fig, (ax0, ax1) = plt.subplots(2, sharex=True)
    ax0.set_position([0.125, 0.12, 0.8, 0.78])
    ax1.set_position([0.125, 0.05, 0.8, 0.05])

    ax1.get_yaxis().set_ticks([])
    ax0.hist(x, bins=n/20, normed=1, alpha=0.18)
    ax0.set_ylabel('Density')


    ## plot the foreground points in the second axes
    for i, c in enumerate(np.unique(cluster[:, 1])):
        ix = cluster[np.where(cluster[:, 1] == c)[0], 0]
        ax1.scatter(x[ix], np.zeros((len(ix),)), alpha=0.08, s=20,
            color=palette.colorset[i])

        if fhat is not None:
            ylim = ax0.get_ylim()
            eps = 0.02 * (max(fhat) - min(fhat))
            ax0.set_ylim(bottom=min(0.0-eps, ylim[0]), top=max(max(fhat)+eps,
                ylim[1]))
            ax0.scatter(x[ix], fhat[ix], s=12, alpha=0.5,
                color=palette.colorset[i])

    if f is not None:   # plot the density
        ax0.plot(f[:,0], f[:,1], color='blue', ls='-', lw=1)

    if fhat is not None:  # plot the estimated density
        ax0.plot(x, fhat, color='black', lw=1.5, alpha=0.6)

    if levels is not None:  # plot horizontal guidelines
        for lev in levels:
            ax0.axhline(lev, color='red', lw=1, ls='--', alpha=0.7)

    return fig


def plot_foreground(X, clusters, title='', xlab='x', ylab='y', zlab='z',
    fg_alpha=0.75, bg_alpha=0.3, edge_alpha=1.0, **kwargs):
    """
    Draw a scatter plot of 2D or 3D data, colored according to foreground
    cluster label.

    Parameters
    ----------
    X : 2-dimensional numpy array
        Data points represented by rows. Must have 2 or 3 columns.

    clusters : 2-dimensional numpy array
        A cluster matrix: rows represent points in 'x', with first entry as the
        index and second entry as the cluster label. The output of all
        LevelSetTree clustering methods are in this format.

    title : string
        Axes title

    xlab, ylab, zlab : string
        Axes axis labels

    fg_alpha : float
        Transparency of the foreground (clustered) points. A float between 0
        (transparent) and 1 (opaque).

    bg_alpha : float
        Transparency of the background (unclustered) points. A float between 0
        (transparent) and 1 (opaque).

    kwargs : keyword parameters
        Plot parameters passed through to Matplotlib Axes.scatter function.

    Returns
    -------
    fig : matplotlib figure
        Use fig.show() to show the plot, fig.savefig() to save it, etc.

    ax : matplotlib axes object
        Allows more direct plot customization in the client function.
    """

    ## make the color matrix
    n, p = X.shape
    base_clr = [190.0 / 255.0] * 3  ## light gray
    black = [0.0, 0.0, 0.0]

    rgba_edge = makeColorMatrix(n, bg_color=black, bg_alpha=edge_alpha, ix=None)
    rgba_clr = makeColorMatrix(n, bg_color=base_clr, bg_alpha=bg_alpha,
        ix=clusters[:, 0], fg_color=clusters[:, 1], fg_alpha=fg_alpha)

    if p == 2:
        fig, ax = plt.subplots()
        ax.scatter(X[:,0], X[:,1], c=rgba_clr, edgecolors=rgba_edge, **kwargs)

    elif p == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        fig.subplots_adjust(bottom=0.0, top=1.0, left=-0.05, right=0.98)
        ax.set_zlabel(zlab)
        ax.scatter(X[:,0], X[:,1], X[:,2], c=rgba_clr, edgecolors=rgba_edge,
            **kwargs)

    else:
        fig, ax = plt.subplots()
        print "Plotting failed due to a dimension problem."

    ax.set_title(title)
    ax.set_xlabel(xlab); ax.set_ylabel(ylab)

    return fig, ax


def set_plot_params(axes_titlesize=22, axes_labelsize=18, xtick_labelsize=14,
    ytick_labelsize=14, figsize=(9, 9), n_ticklabel=4):
    """
    A handy function for setting matplotlib parameters without adding trival
    code to working scripts.

    Parameters
    ----------
    axes_titlesize : integer
        Size of the axes title.

    axes_labelsize : integer
        Size of axes dimension labels.

    xtick_labelsize : integer
        Size of the ticks on the x-axis.

    ytick_labelsize : integer
        Size of the ticks on the y-axis.

    figure_size : tuple (length 2)
        Size of the figure in inches.

    Returns
    -------
    """

    mpl.rc('axes', labelsize=axes_labelsize)
    mpl.rc('axes', titlesize=axes_titlesize)
    mpl.rc('xtick', labelsize=xtick_labelsize)
    mpl.rc('ytick', labelsize=ytick_labelsize)
    mpl.rc('figure', figsize=figsize)

    def autoloc(self):
        ticker.MaxNLocator.__init__(self, nbins=n_ticklabel)
    ticker.AutoLocator.__init__ = autoloc



