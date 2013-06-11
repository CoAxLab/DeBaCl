############################################
## Brian P. Kent
## plot_utils.py
## Created: 20120712
## Updated: 20130207
###########################################

##############
### SET UP ###
##############
"""
Repository of plotting functions useful for the DeBaCl project, particularly the
interactive level set tree analysis tools.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl

mpl.rc('axes', labelsize=18)
mpl.rc('axes', titlesize=22) 
mpl.rc('xtick', labelsize=14)
mpl.rc('ytick', labelsize=14) 
mpl.rc('figure', figsize=(9, 9))



################################################
### COLOR PALETTE CLASS AND HELPER FUNCTIONS ###
################################################

class Palette(object):
	"""
	Define my favorite colors for use in plots and methods to simplify using the colors.
	
	Parameters
	----------
	use : {'scatter', 'lines'}, optional
		Purpose for the palette. Different palettes work better in different
		applications.
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
					(238, 213, 183), #bisque
					(82, 82, 82), #dark gray
					(150, 150, 150), #gray
					(240, 240, 240) # super light gray
					]) / 255.0
					

 	def applyColorset(self, ix):
 		"""
 		Turn a numpy array of group labels (integers) into RGBA colors.
 		"""
		return self.colorset[ix] 		
 	 	
 	
 	def makeColorMap(self):
 		"""
 		Returns a matplotlib colormap indexed by integers 0,1,...,20 with good colors.
 		"""
	
		n_color = self.palette.shape[0]
		cmap = clr.ListedColormap(colors=self.palette, name='plotUtilsCMap')
		cnorm = clr.BoundaryNorm(range(n_color), n_color)
		return cmap, cnorm 		
 		


def makeColorMatrix(n, bg_color, bg_alpha, ix=None,
	fg_color=[228/255.0, 26/255.0, 28/255.0], fg_alpha=1.0):
	"""
	Construct the RGBA color parameter for a matplotlib plot. This function is
	intended to allow for a set of "foreground" points to be colored according
	to integer labels (e.g. according to clustering output), while "background"
	points are all colored something else (e.g. light gray). It is used
	primarily in the interactive plot tools for DeBaCl but can also be used
	directly by a user to build a scatterplot from scratch using more
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
		Only relevant if 'ix' is specified. If 'fg_color' is a list of integers then
		each entry in 'fg_color' indicates the color of the corresponding foreground
		point. If 'fg_color' is a list of 3 floats, then all foreground points will be
		that RGB color. The default is to color all foreground points red.
	
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
		if np.array(fg_color).dtype.kind == 'i':  # then fg_color is a palette index
			palette = Palette()
			fg_color = palette.applyColorset(fg_color)
		
		rgba[ix, 0:3] = fg_color
		rgba[ix, 3] = fg_alpha
		
	return rgba




######################
### SPECIFIC PLOTS ###
######################

def plotPoints(X, size=20, clr='blue', symb='o', alpha=None, edgecolor=None, title='',
	xlab='x', ylab='y', zlab='z', azimuth=160, elev=10):
	"""
	Draw a scatter plot of 2D or 3D data.
	
	Except for the data array 'X', see the matplotlib documentation for more detail on
	plot parameters.
	
	Parameters
	----------
	X : 2D numpy array
		Data points represented by rows. Must have 2 or 3 columns.
		
	size : int, optional
	
	clr : string, list of ints, or RGBA matrix, optional
	
	symb : string, optional
	
	alpha : float, optional
	
	edgecolor : RGBA matrix, optional
	
	title, xlab, ylab, zlab : string, optional
	
	azimuth, elev : int, optional
		Azimuth should be in the interval [0, 359]. Elevation should be in the range [0,
		90].
	
	Returns
	-------
	fig : matplotlib figure
		Use fig.show() to show the plot, fig.savefig() to save it, etc.
	"""

	n, p = X.shape
	
	## Deal with 'clr' vectors
	if isinstance(clr, np.ndarray) and clr.ndim == 1:
		palette = Palette()
		clr = palette.applyColorset(clr)

	if p == 2:
		fig, ax = plt.subplots()
		ax.set_title(title)
		ax.set_xlabel(xlab)
		ax.set_ylabel(ylab)
		ax.scatter(X[:,0], X[:,1], s=size, c=clr, marker=symb, alpha=alpha,
			edgecolor=edgecolor)
		
	elif p == 3:
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		fig.subplots_adjust(bottom=0.0, top=1.0, left=-0.05, right=0.98)
		ax.set_title(title)
		ax.set_xlabel(xlab)
		ax.set_ylabel(ylab)
		ax.set_zlabel(zlab)
		ax.scatter(X[:,0], X[:,1], X[:,2], s=size, c=clr, edgecolor=edgecolor)
		ax.azim = azimuth
		ax.elev = elev
		
	else:
		fig, ax = plt.subplots()
		print "Plotting failed due to a dimension problem."
			
	return fig
	
	

def clusterHistogram(x, cluster, fhat=None, f=None, levels=None):
	"""
	Plot a histogram and illustrate the location of selected cluster points.
	
	The primary plot axis is a histogram. Under this plot is a second axis that shows
	the location of the points in 'cluster', colored according to cluster label. If
	specified, also plot a density estimate, density function (or any function), and
	horizontal guidelines. This is the workhorse of the DeBaCl interactive tools for 1D
	data.
	
	Parameters
	----------
	x : 1D numpy array of floats
		The data.
	
	cluster : 2D numpy array
		A cluster matrix: rows represent points in 'x', with first entry as the index
		and second entry as the cluster label. The output of all LevelSetTree
		clustering methods are in this format.
		
	fhat : list of floats, optional
		Density estimate values for the data in 'x'. Plotted as a black curve, with
		points colored according to 'cluster'.
	
	f : 2D numpy array, optional
		Any function. Arguments in the first column and values in the second. Plotted
		independently of the data as a blue curve, so does not need to have the same
		number of rows as values in 'x'. Typically this is the generating probability
		density function for a 1D simulation.
	
	levels : list of floats, optional
		Each entry in 'levels' causes a horizontal dashed red line to appear at that
		value.
	
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
			ax0.set_ylim(bottom=min(0.0-eps, ylim[0]), top=max(max(fhat)+eps, ylim[1]))
			ax0.scatter(x[ix], fhat[ix], s=12, alpha=0.5, color=palette.colorset[i])
	
	
	if f is not None:	# plot the density
		ax0.plot(f[:,0], f[:,1], color='blue', ls='-', lw=1)
	
	if fhat is not None:  # plot the estimated density 
		ax0.plot(x, fhat, color='black', lw=1.5, alpha=0.6)

	if levels is not None:  # plot horizontal guidelines
		for lev in levels:
			ax0.axhline(lev, color='red', lw=1, ls='--', alpha=0.7)
			
	return fig
	





