############################################
## Brian P. Kent
## clustUtils.py
## Created: 20120718
## Updated: 20120827
## Repository of utility functions useful for clustering algorithms
###########################################


##############
### SET UP ###
##############
import numpy as np
import scipy.spatial.distance as spdist
import scipy.special as spspec
import munkres




#########################
### CLASS DEFINITIONS ###
#########################-
	
class Clustering:
	"""
	Define a clustering object, which contains methods for aligning group and cluster
	labels (if the group labels are defined) and a colorset for plotting up to 19
	clusters.
	"""
	
	def __init__(self, label, ctr, grp, k):
		self.label = label
		self.center = ctr
		self.group = grp
		self.k = k
		self.n = len(label)
		
		
		
	def alignLabels(self):
		"""
		Permute cluster labels or group labels to maximize the number of observations
		where the group label matches the cluster label. Uses the 'munkres' library, which
		includes the polynomial time Hungarian algorithm. If there are more groups than
		clusters the group labels are permuted, otherwise the cluster labels are permuted.
		"""
	
		m = munkres.Munkres()
		t = self.makeCrosstab()
		cost = 1 - t/(self.n * 1.0)
		cost = cost.tolist()

		align = m.compute(cost)
		
		n_grp, n_clust = t.shape
		
		if n_clust >= n_grp:
			temp_label = np.zeros((self.n, ), dtype=np.int) - 1
			labels = np.unique(self.label)
			diag_labels = [x[1] for x in align]
			free_labels = np.setdiff1d(labels, diag_labels)
		
			for (new, old) in align:
				ix = np.where(self.label == old)[0]
				temp_label[ix] = new

			m = np.max(temp_label) + 1
			
			for i, old in enumerate(free_labels):
				ix = np.where(self.label == old)[0]
				temp_label[ix] = i + m
			
			self.label = temp_label
			
		else:
			temp_group = np.zeros((self.n, ), dtype=np.int) - 1
			groups = np.unique(self.group)
			diag_groups = [x[0] for x in align]
			free_groups = np.setdiff1d(groups, diag_groups)
	
			for (old, new) in align:
				ix = np.where(self.group == old)[0]
				temp_group[ix] = new
				
			m = np.max(temp_group) + 1
		
			for i, old in enumerate(free_groups):
				ix = np.where(self.group == old)[0]
				temp_group[ix] = i + m

			self.group = temp_group
			

		
	def makeCrosstab(self):
		"""
		Generate the crosstab of groups vs clusters. Groups are represented by rows in the
		cross-tabulation and clusters by columns.
		"""
		
		n_grp = len(np.unique(self.group))
		n_clst = len(np.unique(self.label))
		
		crosstab, xedge, yedge = np.histogram2d(self.group, self.label,
			bins = (range(n_grp+1), range(n_clst+1)))
			
		crosstab = crosstab.astype(int)
		return crosstab
		
		
		
	def computeError(self):
		"""
		Return the fraction of observations whose cluster and group labels do not match.
		"""
		
		return (1.0 * sum(self.label != self.group)) / self.n
		
		
		
	def computeSSW(self, X):
		"""
		Compute the within-cluster sum of squares between each point in X and it's
		center. The cluster labels in self.label are assumed to correspond to the row 
		of X with the same index. Requires self.center to be defined.
		"""

		C = self.center[self.label,:]
		D = X - C
		D = D*D
		
		return np.sum(D)
		
			
				
	def alignLabelsByCoord(self, x):
		'''Change the labels in self.label so that the labels are ordered according to
		each cluster's minimum value in the 'x' vector.
		
		Inputs
		* (self)
		* x - a one dimensional vector
		
		Outputs
		* a relabeling of the clusters. Membership should remain the same.
		* cluster centers are also reordered to match the new cluster labels.'''
		
		mins = np.zeros((self.k,))
		for c in range(self.k):
			mins[c] = min(x[self.label==c])
	
		ix = np.argsort(mins)
		label = np.argsort(ix)
		self.label = label[self.label]
		self.center = self.center[ix,:]
		
		
		
	def computeCenters(self, X):
		"""
		Computes the mean of the points in X (each row is a point) grouped by the clustering
		in 'self'.
		"""
		
		p = X.shape[1]
		C = np.zeros((self.k, p))
		for i in range(self.k):
			ix_clust = np.where(self.label == i)[0]
			C[i,:] = np.mean(X[ix_clust,:], axis=0)
		
		self.center = C
		
		
		
	def matchLabels(self, ctr):
		'''JY's implementation
			Match the center labels by comparing self center points and original center point
			*ctr: original center points''' 
		print 'ctr.shape[0]: ', ctr.shape
		print 'self.center.shape[0]: ', self.center.shape
		minIx = np.zeros(ctr.shape[0])
		copyCenter = self.center
		
		for i, ci in enumerate(self.center):
			D = ctr-ci
			minIx[i] = np.argmin(np.sum(D*D, axis=1))
			print 'D*D: ', 	np.sum(D*D, axis=1), 'argmin(): ', np.argmin(np.sum(D*D, axis=1))
			print 'i:', i,'minIx[i]: ', minIx[i] 
		
			self.center[minIx] = copyCenter 
			for i in range(np.alen(self.label)):
				self.label[i] = minIx[self.label[i]]
	
	
	
	def matchLabels_spc(self, ctr, X):
		'''JY's implementation
			Match the center labels by comparing self center points and original center point
			*ctr: original center points
			*X: original pts''' 
		minIx = np.zeros((ctr.shape[0],), dtype=np.int)
		clstCtr =	self.computeCenters(X)
		copyCenter = self.center.copy()
		print 'ctr.shape: ', ctr.shape
		print 'clstCtr.shape: ', clstCtr.shape

		for i, ci in enumerate(clstCtr):
			D = ctr-ci
			minIx[i] = np.argmin(np.sum(D*D, axis=1))
			print 'D*D: ', 	np.sum(D*D, axis=1), 'argmin(): ', np.argmin(np.sum(D*D, axis=1))
			print 'i:', i,'minIx[i]: ', minIx[i] 
		
		self.center[minIx] = copyCenter 
		
		for i in range(np.alen(self.label)):
			self.label[i] = minIx[self.label[i]]




##################################
### GRAPH GENERATING FUNCTIONS ###
##################################

def gaussianGraph(x, sigma):
	"""
	Generate a (complete) neighborhood graph using a Gaussian kernel.
	"""
	
	d = spdist.pdist(x, metric='sqeuclidean')
	c = np.percentile(d, 1)
	W = np.exp(-1 * d / sigma)
	W = spdist.squareform(W)
	
	return W
	
	


def epsilonGraph(x, eps=None, q=0.05):

	d = spdist.pdist(x, metric='euclidean')
	D = spdist.squareform(d)

	if eps == None:
		eps = np.percentile(d, round(q*100))
		
	W = D <= eps

	return W, eps
	
	
	
def knnGraph(x, k=None, q=0.05):
	
	n, p = x.shape
	if k == None:
		k = int(round(q * n))

	d = spdist.pdist(x, metric='euclidean')
	D = spdist.squareform(d)
			
	## identify which indices count as neighbors for each node
	rank = np.argsort(D, axis=1)
	ix_nbr = rank[:, 0:k]   # should this be k+1 to match Kpotufe paper?
	ix_row = np.tile(np.arange(n), (k, 1)).T
	
	## make adjacency matrix for unweighted graph
	W = np.zeros(D.shape, dtype=np.bool)
	W[ix_row, ix_nbr] = True
	W = np.logical_or(W, W.T)
	
	## find the radius of the k'th neighbor
	k_nbr = ix_nbr[:, -1]
	k_radius = D[np.arange(n), k_nbr]
		
		
	return W, k_radius
	
	
	
def kpotufeDensity(k_radius, n, p, k):
	
	unit_vol = np.pi**(p/2.0) / spspec.gamma(1 + p/2.0)
	const = (1.0 * k) / (n * unit_vol)
	f_hat = const / k_radius**p
	
	return f_hat
	
	


########################################
### SPECTRAL DECOMPOSITION FUNCTIONS ###
########################################

def spectralDecompose():
	"""
	Compute an eigen decomposition of a matrix.
	"""
 	pass
 	
 	
 	
def findElbow():
	"""
 	Find the optimal gap in any vector of values, usually eigenvalues.
 	"""
 	pass
 	
 	


#################################
### GENERIC UTILITY FUNCTIONS ###
#################################

def arrayMatch(y, x):
	"""
	Returns the index in x for each element in y. Based closely on the HYRY's answer on
	stackoverflow.com at http://stackoverflow.com/questions/8251541/numpy-for-every-element-
	in-one-array-find-the-index-in-another-array
	"""
	
	index = np.argsort(x)
	sorted_x = x[index]
	sorted_index = np.searchsorted(sorted_x, y)
	yindex = np.take(index, sorted_index, mode='clip')
	mask = x[yindex] != y
	
	return yindex, mask
	
	


def drawSample(n, k):
	"""
	Chooses k indices from range(n) without replacement by shuffling range(n) uniformly over
	all permutations. In numpy 1.7 and beyond, the "choice" function will replace this code.
	"""
	
	ix = np.arange(n)
	np.random.shuffle(ix)
	ix_keep = ix[0:k]
	
	return ix_keep
	
	
	
def assignBackgroundPoints(X, clusters, method=None, k=1):
	"""
	The cluster assignments of some of the rows of 'X' should be listed as rows in 'clusters'.
	The other rows in X are assigned to the clusters with one of many possible methods.
	"""

	n, p = X.shape
	labels = np.unique(clusters[:,1])
	n_label = len(labels)

	assignments = np.zeros((n, ), dtype=np.int) - 1
	assignments[clusters[:,0]] = clusters[:,1]
	ix_background = np.where(assignments == -1)[0]
	
	if len(ix_background) == 0:
		return clusters
		

	if method == 'centers':

		## get cluster centers
		ctrs = np.empty((n_label, p), dtype=np.float)
		ctrs.fill(np.nan)

		for i, c in enumerate(labels):
			ix_c = clusters[np.where(clusters[:,1] == c)[0], 0]
			ctrs[i, :] = np.mean(X[ix_c,:], axis=0)

		## get the background points
		X_background = X[ix_background, :]

		## distance between each background point and all cluster centers and optimal center
		d = spdist.cdist(X_background, ctrs)
		ctr_min = np.argmin(d, axis=1)
		assignments[ix_background] = labels[ctr_min]	

		
	elif method == 'knn':
	
		## make sure k isn't too big
		k = min(k, np.min(np.bincount(clusters[:,1])))		

		## find distances between background and upper points
		X_background = X[ix_background, :]
		X_upper = X[clusters[:,0]]
		d = spdist.cdist(X_background, X_upper)

		## find the k-nearest neighbors
		rank = np.argsort(d, axis=1)
		ix_nbr = rank[:, 0:k]

		## find the cluster membership of the k-nearest neighbors
		knn_clusters = clusters[ix_nbr, 1]	
		knn_cluster_counts = np.apply_along_axis(np.bincount, 1, knn_clusters, None, n_label)
		knn_vote = np.argmax(knn_cluster_counts, axis=1)

		assignments[ix_background] = labels[knn_vote]


	elif method == 'meanshift':
		## use the mean shift algorithm to assign background points
		print "Sorry, this method has not been implemented yet."
		clust = None
		
		
	elif method == 'zero':
		assignments += 1


	else:  # assume method == None
		assignments[ix_background] = max(labels) + 1


	return np.array([range(n), assignments], dtype=np.int).T

	

def consecutiveLabels(x):
	"""
	Relabels the integers in x to be consecutive whole numbers.
	"""
	
	x = np.asarray(x)
	out = np.zeros(x.shape, dtype=np.int) - 1
	
	for i, c in enumerate(np.unique(x)):
		ix = np.where(x == c)[0]
		out[ix] = i

	return out
	
	
	



	
	






	
	
	
	



