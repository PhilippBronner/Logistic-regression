import numpy as np 
import pandas as pd 
# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)


class KMeans:
    
    def __init__(self,dim = 2, n_centroids = 2):
        # NOTE: Feel free add any hyperparameters 
        # (with defaults) as you see fit
        self.centers = np.zeros((n_centroids,dim))
        self.dim = dim
        self.n_centroids = n_centroids
        pass
        
    def fit(self, X, ax_scale = 1):
        """
        Estimates parameters for the classifier
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
        """
        #Initial Centers
        n = len(X[:,0])
        selection = np.random.randint(n,size = self.n_centroids)
        self.centers = np.take(X, selection, axis = 0)
        center_obs =  [self.centers]
        dcc = 1
        while dcc > 0.1:
            self.centers = self.get_centroids(X, self.centers, ax_scale)
            center_obs.append(self.centers)
            dcc = np.sqrt(np.sum((center_obs[-1]-center_obs[-2])**2))
        return center_obs
    
    def predict(self, X):
        """
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            X (array<m,n>): a matrix of floats with 
                m rows (#samples) and n columns (#features)
            
        Returns:
            A length m integer array with cluster assignments
            for each point. E.g., if X is a 10xn matrix and 
            there are 3 clusters, then a possible assignment
            could be: array([2, 0, 0, 1, 2, 1, 1, 0, 2, 2])
        """
        m = self.n_centroids #number of centrodids
        dim = self.dim #X[0,:] dimensionality
        n = len(X[:,0])
        #Calculate distances
        dXc = np.zeros((n, dim, m))
        for ii in range(m):
            dXc[:,:,ii] = X-self.centers[ii,:]
        dXc_abs = np.sqrt(np.sum(dXc**2,axis = 1))
        minimum = np.amin(dXc_abs, axis = 1)
        cluster_assignment = []
        for ii in range(n): 
            j = np.where(dXc_abs[ii,:] == minimum[ii])
            cluster_assignment.append( int(j[0][0]))
        
        return np.array(cluster_assignment)

    
    def get_centroids(self, X, centers, ax_scale):
        """
        Returns the centroids found by the K-mean algorithm
        
        Example with m centroids in an n-dimensional space:
        >>> model.get_centroids()
        numpy.array([
            [x1_1, x1_2, ..., x1_n],
            [x2_1, x2_2, ..., x2_n],
                    .
                    .
                    .
            [xm_1, xm_2, ..., xm_n]
        ])
        """
        m = len(centers[:,0])
        dim = len(centers[0,:]) #X[0,:] dimensionality
        n = len(X[:,0])
            #Calculate distances
        dXc = np.zeros((n, dim, m))
        for ii in range(m):
            dXc[:,:,ii] = X-centers[ii,:]
        #dXc[:,1,:] = dXc[:,1,:]*ax_scale
        dXc_abs = np.sqrt(np.sum(dXc**2,axis = 1))
        #for ii in range(n):
        minimum = np.amin(dXc_abs, axis = 1)
        #dXC_abs
        sp = np.zeros((m,dim))
        counter = np.zeros((m,dim))
        for ii in range(n): 
            j = np.where(dXc_abs[ii,:] == minimum[ii])
            sp[j,:] += X[ii,:]
            counter[j,:] += 1
        sp = sp/counter
        return sp
        # TODO: Implement 
        #raise NotImplementedError()
    
    
    
    
# --- Some utility functions 


def euclidean_distortion(X, z):
    """
    Computes the Euclidean K-means distortion
    
    Args:
        X (array<m,n>): m x n float matrix with datapoints 
        z (array<m>): m-length integer vector of cluster assignments
    
    Returns:
        A scalar float with the raw distortion measure 
    """

    
    X, z = np.asarray(X), np.asarray(z)
    assert len(X.shape) == 2
    assert len(z.shape) == 1
    assert X.shape[0] == z.shape[0]
    
    distortion = 0.0
    for c in np.unique(z):
        Xc = X[z == c]
        mu = Xc.mean(axis=0)
        distortion += ((Xc - mu) ** 2).sum()
        
    return distortion

def euclidean_distance(x, y):
    """
    Computes euclidean distance between two sets of points 
    
    Note: by passing "y=0.0", it will compute the euclidean norm
    
    Args:
        x, y (array<...,n>): float tensors with pairs of 
            n-dimensional points 
            
    Returns:
        A float array of shape <...> with the pairwise distances
        of each x and y point
    """
    return np.linalg.norm(x - y, ord=2, axis=-1)

def cross_euclidean_distance(x, y=None):
    """
    Compute Euclidean distance between two sets of points 
    
    Args:
        x (array<m,d>): float tensor with pairs of 
            n-dimensional points. 
        y (array<n,d>): float tensor with pairs of 
            n-dimensional points. Uses y=x if y is not given.
            
    Returns:
        A float array of shape <m,n> with the euclidean distances
        from all the points in x to all the points in y
    """
    y = x if y is None else y 
    assert len(x.shape) >= 2
    assert len(y.shape) >= 2
    return euclidean_distance(x[..., :, None, :], y[..., None, :, :])


def euclidean_silhouette(X, z):
    """
    Computes the average Silhouette Coefficient with euclidean distance 
    
    More info:
        - https://www.sciencedirect.com/science/article/pii/0377042787901257
        - https://en.wikipedia.org/wiki/Silhouette_(clustering)
    
    Args:
        X (array<m,n>): m x n float matrix with datapoints 
        z (array<m>): m-length integer vector of cluster assignments
    
    Returns:
        A scalar float with the silhouette score
    """
    X, z = np.asarray(X), np.asarray(z)
    assert len(X.shape) == 2
    assert len(z.shape) == 1
    assert X.shape[0] == z.shape[0]
    
    # Compute average distances from each x to all other clusters
    clusters = np.unique(z)
    D = np.zeros((len(X), len(clusters)))
    for i, ca in enumerate(clusters):
        for j, cb in enumerate(clusters):
            in_cluster_a = z == ca
            in_cluster_b = z == cb
            d = cross_euclidean_distance(X[in_cluster_a], X[in_cluster_b])
            div = d.shape[1] - int(i == j)
            D[in_cluster_a, j] = d.sum(axis=1) / np.clip(div, 1, None)
    
    # Intra distance 
    a = D[np.arange(len(X)), z]
    # Smallest inter distance 
    inf_mask = np.where(z[:, None] == clusters[None], np.inf, 0)
    b = (D + inf_mask).min(axis=1)
    
    return np.mean((b - a) / np.maximum(a, b))
