from sklearn.cluster        import KMeans
from sklearn.neighbors      import kneighbors_graph
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph   import laplacian
import numpy as np


"""Args:
X: input samples, array (num, dim)
n_clusters:   no. of clusters
n_neighbours: neighborhood size

Returns:
Y: labels for samples, array (num,)
"""
def spectral_clustering(X, n_clusters=2, n_neighbors=10):
    n, d = X.shape
    A = kneighbors_graph(X, n_neighbors, mode='connectivity').toarray()
    L = laplacian(A, normed=True)
    w, v = np.linalg.eig(L)
    w, v = w.real, v.real
    i = np.argsort(w)
    w, v = w[i], v[:,i]
    Y = KMeans(n_clusters).fit_predict(v[:,:2])
    return Y
