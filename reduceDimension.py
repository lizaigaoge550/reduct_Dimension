from sklearn import (manifold,  decomposition, ensemble,
                     discriminant_analysis, random_projection)
from sklearn.decomposition import PCA


def Pca(x,n):
    pca = PCA(n_components=n)
    return pca.fit_transform(x)


def LDA(x,label):
    x.flat[::x.shape[1]+1] += 0.01
    x_lda = discriminant_analysis.LinearDiscriminantAnalysis().fit_transform(x)
    return x_lda

def Isomap(x):
    x_iso = manifold.Isomap(n_components=2).fit_transform(x)
    return x_iso

def LLE(x,method):
    x_lle = manifold.LocallyLinearEmbedding(n_components=2,method=method).fit_transform(x)
    return x_lle

def MDS(x):
    x_mds = manifold.MDS(n_components=2, n_init=1, max_iter=100).fit_transform(x)
    return x_mds

def RTE(x):
    hasher = ensemble.RandomTreesEmbedding(n_estimators=200, random_state=0, max_depth=5)
    x_transformed = hasher.fit_transform(x)
    x_reduced = decomposition.TruncatedSVD(n_components=2).fit_transform(x_transformed)
    return x_reduced

def SE(x):
    embedder = manifold.SpectralEmbedding(n_components=2, random_state=0, eigen_solver='arpack')
    return embedder.fit_transform(x)

def TSNE(x):
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    return tsne.fit_transform(x)
