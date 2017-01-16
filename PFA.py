import pandas as pd
from sklearn.cluster import KMeans
import numpy as np


def groupby(u_q_):
    d = {}
    for u in u_q_:
        if u[0] not in d:d[int(u[0])] = [u[1:]]
        else: d[int(u[0])].append(u[1:])
    return d


def getDataSets(file):
    datas = pd.read_csv(file).values
    label = datas[:,-1].astype(np.int)
    if np.any(label == 2):
        label -= 1
    return (datas[:,:-1],label)

def PFAdimension(x,q,p):
    covmatrix = np.cov(x.T,ddof=1)
    U,S,V = np.linalg.svd(covmatrix)
    n = len(U)
    u_q = U[:,:q]
    #
    kmeans = KMeans(n_clusters=p).fit(u_q)

    label = kmeans.labels_
    centers = kmeans.cluster_centers_
    init = 0
    centerdict = {}
    for cen in centers:
        centerdict[init] = cen
        init += 1
    u_q_ = np.c_[label,u_q,np.arange(n)]
    pos = []
    project = []
    d = groupby(u_q_)
    for (k,v) in d.iteritems():
        value = float('inf')
        index = 0
        pro = []
        for vs in v:
            if np.sum(np.abs(vs[:-1]-centerdict[k])) < value:
                value = np.sum(np.abs(vs[:-1]-centerdict[k]))
                index = vs[-1]
                pro = vs[:-1]
        pos.append(int(index))

        project.append(pro)
    pos = np.array(pos,dtype=int)
    return np.dot(x[:,pos],np.array(project))

# def IG(x,label):
#     p_0 = np.bincount(label)[0] / len(label)
#     p_1 = 1 - p_0
#     for feature in x.T:
#         np.
# x,label = getDataSets('suithoughtDepE19/SFFS100trainset.csv')
# for feature in x:
#     f = pd.cut(feature,2,labels=['0','1'])
#     print f[4]


# print np.shape(x)
# print PFAdimension(x,2,5)