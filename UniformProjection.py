#coding=utf-8
import numpy as np
from PFA import getDataSets
from GramSchmidt import GramSchmidt
import numpy.linalg as la

def init(row,column):
    x = np.random.normal(size=(row,column))
    y = GramSchmidt(x)
    return y

def gaussian(sigma):
        return lambda x, y: \
            np.exp(-np.sqrt(la.norm(x-y) ** 2 / (2 * sigma ** 2)))

def covraince(x1):
    return np.corrcoef(x1,rowvar=False,ddof=1)

def uniform_project(x, y, d):
    c2 = x[np.where(y == 1)[0],:]
    c1 = x[np.where(y == 0)[0],:]


    a1 = covraince(c1);a2 = covraince(c2)
    #初始化U
    u = init(np.shape(c1)[1],d)
    thres = 0.01
    f = 0
    #迭代
    while True:
        C = reduce(np.dot,[a1,u,u.T,a1]) + reduce(np.dot,[a2,u,u.T,a2])
        f_ = np.trace(C)
        if (f_ - f) / f_ <= thres:break
        f = f_
        [_,_,v] = la.svd(C)
        u = v[:,:d]
    return u
# if __name__ == '__main__':
#     x,label = getDataSets('suithoughtDepE1a\\SFFS200trainset.csv')
#     print uniform_project(x,label,20)
