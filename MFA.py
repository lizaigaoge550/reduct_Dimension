#coding=utf-8
import numpy as np
from scipy.spatial.distance import euclidean
from scipy.linalg import svd
def initial_W(n):
    return np.zeros([n,n]), np.zeros([n,n])

def compute_W(W_1,W_2,la,lb,D,k1,k2):
    for i in range(len(D)):
        k_1 , k_2, indexNow = 0,0,i
        for j in range(len(D[i])):
            if j == i:continue
            if k_1 < k1 and k_2 < k2:
                if D[i,j] in la:
                    W_1[i,D[i,j]] = 1
                    k_1 += 1
                elif D[i,j] in lb:
                    W_2[i,D[i,j]] = 1
                    k_2 += 1
            elif k_1 < k1:
                W_1[i,D[i,j]] = 1
                k_1 += 1
            elif k_2 < k2:
                W_2[i,D[i,j]] = 1
                k_2 += 1
            else:
                break
    return W_1,W_2



def construct(data,n):
    D = np.zeros([n,n])
    #计算每个点和其他点的距离
    for row in range(len(data)):
        for col in range(len(data)):
            D[row,col] = euclidean(data[row],data[col])
    #排序D
    D = np.argsort(D,axis=1)
    return D

def contructlabelList(label):
    u = np.unique(label)
    l = []
    for i in u:
        l.append(np.argwhere(label==i))
    return l[0],l[1]

def compute_D(W_1,W_2):
    d_1 = np.sum(W_1,axis=1)
    d_2 = np.sum(W_2,axis=1)

    return np.diag(d_1),np.diag(d_2)

def MFA(data,label,k1,k2,geshu):
    '''

    :param data: 数据 样本*特征 (n*m)
    :param k1: 类内
    :param k2: 类间
    :return: 投影矩阵
    '''
    n,m = np.shape(data)[0],np.shape(data)[1]
    #构造一个距离矩阵 D --> n*n
    D = construct(data,n)

    #构建类标存储list 即每个类的所有样本点的类标存储在一个list中
    la,lb = contructlabelList(label)

    #首先初始化两个矩阵 W_1,W_2, W_1放类内的距离相关值，W_2放类间距离相关值, 维度(n*n)
    W_1,W_2 = initial_W(n)

    #计算和每个样本点同类的最小距离k1个样本点,结果放入W_1, 不同类的最小距离k2个样本点, 结果放入W_2
    W_1,W_2 = compute_W(W_1,W_2,la.flatten(),lb.flatten(),D,k1,k2)


    #计算D_1,D_2
    D1,D2 = compute_D(W_1,W_2)


    intra = reduce(lambda a,b:np.dot(a,b),[data.T,D1-W_1,data])
    inter = reduce(lambda a,b:np.dot(a,b),[data.T,D2-W_2,data])
    a = np.dot(np.linalg.pinv(inter),intra)
    _,_,v = svd(a)
    return v[:,:geshu]


