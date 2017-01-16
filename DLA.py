#coding=utf-8
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import scipy.spatial as spatial
import numpy as np
from scipy.spatial import distance
from scipy.linalg import svd

import matplotlib.pyplot as plt
#from classification import Cross_Validation
#from PFA import getDataSets

def find_different(points,all,y,t):
    label = []
    for i in range(len(all)):
        if all[i] in points:
            label.append(y[i])
    label = np.array(label)
    return  len(np.where(label != y[t])[0])

def radius(x,r,all,y,i):
    point_tree = spatial.cKDTree(all)
    points_index = point_tree.query_ball_point(x,r)
    points = all[np.array(points_index,dtype=int),:]
    return find_different(points,all,y,i)

def nearest(point,points,k1,y,label,type):
    dist = {}
    for i in range(len(points)):
        if np.array_equal(point,points[i]):continue
        if type == "tonglei":
            if y[i] != label :continue
        else:
            if y[i] == label:continue
        dist[i] = distance.euclidean(point,points[i])
    dist = sorted(dist.iteritems(),key=lambda (key,value):value,reverse=False)
    count = 0
    index = []
    while count < k1:
        index.append(dist[count][0])
        count += 1
    return index


def choose_identity_class(point,points,y,label,k1,k2):

    #找出和label 相同的那些点
    i = nearest(point,points,k1,y,label,'tonglei')

    #找出和label不同的那些点
    j = nearest(point,points,k2,y,label,'yilei')


    return i+j

def construct(s,k):
    c = 0
    for i in k:
        s[i,c] = 1
        c += 1

def DLA(x,y,r,k1,k2,d):
    #先用PCA降维
    pca_x = PCA().fit_transform(x)
    #取相同的类两个样本 k1 k2 不同的类两个样本 t1 , t2
    l = 0
    for i in range(len(pca_x)):
        #选择和这点最近的k1 k2 获取index
        k = choose_identity_class(pca_x[i],pca_x,y,y[i],k1,k2)
        k.append(i)
        #计算X的weight 在半径为1的范围内找样本
        n = radius(pca_x[i],r,pca_x,y,y[i])
        x_w = np.exp(-(1/(n+0.1)))

        #计算S
        #initial S
        s = np.zeros((np.shape(x)[0],k1+k2+1))
        construct(s,k)

        #计算L
        w = [1]*k1 + [-0.5]*k2
        w = np.diag(w)
        I = np.identity(k1+k2)
        e = np.ones(k1+k2)
        L = reduce(np.dot,[np.r_['0,2',-e,I],w,np.c_[-e,I]])

        l += reduce(np.dot,[s*x_w,L,s.T])
    _,_,v = svd(np.dot(np.dot(x.T,l),x))

    #v = v[::-1]
    return v[:d,:].T

# if __name__ == '__main__':
#     from sklearn import cross_validation
#
#     #x = np.r_['0,2',np.random.normal(0,1,(50,10)),np.random.normal(2,1,(50,10))]
#
#
#     cv = cross_validation.ShuffleSplit(len(x),n_iter=10,test_size=0.2,random_state=0)
#     #label = [0]* 50 + [1]*50
#     #label = np.array(label)
#
#     Cross_Validation(cv,x,label,SVC())
#     jieguo = np.dot(x,DLA(x,label,2,2,2,2).T)
#     print "**********jieguo***********"
#     #plt.scatter(jieguo[:50,0],jieguo[:50,1],marker='o',s = 100,color="red")
#     #plt.scatter(jieguo[50:,0],jieguo[50:,1],marker='x',s = 100,color="blue")
#     #plt.show()
#     Cross_Validation(cv,jieguo,label,SVC())