#coding=utf-8
from __future__ import division
import numpy as np
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import euclidean

def costFunction(x_1,x_2,y,m1,m2,delta):
    cost = 0
    for i in range(len(x_1)):
        d= delta[0]*euclidean(x_1[i],m1) - delta[1]*euclidean(x_2[i],m2)
        cost += np.log(1+np.exp(-y[i]*d))
    return cost


def minbatchSGD(epoch,data_1,data_2,y,m1,m2,belta,parameter_1,parameter_2,learning_rate):
    fw = open('log','w')
    cost = costFunction(data_1,data_2,y,m1,m2,belta)
    d = belta
    eps = 1e-8
    m = np.zeros(len(belta));v = np.zeros(len(belta))
    while True:
        for i in range(len(data_1)):

            #计算h(a)函数
            d1 = euclidean(data_1[i],m1)
            d2 = euclidean(data_2[i],m2)
            fw.write('y: %s , d1-d2 %s'%(y[i],belta[0]*d1-belta[1]*d2) + '\n')
            h = np.array([d1, - d2])
            delta = 1/(1+np.exp(-y[i]*np.dot(belta,h.T)))*np.exp(-y[i]*np.dot(belta,h.T))*(-y[i])*h
            m = parameter_1*m + (1-parameter_1)*delta
            v = parameter_2*v + (1-parameter_2)*(delta**2)
            belta -= learning_rate*m/(np.sqrt(v)+eps)
        # 重新计算cost
        c = costFunction(data_1, data_2, y, m1, m2, belta)
        fw.write('Cost : ' + str(c)+'\n')
        if c < cost:
            cost = c
            d = belta
        else:
            break
    return d

def reduceDimension(projection,validX):
    valid_matrix_1 = np.zeros([len(validX), 10, 72])
    valid_matrix_2 = np.zeros([len(validX), 10, 72])
    # 对test做变换 test是len(test),320*72--> len(test),320,72
    validX = np.reshape(validX, [len(validX), 300, 72])
    for i in range(np.shape(validX)[-1]):
        m = validX[:, :, i]
        r1, r2 = projection[i]
        r_1 = np.dot(m, r1)
        r_2 = np.dot(m, r2)
        valid_matrix_1[:, :, i] = r_1
        valid_matrix_2[:, :, i] = r_2

    valid_1 = np.reshape(valid_matrix_1, [-1, np.shape(valid_matrix_1)[1] * np.shape(valid_matrix_1)[-1]])
    valid_2 = np.reshape(valid_matrix_2, [-1, np.shape(valid_matrix_2)[1] * np.shape(valid_matrix_2)[-1]])
    return valid_1,valid_2

