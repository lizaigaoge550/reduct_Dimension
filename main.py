#coding=utf-8
import scipy.io as io
from sklearn.decomposition import PCA
import numpy as np
import cPickle as pk
from scipy.spatial.distance import euclidean
from DLA2 import DLA
from utils import *
import random
from updata_parameter import minbatchSGD,reduceDimension
from sklearn.model_selection import KFold
from sklearn.svm import LinearSVC,SVC
#from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.model_selection import ShuffleSplit
from MFA import MFA
import matplotlib.pyplot as plt
'''
读取数据并分类
'''


def compute_count(data,range_):
    count = 0
    for i in range(len(data)):
        if data[i] > range_[i][0] and data[i] < range_[i][1]:
            count += 1
    return count


def compute_range(data):
    #获得最大值
    max_ = np.max(data,axis=0)
    min_ = np.min(data,axis=0)
    return zip(min_,max_)




def batches(n_split,n):
    batch = []

    class_1 = np.arange(n/2)
    class_2 = np.arange(n/2,n)
    step = (n/2)/n_split
    for s in range(n_split):
        if s + step < n/2:
            test_index = np.array([class_1[s:s+step],class_2[s:s+step]]).flatten()
            #train_index = np.array([class_1[s+step:],class_2[s+step:]]).flatten()
            train_index = np.array([np.delete(class_1,class_1[s:s+step]),np.delete(class_2,class_1[s:s+step])]).flatten()
            #每次留10个做validation
            #valid_index = np.array([train_index[0][:10],train_index[1][:10]]).flatten()
            #train_index = np.array([train_index[0][10:],train_index[1][10:]]).flatten()
        else:
            test_index = np.array([class_1[s:], class_2[s:]]).flatten()
            train_index = np.array([np.delete(class_1,class_1[s:]),np.delete(class_2,class_1[s:])]).flatten()
            #valid_index = np.array([train_index[0][:10], train_index[1][:10]]).flatten()
            #train_index = np.array([train_index[0][10:], train_index[1][10:]]).flatten()
        batch.append([train_index,test_index])
    return batch

#r1是320*10 r2是320*10 m是472*320
def compute(r1,r2,m):
    #先把m分成两个236*320
    return np.r_[np.dot(m[0:len(m)/2,:],r1),np.dot(m[len(m)/2:,:],r2)]


def remove_nan(m):
    length = len(m)
    return  np.reshape(m[~np.isnan(m)],[length,-1])


def standardization(m):
    return (m-np.mean(m,axis=0))/np.std(m,axis=0,ddof=1)


def Cross_Validation(cv,x,label,clf):
    trues = []
    pres = []
    for train_index,test_index in cv.split(x):
        trainX = x[train_index]
        trainY = label[train_index]

        test_x = x[test_index]
        testY = label[test_index]

        clf.fit(trainX,trainY)

        pres.extend(clf.predict(test_x))
        trues.extend(testY)
    print metrics.classification_report(trues,pres)

def preprocessing(data,n):
    #data是一个320*472*72
    pre_matrix = np.zeros([n,472,72])
    for i in range(np.shape(data)[-1]):
        m = data[:, :, i].T
        m = standardization(m)
        m = remove_nan(m)
        #m = 472*n_compoents
        m = PCA(n_components=n).fit_transform(m)
        pre_matrix[:,:,i] = m.T
    return pre_matrix

#data是一个320*472*72的矩阵,做10折交叉验证
data = io.loadmat('./matlab/happy_angry_gabor_result.mat')
matrix = data['r']
#y = np.tile([-1,-1,-1,-1,1,1,1,1],59)
y = np.sort(np.tile([-1,1],236))
#ten_fold = ShuffleSplit(n_splits=10)
n_componts = 300
true = []
Pre = []

x = preprocessing(matrix,n_componts)

#先把数据转换成[472,n_compoent*72]
x = np.reshape(x,[472,-1])

batch_x = batches(n = len(x))
for train_index,valid_index,test_index in  batch_x:
    trainX = x[train_index]
    trainY = y[train_index]

    validX = x[valid_index]
    validY = y[valid_index]

    testX = x[test_index]
    testY = y[test_index]

    result = np.zeros([len(trainX),10,72],dtype=np.float)
    trainX = np.reshape(trainX,[n_componts,len(trainX),72])
    #trainX = preprocessing(trainX,n_componts)

    projection = {}
    #f = False
    #matrix 是一个320*472*72维的矩阵, 其中320*236*72是happy, 320*236*72是angry

    for i in range(np.shape(trainX)[-1]):
        #print i
        #取出一个 320*472的矩阵，并且转置472*320
        m = trainX[:,:,i].T
        #np.savetxt('m.txt',m)
        #做standazition
        #m = standardization(m)
        #np.savetxt('m_.txt',m)
        #print i
        #m = remove_nan(m)
        #m = PCA(n_components=n_componts).fit_transform(m)
        r1,r2 = DLA(m,trainY,0.1,5,3,10)
        projection[i] = [r1,r2]
        r = compute(r1,r2,m)
        #r = MFA(m,y,5,5,20)
        #变成427*9的矩阵,保存在result中
        #result[:,:,i] = np.dot(m,r)
        result[:,:,i] = r
     #   if i == 7:
     #       f = True
        #result = result.astype(dtype=np.float)
    try:
        #np.savetxt('mfaresult.txt',result)
        pk.dump(projection,open('projection','wb'))
    except:
        #np.savez('mfaresult',result)
        pass
    #利用svm分类 有472个样本,每个样本的维度是9*72
    #result = 472,9,72 转变成 472,9*72
    data_matrix = np.reshape(result,[-1,np.shape(result)[1] * np.shape(result)[-1]])
    #计算两类的中心点
    mean_1 = np.mean(data_matrix[0:len(data_matrix)/2,:],axis=0)
    mean_2 = np.mean(data_matrix[len(data_matrix)/2:,:],axis=0)
    #np.savetxt('mean_1',mean_1)
    #np.savetxt('mean_2',mean_2)
    #range_1 = compute_range(data_matrix[0:len(data_matrix)/2])
    #range_2 = compute_range(data_matrix[len(data_matrix)/2:])
    #训练参数min-batch SGD 每次用一个样本来调参
    #alpha = random.random()
    #belta = random.random()
    #mean_1 = np.loadtxt(open('mean_1'))
    #mean_2 = np.loadtxt(open('mean_2'))
    #projection = pk.load('projection')
    #canshu = np.array([alpha,belta])
    valid_1,valid_2 = reduceDimension(projection,validX)
    #svm = LinearSVC()
    svm = SVC()

    features = np.r_['0,2',np.array(map(lambda a:euclidean(a,mean_1),valid_1)),np.array(map(lambda a:euclidean(a,mean_2),valid_2))].T
    svm.fit(features,validY)
    #belta_ = minbatchSGD(100, valid_1, valid_2, validY, mean_1, mean_2, canshu, 0.9,0.999,0.001)
    #qda =  QDA(data_matrix,trainY)
    #建立分类器
    #svm = LinearSVC()
    #svm.fit(data_matrix,trainY)

    test_matrix_1 = np.zeros([len(testX), 10, 72])
    test_matrix_2 = np.zeros([len(testX), 10, 72])
    #对test做变换 test是len(test),320*72--> len(test),320,72
    testX = np.reshape(testX,[len(testX),n_componts,72])
    for i in range(np.shape(testX)[-1]):
        m = testX[:,:,i]
        #m = standardization(m)
        #m = remove_nan(m)
        #m = PCA(n_components=n_componts).fit_transform(m)
        r1,r2 = projection[i]
        r_1 = np.dot(m,r1)
        r_2 = np.dot(m,r2)
        test_matrix_1[:,:,i] = r_1
        test_matrix_2[:,:,i] = r_2

    test_1= np.reshape(test_matrix_1,[-1,np.shape(test_matrix_1)[1]*np.shape(test_matrix_1)[-1]])
    test_2= np.reshape(test_matrix_2, [-1, np.shape(test_matrix_2)[1] * np.shape(test_matrix_2)[-1]])
    test_features = np.r_['0,2',np.array(map(lambda a:euclidean(a,mean_1),test_1)),np.array(map(lambda a:euclidean(a,mean_2),test_2))].T



    test_predict = svm.predict(test_features)
    #p_1 = svm.predict(test_1)
    #p_2 = svm.predict(test_2)
    # for i in range(len(p_1)):
    #     if p_1[i] == p_2[i]:test_predict.append(p_1[i])
    #     else:
    #         if euclidean(test_1[i], mean_1) > euclidean(test_2[i], mean_2):
    #             test_predict.append(1)
    #         elif euclidean(test_1[i], mean_1) > euclidean(test_2[i], mean_2):
    #             test_predict.append(-1)

    #计算类标
    # for i in range(len(test_1)):
    #     #计算距离
    #     if belta_[0]*euclidean(test_1[i],mean_1) > belta_[1]*euclidean(test_2[i],mean_2):
    #     #if compute_count(test_1[i],range_1) < compute_count(test_2[i],range_2):
    #        test_predict.append(1)
    #     #elif compute_count(test_1[i],range_1) > compute_count(test_2[i],range_2):
    #     elif belta_[0]*euclidean(test_1[i],mean_1) < belta_[1]*euclidean(test_2[i],mean_2):
    #        test_predict.append(-1)
    #     else:
    #        if np.random.uniform(0,1) > 0.5:
    #            test_predict.append(1)
    #        else:
    #         test_predict.append(-1)
    true.extend(testY)
    Pre.extend(test_predict)

    cv = ShuffleSplit()
    cv.split()
    #C_range = np.logspace(-5, 15, 100,base=2)
    #10折交叉验证
    #ten_fold = KFold(n_splits=10)
    #for c in C_range:
    #svm = LinearSVC()
    #print "C : %s"%c
    #Cross_Validation(ten_fold,data_matrix,y,svm)
    #svm.fit(data_matrix,trainY)
    #svm.predict(test_)
print metrics.classification_report(true,Pre)