#coding=utf-8
import numpy as np


#n是472, 每个人8段, 一共59人, 其中4段做训练, 两段做验证(用于调参), 两段做测试, 也就是做4折的交叉验证
def batches(n):
    #先把472按8个隔开
    index = np.arange(n)
    interval = np.split(index,118)
    batch = []
    #batch_y = []
    for i in range(4):
        train = []
        valid = []
        test = []
        #train_1 = [];train_2 = []
        #test_1 = [];test_2 = []
        #valid_1 = [];valid_2 = []
        #train_y = []
        #test_y = []
        #valid_y = []
        #interval = [0,1,2,3,4,5,6,7] 其中
        # i = 0 , train -(0,4)(1,5) valid-(2,6)  test-(3,7)
        # i = 1 , train -(0,4)(1,5) valid-(3,7)  test-(2,6)
        # i = 2 , train -(0,4)(2,6) valid-(3,7)  test-(1,5)
        # i = 1 , train -(1,5)(2,6) valid-(3,7)  test-(0,4)
        for item in interval:
            #s = np.split(item,2)
            #获取组合
            #index = map(lambda a:list(a),zip(s[0],s[1]))
            if i == 0:
         #       train_1.extend([index[0][0],index[1][0],index[2][0]]);train_2.extend([index[0][1],index[1][1],index[2][1]])
                #train_1.extend();train_2.extend([])
         #       test_1.extend([index[-1][0]]);test_2.extend([index[-1][1]])
                #训练集取前2个，测试集最后一个
                train.extend([item[0],item[1]]);valid.extend([item[2]]);test.extend([item[-1]])
            elif i == 1:
                #train_1.extend([index[0][0],index[1][0],index[-1][0]]);train_2.extend([index[0][1],index[1][1],index[-1][1]])
                #train_1.extend([]);train_2.extend([])
                #test_1.extend([index[2][0]]);test_2.extend([index[2][1]])
                #训练集0,1,3 测试集2
                train.extend([item[0], item[1]]);valid.extend([item[-1]]); test.extend([item[2]])
            elif i == 2:
                #train_1.extend([index[0][0], index[1][0],index[-1][0]]);train_2.extend([index[0][1], index[1][1],index[1][0]])
                #train_1.extend([]);train_2.extend([])
                #test_1.extend([index[1][0]]);test_2.extend([index[1][1]])
                #训练集0,2,3 测试集1
                train.extend([item[0], item[2]]);valid.extend([item[-1]]);test.extend([item[1]])
            else:
                #train_1.extend([index[0][0], index[1][0],index[-1][0]]);train_2.extend([index[0][1], index[1][1],index[-1][1]])
                #train_1.extend([]);train_2.extend([])
                #test_1.extend([index[0][0]]);test_2.extend([index[0][1]])
                #训练集1,2,3 测试集0
                train.extend([item[1], item[2]]);valid.extend([item[-1]]);test.extend([item[0]])
        #train.extend(train_1);train.extend(train_2)
        #valid.extend(valid_1);valid.extend(valid_2)
        #test.extend(test_1);test.extend(test_2)
        batch.append([train,valid,test])
        #batch_y.append([train_y,valid_y,test_y])
    return batch

