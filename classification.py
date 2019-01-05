


# **import libraries and data**




import numpy as np
import pandas as pd 
import os
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from sklearn.model_selection import KFold
import random
from data import trainAp11,trainAp115,trainAp12,trainAp125,trainAp13,trainAp14,trainAp15,trainAp16,trainAp,testAp,apTestFull,apTrainFull




# ## Classification

# Methods that we tend to use are logistic regression, RF, GBDT and Gradient boost Machine. (temp) 

# ***Logistic Regression***




from sklearn.linear_model import LogisticRegression




##Firstly we did cross validation
##full dataset is apTrainFull_df
##divide train dataset into 10 folds
##use each fold as test data and other folds as train
##compute mean precision error of 10 folds.
def cvLog(trainDt,testDt,folds,cost,penalty,sol):
    
    ##cv accuracy
    kf = KFold(n_splits=folds,random_state=1)
    cvAccuracy=list()
    for train_idx, test_idx in kf.split(trainDt):
        train_tem,test_tem=trainDt.iloc[train_idx,:],trainDt.iloc[test_idx,:]
        y=train_tem['TARGET']
        X=train_tem.drop('TARGET',axis=1)
        clf = LogisticRegression(penalty=penalty,random_state=0,solver=sol,C=cost).fit(X, y)
        y1=test_tem['TARGET']
        X1=test_tem.drop('TARGET',axis=1)
        cvAccuracy.append(1-sum(abs(clf.predict(X1)-y1))/len(y1))
    cvAcc=sum(cvAccuracy)/len(cvAccuracy)
    
    ##train accuracy
    y_train=trainDt['TARGET']
    X_train=trainDt.drop('TARGET',axis=1)
    trainMod= LogisticRegression(penalty=penalty,random_state=0,solver=sol,C=cost).fit(X_train, y_train)
    trainAcc=1-sum(abs(trainMod.predict(X_train)-y_train))/len(y_train)
    ##test accuracy
    y_test=testDt['TARGET']
    X_test=testDt.drop('TARGET',axis=1)    
    testAcc=1-sum(abs(trainMod.predict(X_test)-y_test))/len(y_test)
    return([cvAcc,trainAcc,testAcc])





over_result=list()
for datasets in [trainAp11,trainAp115,trainAp12,trainAp125,trainAp13,trainAp14,trainAp15,trainAp16]:
    over_result.append(cvLog(datasets,testAp,10,1,'l2','newton-cg'))





accs=list()
for costs in [2.734**i for i in range(-5,5,1)]:
    accs.append(cvLog(trainAp,testAp,10,costs,'l2','newton-cg'))





accs





cv_acc=[i[0] for i in accs]
train_acc=[i[1] for i in accs]
test_acc=[i[2] for i in accs]
costs=[2.734**i for i in range(-5,5,1)]





plt.plot(costs,cv_acc)
plt.plot(costs,train_acc)
plt.plot(costs,test_acc)
plt.legend(['cv','train','test'])
plt.xlabel('costs')
plt.ylabel('accuracy')
plt.show()


# ***RF***




from sklearn.ensemble import RandomForestClassifier




##we separate train dataset into 10 folds
##choose 3 folds as test data and 7 folds as train data
##Firstly we did cross validation
##full dataset is apTrainFull_df
##divide train dataset into 10 folds
##use each fold as test data and other folds as train
##compute mean precision error of 10 folds.
def cvRF(trainDt,testDt,folds,trees,maxDep=10):
    
    ##cv accuracy
    kf = KFold(n_splits=folds,random_state=1)
    
    cvAccuracy=list()
    
    for train_idx, test_idx in kf.split(trainDt):
        train_tem,test_tem=trainDt.iloc[train_idx,:],trainDt.iloc[test_idx,:]
        y=train_tem['TARGET']
        X=train_tem.drop('TARGET',axis=1)
        
        clf=RandomForestClassifier(n_estimators=trees,max_depth=maxDep).fit(X, y)
        y1=test_tem['TARGET']
        X1=test_tem.drop('TARGET',axis=1)
        cvAccuracy.append(1-sum(abs(clf.predict(X1)-y1))/len(y1))
    
    cvAcc=sum(cvAccuracy)/len(cvAccuracy)
    
    ##train accuracy
    y_train=trainDt['TARGET']
    X_train=trainDt.drop('TARGET',axis=1)
    trainMod= RandomForestClassifier(n_estimators=trees,max_depth=maxDep).fit(X_train, y_train)
    trainAcc=1-sum(abs(trainMod.predict(X_train)-y_train))/len(y_train)
    
    ##test accuracy
    y_test=testDt['TARGET']
    X_test=testDt.drop('TARGET',axis=1)    
    testAcc=1-sum(abs(trainMod.predict(X_test)-y_test))/len(y_test)
    return([cvAcc,trainAcc,testAcc,trainMod.feature_importances_])


over_resultRF=list()
for datasets in [trainAp11,trainAp115,trainAp12,trainAp125,trainAp13,trainAp14,trainAp15,trainAp16]:
    over_resultRF.append(cvRF(trainAp,testAp,10,1000))



accsRF=list()
for trees in [10,100,1000,2000,3000,4000,5000,10000]:
    accsRF.append(cvRF(trainAp,testAp,10,trees))



cv_accRF=[i[0] for i in accsRF]
train_accRF=[i[1] for i in accsRF]
test_accRF=[i[2] for i in accsRF]
trees=[10,100,1000,2000,3000,4000,5000,10000]

plt.plot(trees,cv_accRF)
plt.plot(trees,train_accRF)
plt.plot(trees,test_accRF)
plt.legend(['cv','train','test'])
plt.xlabel('trees number')
plt.ylabel('accuracy')
plt.show()


# ***GBDT***

from sklearn.ensemble import GradientBoostingClassifier



##we separate train dataset into 10 folds
##choose 3 folds as test data and 7 folds as train data
##Firstly we did cross validation
##full dataset is apTrainFull_df
##divide train dataset into 10 folds
##use each fold as test data and other folds as train
##compute mean precision error of 10 folds.
def cvGBDT(trainDt,testDt,folds,lr):
    
    ##cv accuracy
    kf = KFold(n_splits=folds,random_state=1)
    
    cvAccuracy=list()
    
    for train_idx, test_idx in kf.split(trainDt):
        train_tem,test_tem=trainDt.iloc[train_idx,:],trainDt.iloc[test_idx,:]
        y=train_tem['TARGET']
        X=train_tem.drop('TARGET',axis=1)
        
        clf=GradientBoostingClassifier(loss='deviance', learning_rate=lr, n_estimators=100).fit(X, y)
        y1=test_tem['TARGET']
        X1=test_tem.drop('TARGET',axis=1)
        cvAccuracy.append(1-sum(abs(clf.predict(X1)-y1))/len(y1))
    
    cvAcc=sum(cvAccuracy)/len(cvAccuracy)
    
    ##train accuracy
    y_train=trainDt['TARGET']
    X_train=trainDt.drop('TARGET',axis=1)
    trainMod= GradientBoostingClassifier(loss='deviance', learning_rate=lr, n_estimators=100).fit(X_train, y_train)
    trainAcc=1-sum(abs(trainMod.predict(X_train)-y_train))/len(y_train)
    
    ##test accuracy
    y_test=testDt['TARGET']
    X_test=testDt.drop('TARGET',axis=1)    
    testAcc=1-sum(abs(trainMod.predict(X_test)-y_test))/len(y_test)
    return([cvAcc,trainAcc,testAcc,trainMod.feature_importances_])



over_resultG=list()
for datasets in [trainAp11,trainAp115,trainAp12,trainAp125,trainAp13,trainAp14,trainAp15,trainAp16]:
    over_resultRF.append(cvG(trainAp,testAp,10,1))



accsG=list()
for lrs in [2.734**i for i in range(-5,5,1)]:
    accsG.append(cvGBDT(trainAp,testAp,10,lrs))
    
cv_accG=[i[0] for i in accsG]
train_accG=[i[1] for i in accsG]
test_accG=[i[2] for i in accsG]
lrss=[2.734**i for i in range(-5,5,1)]

plt.plot(lrss,cv_accG)
plt.plot(lrss,train_accG)
plt.plot(lrss,test_accG)
plt.legend(['cv','train','test'])
plt.xlabel('learning rate')
plt.ylabel('accuracy')
plt.show()

