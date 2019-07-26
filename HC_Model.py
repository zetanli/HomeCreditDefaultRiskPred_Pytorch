#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from scipy.stats import linregress
from sklearn.metrics import recall_score
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import KFold


# In[2]:


dt = pd.read_csv("HCDR.csv")


# In[3]:


###undersample
rows = dt.loc[dt['TARGET']==1].shape[0]
class0 = dt.loc[dt['TARGET']==0].sample(rows,random_state=111).reset_index(drop=True)
class1 = dt.loc[dt['TARGET']==1].reset_index(drop=True)
dt2 = pd.concat([class0,class1],axis = 0).reset_index(drop=True)


# In[3]:


###over sample

rows0 = dt.loc[dt['TARGET']==0].shape[0]
class1 = dt.loc[dt['TARGET']==1].sample(rows0,replace = True,random_state=111).reset_index(drop=True)
class0 = dt.loc[dt['TARGET']==0].reset_index(drop=True)
dt2 = pd.concat([class0,class1],axis = 0).reset_index(drop=True)


# In[4]:


###over sample is better


# In[5]:


Y = dt2['TARGET']
X = dt2.drop( ['TARGET','SK_ID_CURR'],axis=1)


# In[6]:


columns = list(X.columns)


# In[7]:


X = preprocessing.scale(X)


# In[8]:


X = pd.DataFrame(X,columns = columns)


# In[9]:


X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size = 0.05,random_state = 111)


# In[10]:


X_train,X_val,y_train,y_val = train_test_split(X_train,y_train,test_size = 0.25,random_state = 222)


# In[11]:


columns = list(X.columns)


# In[12]:


clf = RandomForestClassifier(n_estimators=1000, random_state=0, n_jobs=-1)
clf.fit(X_train, y_train)


# In[13]:


importance = pd.DataFrame({'columns':list(X.columns),'fi':clf.feature_importances_})


# In[14]:


importance = importance.sort_values(by='fi',ascending = False).reset_index(drop=True)


# In[15]:


i1 = list(importance.iloc[:100,0])


# In[ ]:


##Logistics Regression


# In[16]:


lr = LogisticRegression(penalty = 'l2',C=0.01)


# In[17]:


lr.fit(X_train[i1],y_train)


# In[18]:


y_pred = lr.predict(X_val[i1])


# In[19]:


recall_score(y_val,y_pred)


# In[8]:


### DNN


# In[16]:


from torch import tensor
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# In[17]:


X_train_tensor = torch.from_numpy(X_train[i1].values.astype(np.float32))
y_train_tensor = torch.from_numpy(y_train.astype(np.float32).values)
X_v_tensor = torch.from_numpy(X_val[i1].values.astype(np.float32))
y_v_tensor = torch.from_numpy(y_val.astype(np.float32).values)


# In[18]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# In[19]:


class Net1(nn.Module):
    def __init__(self,n_features,n_hidden,n_out):
        super(Net1,self).__init__()
        ##input combination
        self.fc1 = nn.Linear(n_features,n_hidden)
        self.fc2 = nn.Linear(n_hidden,n_out)
        
    def forward(self,x):
        
        ##hidden output
        x = F.tanh(self.fc1(x))
        
        ##out output
        x = F.sigmoid(self.fc2(x))
        
        return x
    def label(self,x):
        res = self.forward(x)
        ans = []
        for i in res:
            if i[0]>0.5:
                ans.append(1)
            else:
                ans.append(0)
                
        return tensor(ans)
    
class Net2(nn.Module):
    def __init__(self,n_features,n_hidden1,n_hidden2,n_out):
        super(Net2,self).__init__()
        ##input combination
        self.fc1 = nn.Linear(n_features,n_hidden1)
        self.fc2 = nn.Linear(n_hidden1,n_hidden2)
        self.fc3 = nn.Linear(n_hidden2,n_out)
    def forward(self,x):
        
        ##hidden output
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        ##out output
        x = F.sigmoid(self.fc3(x))
        
        return x
    def label(self,x):
        res = self.forward(x)
        ans = []
        for i in res:
            if i[0]>0.5:
                ans.append(1)
            else:
                ans.append(0)
                
        return tensor(ans)
    
class Net3(nn.Module):
    def __init__(self,n_features,n_hidden1,n_hidden2,n_hidden3,n_out):
        super(Net3,self).__init__()
        
        ##input combination
        self.fc1 = nn.Linear(n_features,n_hidden1)
        self.fc2 = nn.Linear(n_hidden1,n_hidden2)
        self.fc3 = nn.Linear(n_hidden2,n_hidden3)
        self.fc4 = nn.Linear(n_hidden3,n_out)
        
    def forward(self,x):
        
        ##hidden output
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = F.tanh(self.fc3(x))
        ##out output
        x = F.sigmoid(self.fc4(x))
        
        return x
    def label(self,x):
        res = self.forward(x)
        ans = []
        for i in res:
            if i[0]>0.5:
                ans.append(1)
            else:
                ans.append(0)
                
        return tensor(ans)


# In[2]:


###do cross validation to choose best hyperparameters
###tunning parameters are hidden layers, learning rates, echos, hidden nodes


# In[21]:


def tune1(X,y,lra,echos,hidden_nodes,Xcv,ycv):
    m = Net1(X.shape[1],hidden_nodes,1)
    m = m.to(device)
    opt = torch.optim.SGD(m.parameters(),lr=lra)
    loss = torch.nn.BCELoss()
    for t in range(echos):
        pred = m.forward(X)
        los = loss(pred,y)
        opt.zero_grad()
        los.backward()
        opt.step()
        
    cvpred = m.label(Xcv)
    cvpred = cvpred.cpu()
    recall = recall_score(cvpred,ycv)
    return recall


def tune2(X,y,lra,echos,hid1,hid2,Xcv,ycv):
    m = Net2(X.shape[1],hid1,hid2,1)
    m = m.to(device)
    opt = torch.optim.SGD(m.parameters(),lr=lra)
    loss = torch.nn.BCELoss()
    for t in range(echos):
        pred = m.forward(X)
        los = loss(pred,y)
        opt.zero_grad()
        los.backward()
        opt.step()
        
    cvpred = m.label(Xcv)
    cvpred = cvpred.cpu()
    recall = recall_score(cvpred,ycv)
    return recall
        
    
def tune3(X,y,lra,echos,hid1,hid2,hid3,Xcv,ycv):
    m = Net3(X.shape[1],hid1,hid2,hid3,1)
    m = m.to(device)
    opt = torch.optim.SGD(m.parameters(),lr=lra)
    loss = torch.nn.BCELoss()
    for t in range(echos):
        pred = m.forward(X)
        los = loss(pred,y)
        opt.zero_grad()
        los.backward()
        opt.step()
        
    cvpred = m.label(Xcv)
    cvpred = cvpred.cpu()
    recall = recall_score(cvpred,ycv)
    return recall


# In[22]:


hids = [50,100,500,1000]
lras = [0.5,0.1,0.05,0.01,0.001]
echos = [100,500,1000,5000,10000]


# In[23]:


##5folds
kf = KFold(n_splits = 5,random_state=222)


# In[24]:


##1 hidden layer
def cv1(trainX,trainy,lra,echos,hid1):
    recalls = 0
    for train,test in kf.split(trainX):
        tmp_train_X = trainX[train,]
        tmp_train_y = trainy[train,]
        tmp_test_X = trainX[test,]
        tmp_test_y = trainy[test,]
        tmp_train_X = tmp_train_X.to(device)
        tmp_train_y = tmp_train_y.to(device)
        tmp_test_X = tmp_test_X.to(device)

        re1 = tune1(tmp_train_X,tmp_train_y,lra,echos,hid1,tmp_test_X,tmp_test_y)
        recalls+=re1
    
    return(recalls/5)


# In[25]:


h = list()
l = list()
e = list()
res = list()
times=0
for i in hids:
    for j in lras:
        for k in echos:
            h.append(i)
            l.append(j)
            e.append(k)
            times += 1
            print(times)
            res.append(cv1(X_train_tensor,y_train_tensor,j,k,i))


# In[44]:



def cv2(trainX,trainy,lra,echos,hid1,hid2):
    recalls = 0
    for train,test in kf.split(trainX):
        tmp_train_X = trainX[train,]
        tmp_train_y = trainy[train,]
        tmp_test_X = trainX[test,]
        tmp_test_y = trainy[test,]
        tmp_train_X = tmp_train_X.to(device)
        tmp_train_y = tmp_train_y.to(device)
        tmp_test_X = tmp_test_X.to(device)

        re1 = tune2(tmp_train_X,tmp_train_y,lra,echos,hid1,hid2,tmp_test_X,tmp_test_y)
        recalls+=re1
    
    return(recalls/5)


# In[45]:


h1 = list()
h2 = list()
l2 = list()
e2 = list()
res2 = list()
for i in hids:
    for n in hids:
        for j in lras:
            for k in echos:
                h1.append(i)
                h2.append(n)
                l2.append(j)
                e2.append(k)
                print(str(i) + ' ' + str(n) + ' ' + str(j) + ' '+str(k))
                res2.append(cv2(X_train_tensor,y_train_tensor,j,k,i,n))


# In[101]:


def cv3(trainX,trainy,lra,echos,hid1,hid2,hid3):
    recalls = 0
    for train,test in kf.split(trainX):
        tmp_train_X = trainX[train,]
        tmp_train_y = trainy[train,]
        tmp_test_X = trainX[test,]
        tmp_test_y = trainy[test,]
        tmp_train_X = tmp_train_X.to(device)
        tmp_train_y = tmp_train_y.to(device)
        tmp_test_X = tmp_test_X.to(device)

        re1 = tune3(tmp_train_X,tmp_train_y,lra,echos,hid1,hid2,hid3,tmp_test_X,tmp_test_y)
        recalls+=re1
    
    return(recalls/5)


# In[ ]:


hi1 = list()
hi2 = list()
hi3 = list()
l3 = list()
e3 = list()
res3 = list()
for i in hids:
    for n in hids:
        for j in lras:
            for k in echos:
                for q in hids:
                hi1.append(i)
                hi2.append(n)
                hi3.append(n)
                l3.append(j)
                e3.append(k)
                print(str(i) + ' ' + str(n) + ' ' + str(j) + ' '+str(k))
                res3.append(cv3(X_train_tensor,y_train_tensor,j,k,i,n,q))


# In[ ]:


dt_n1 = pd.DataFrame({'hidden_nodes1':h,'lr':l,'epoch':e,'recall':res})


# In[ ]:


dt_n2 = pd.DataFrame({'hidden_nodes1':h1,'hidden_nodes2':h2,'lr':l2,'epoch':e2,'recall':res2})


# In[48]:


dt_n3 = pd.DataFrame({'hidden_nodes1':h1,'hidden_nodes2':h2,'hidden_nodes2':h3,'lr':l2,'epoch':e2,'recall':res2})


# In[ ]:


pd.concat([dt_n1,dt_n2,dt_n3],axis = 1)['recall'].min()


# In[51]:


dt_n2.loc[dt_n2['recall'] == dt_n2['recall'].max()]


# In[57]:


X_test_tensor = torch.from_numpy(X_test[i1].values.astype(np.float32))
y_test_tensor = torch.from_numpy(y_test.values.astype(np.float32))
X_test_tensor = X_test_tensor.to(device)


# In[97]:


m2 = Net2(X_train_tensor.shape[1],1000,1000,1)
m2 = m2.to(device)
X_train_tensor = X_train_tensor.to(device)
y_train_tensor = y_train_tensor.to(device)
X_v_tensor=X_v_tensor.to(device)
optimizer = torch.optim.SGD(m2.parameters(),lr=0.5)
loss_func = torch.nn.BCELoss()
for t in range(10000):
    prediction = m2.forward(X_train_tensor)
    loss = loss_func(prediction,y_train_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
vpred = m2.label(X_v_tensor)
vpred = vpred.cpu()
recall_score(vpred,y_v_tensor)


# In[84]:


pred = m2.label(X_v_tensor)
pred = pred.cpu()
recall_score(pred,y_v_tensor)


# In[85]:


pred_prob = m2.forward(X_test_tensor)


# In[86]:


pred_prob = pred_prob.cpu()


# In[87]:


import numpy as np
from sklearn import metrics


# In[88]:


fpr, tpr, thresholds = metrics.roc_curve(y_v_tensor.detach().numpy(), pred.detach().numpy(), pos_label=1)


# In[91]:


metrics.auc(fpr, tpr)

