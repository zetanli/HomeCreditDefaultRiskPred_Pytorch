#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.model_selection import KFold
import random
from functools import reduce
from statistics import mode
import re


# In[2]:


##import train dataset


# In[3]:


#per row per client
app_train = 'application_train.csv'

bureau = 'bureau.csv'
previous_app = 'previous_application.csv'

applications = pd.read_csv(app_train)
bureau = pd.read_csv(bureau)
pre_app = pd.read_csv(previous_app)

##monthly for client
bureau_balance = 'bureau_balance.csv'
pos_balance = 'POS_CASH_balance.csv'
credit_card_balance = 'credit_card_balance.csv'

bu_balance = pd.read_csv(bureau_balance)
card_balance = pd.read_csv(credit_card_balance)
pos_balance = pd.read_csv(pos_balance)

##not sure
installments_payment = 'installments_payments.csv'

install = pd.read_csv(installments_payment)


# In[4]:


##impute missing value
##delete columns missing value rate > 0.3
##fill columns as some moments


# In[5]:


pre_keep = list()
app_keep = list()
bu_keep = list()
bu_ba_keep = list()
card_keep = list()
pos_keep = list()
ins_keep = list()

pre_tmp = pre_app.isna().mean().reset_index()
app_tmp = applications.isna().mean().reset_index()
bu_tmp = bureau.isna().mean().reset_index()
bu_ba_tmp = bu_balance.isna().mean().reset_index()
card_tmp = card_balance.isna().mean().reset_index()
pos_tmp = pos_balance.isna().mean().reset_index()
ins_tmp = install.isna().mean().reset_index()


# In[6]:


for i in range(pre_tmp.shape[0]):
    if pre_tmp[0][i] < 0.3:
        pre_keep.append(pre_tmp['index'][i])
for i in range(app_tmp.shape[0]):
    if app_tmp[0][i] < 0.3:
        app_keep.append(app_tmp['index'][i])
for i in range(bu_tmp.shape[0]):
    if bu_tmp[0][i] < 0.3:
        bu_keep.append(bu_tmp['index'][i])
        
for i in range(bu_ba_tmp.shape[0]):
    if bu_ba_tmp[0][i] < 0.3:
        bu_ba_keep.append(bu_ba_tmp['index'][i])
        
for i in range(card_tmp.shape[0]):
    if card_tmp[0][i] < 0.3:
        card_keep.append(card_tmp['index'][i])
        
for i in range(pos_tmp.shape[0]):
    if pos_tmp[0][i] < 0.3:
        pos_keep.append(pos_tmp['index'][i])
        
for i in range(ins_tmp.shape[0]):
    if ins_tmp[0][i] < 0.3:
        ins_keep.append(ins_tmp['index'][i])

applications = applications[app_keep]
pre_app = pre_app[pre_keep]
bureau = bureau[bu_keep]
bu_balance = bu_balance[bu_ba_keep]
card_balance = card_balance[card_keep]
pos_balance = pos_balance[pos_keep]
install = install[ins_keep]


# In[7]:


##fill na
pre_app[['AMT_ANNUITY','AMT_CREDIT','AMT_GOODS_PRICE','CNT_PAYMENT']] = pre_app[['AMT_ANNUITY','AMT_CREDIT',
    'AMT_GOODS_PRICE','CNT_PAYMENT']].fillna(pre_app[['AMT_ANNUITY','AMT_CREDIT','AMT_GOODS_PRICE','CNT_PAYMENT']].mean())

pre_app[['PRODUCT_COMBINATION']] = pre_app[['PRODUCT_COMBINATION']].fillna(mode(pre_app['PRODUCT_COMBINATION']))

applications[['AMT_ANNUITY','AMT_GOODS_PRICE','EXT_SOURCE_2','EXT_SOURCE_3',
    'OBS_30_CNT_SOCIAL_CIRCLE','DEF_30_CNT_SOCIAL_CIRCLE','OBS_60_CNT_SOCIAL_CIRCLE',
    'DEF_60_CNT_SOCIAL_CIRCLE']] = applications[['AMT_ANNUITY','AMT_GOODS_PRICE','EXT_SOURCE_2',
    'EXT_SOURCE_3','OBS_30_CNT_SOCIAL_CIRCLE','DEF_30_CNT_SOCIAL_CIRCLE','OBS_60_CNT_SOCIAL_CIRCLE',
    'DEF_60_CNT_SOCIAL_CIRCLE']].fillna(applications[['AMT_ANNUITY','AMT_GOODS_PRICE','EXT_SOURCE_2',
    'EXT_SOURCE_3','OBS_30_CNT_SOCIAL_CIRCLE','DEF_30_CNT_SOCIAL_CIRCLE','OBS_60_CNT_SOCIAL_CIRCLE',
     'DEF_60_CNT_SOCIAL_CIRCLE']].mean())

applications[['CNT_FAM_MEMBERS','DAYS_LAST_PHONE_CHANGE','AMT_REQ_CREDIT_BUREAU_HOUR','AMT_REQ_CREDIT_BUREAU_DAY',
    'AMT_REQ_CREDIT_BUREAU_WEEK','AMT_REQ_CREDIT_BUREAU_MON','AMT_REQ_CREDIT_BUREAU_QRT',
    'AMT_REQ_CREDIT_BUREAU_YEAR']] = applications[['CNT_FAM_MEMBERS','DAYS_LAST_PHONE_CHANGE','AMT_REQ_CREDIT_BUREAU_HOUR',
    'AMT_REQ_CREDIT_BUREAU_DAY','AMT_REQ_CREDIT_BUREAU_WEEK','AMT_REQ_CREDIT_BUREAU_MON','AMT_REQ_CREDIT_BUREAU_QRT',
    'AMT_REQ_CREDIT_BUREAU_YEAR']].fillna(round(applications[['CNT_FAM_MEMBERS','DAYS_LAST_PHONE_CHANGE',
    'AMT_REQ_CREDIT_BUREAU_HOUR','AMT_REQ_CREDIT_BUREAU_DAY','AMT_REQ_CREDIT_BUREAU_WEEK','AMT_REQ_CREDIT_BUREAU_MON',
    'AMT_REQ_CREDIT_BUREAU_QRT','AMT_REQ_CREDIT_BUREAU_YEAR']].mean())) 


applications['NAME_TYPE_SUITE'] = applications['NAME_TYPE_SUITE'].fillna(mode(applications['NAME_TYPE_SUITE']))

bureau[['DAYS_CREDIT_ENDDATE']] = bureau[['DAYS_CREDIT_ENDDATE']].fillna(round(bureau[['DAYS_CREDIT_ENDDATE']].mean()))

bureau[['AMT_CREDIT_SUM','AMT_CREDIT_SUM_DEBT']] = bureau[['AMT_CREDIT_SUM',
    'AMT_CREDIT_SUM_DEBT']].fillna(bureau[['AMT_CREDIT_SUM','AMT_CREDIT_SUM_DEBT']].mean())

card_balance[['AMT_DRAWINGS_ATM_CURRENT', 'AMT_DRAWINGS_OTHER_CURRENT',
       'AMT_DRAWINGS_POS_CURRENT', 'AMT_INST_MIN_REGULARITY',
       'AMT_PAYMENT_CURRENT']] = card_balance[['AMT_DRAWINGS_ATM_CURRENT', 'AMT_DRAWINGS_OTHER_CURRENT',
       'AMT_DRAWINGS_POS_CURRENT', 'AMT_INST_MIN_REGULARITY',
       'AMT_PAYMENT_CURRENT']].fillna(card_balance[['AMT_DRAWINGS_ATM_CURRENT', 'AMT_DRAWINGS_OTHER_CURRENT',
       'AMT_DRAWINGS_POS_CURRENT', 'AMT_INST_MIN_REGULARITY',
       'AMT_PAYMENT_CURRENT']].mean())

card_balance[['CNT_DRAWINGS_ATM_CURRENT', 'CNT_DRAWINGS_OTHER_CURRENT',
    'CNT_DRAWINGS_POS_CURRENT', 'CNT_INSTALMENT_MATURE_CUM']] = card_balance[['CNT_DRAWINGS_ATM_CURRENT', 
    'CNT_DRAWINGS_OTHER_CURRENT','CNT_DRAWINGS_POS_CURRENT', 
    'CNT_INSTALMENT_MATURE_CUM']].fillna(round(card_balance[['CNT_DRAWINGS_ATM_CURRENT', 'CNT_DRAWINGS_OTHER_CURRENT',
    'CNT_DRAWINGS_POS_CURRENT', 'CNT_INSTALMENT_MATURE_CUM']]))

pos_balance[['CNT_INSTALMENT', 'CNT_INSTALMENT_FUTURE']] = pos_balance[['CNT_INSTALMENT', 
    'CNT_INSTALMENT_FUTURE']].fillna(round(pos_balance[['CNT_INSTALMENT', 'CNT_INSTALMENT_FUTURE']].mean()))

install['DAYS_ENTRY_PAYMENT'] = install['DAYS_ENTRY_PAYMENT'].fillna(round(install['DAYS_ENTRY_PAYMENT'].mean()))

install['AMT_PAYMENT'] = install['AMT_PAYMENT'].fillna(install['AMT_PAYMENT'].mean())


# In[8]:


###aggregate other dataset


# In[9]:


##aggregate Pre_app
##check info
pre_app.info()


# In[10]:


pre_app.columns


# In[11]:


pre_dummies = pd.get_dummies(pre_app[['NAME_CONTRACT_TYPE','WEEKDAY_APPR_PROCESS_START','FLAG_LAST_APPL_PER_CONTRACT','NAME_CASH_LOAN_PURPOSE', 
         'NAME_CONTRACT_STATUS','NAME_PAYMENT_TYPE', 'CODE_REJECT_REASON', 'NAME_CLIENT_TYPE','NAME_GOODS_CATEGORY', 
         'NAME_PORTFOLIO', 'NAME_PRODUCT_TYPE','CHANNEL_TYPE','NAME_SELLER_INDUSTRY','NAME_YIELD_GROUP', 
         'PRODUCT_COMBINATION']])
pre_continuous = pre_app.drop(['NAME_CONTRACT_TYPE','WEEKDAY_APPR_PROCESS_START','FLAG_LAST_APPL_PER_CONTRACT','NAME_CASH_LOAN_PURPOSE', 
         'NAME_CONTRACT_STATUS','NAME_PAYMENT_TYPE', 'CODE_REJECT_REASON', 'NAME_CLIENT_TYPE','NAME_GOODS_CATEGORY', 
         'NAME_PORTFOLIO', 'NAME_PRODUCT_TYPE','CHANNEL_TYPE','NAME_SELLER_INDUSTRY','NAME_YIELD_GROUP', 
         'PRODUCT_COMBINATION'],axis=1)

##corr for continuous columns
##delete the corr larger than 0.7
pre_ctmp = pre_continuous.drop(['SK_ID_PREV','SK_ID_CURR'],axis = 1).corr().unstack().reset_index()
pre_ctmp[pre_ctmp[0]!=1][pre_ctmp[0]>=0.7]
#delete 'AMT_ANNUITY','AMT_CREDIT','AMT_GOODS_PRICE'
pre_continuous = pre_continuous.drop(['AMT_ANNUITY','AMT_CREDIT','AMT_GOODS_PRICE'],axis = 1)


# In[12]:


pre_app = pd.concat([pre_continuous,pre_dummies],axis = 1).drop(['SK_ID_PREV'],axis=1)
cols = list()
for i in list(pre_app.columns):
    if re.findall(r'.\_XNA$',i) == []:
        cols.append(i)
        
cols2 = list()
for i in cols:
    if re.findall(r'.\_XAP$',i) == []:
        cols2.append(i)    
pre_app = pre_app[cols2]
pre_app_group = pre_app.groupby('SK_ID_CURR')
pre_app = pre_app_group.sum().reset_index()


# In[13]:




##corr for continuous columns
##delete the corr larger than 0.7
pre_tmp = pre_app.drop(['SK_ID_CURR'],axis = 1).corr().unstack().reset_index()
pre_tmp[pre_tmp[0]!=1][pre_tmp[0]>=0.7].iloc[100:150,:2]

delete = ['CNT_PAYMENT','NAME_CONTRACT_TYPE_Cash loans','NAME_PORTFOLIO_Cash','NFLAG_LAST_APPL_IN_DAY',
          'FLAG_LAST_APPL_PER_CONTRACT_Y','NAME_CONTRACT_STATUS_Approved','NAME_CONTRACT_STATUS_Refused',
         'NAME_PAYMENT_TYPE_Cash through the bank','NAME_CLIENT_TYPE_Repeater','NAME_PORTFOLIO_Cash',
          'CHANNEL_TYPE_Credit and cash offices','PRODUCT_COMBINATION_Cash','HOUR_APPR_PROCESS_START',
          'NAME_PRODUCT_TYPE_x-sell','NAME_SELLER_INDUSTRY_Clothing','NAME_SELLER_INDUSTRY_Construction',
          'NAME_SELLER_INDUSTRY_Furniture','NAME_SELLER_INDUSTRY_Connectivity','PRODUCT_COMBINATION_POS mobile with interest',
         'CHANNEL_TYPE_Car dealer','NAME_CONTRACT_TYPE_Revolving loans','PRODUCT_COMBINATION_Card X-Sell',
          'NAME_CONTRACT_TYPE_Consumer loans','PRODUCT_COMBINATION_POS household with interest',
          'PRODUCT_COMBINATION_Cash X-Sell: middle']

pre_app = pre_app.drop(delete,axis=1)


# In[14]:


###aggregate bureau
bureau.head(10)


# In[15]:


bureau.info()


# In[16]:


bureau.columns


# In[17]:


bu_dum = pd.get_dummies(bureau[['CREDIT_ACTIVE', 'CREDIT_CURRENCY','CREDIT_TYPE']])
bu_con = bureau.drop(['CREDIT_ACTIVE', 'CREDIT_CURRENCY','CREDIT_TYPE'],axis=1)


# In[18]:


##corr for continuous columns
##delete the corr larger than 0.7
bu_tmp = bu_con.drop(['SK_ID_CURR', 'SK_ID_BUREAU'],axis=1).corr().unstack().reset_index()
bu_tmp[bu_tmp[0]!=1][bu_tmp[0]>=0.7]


# In[19]:


bureau = pd.concat([bu_con,bu_dum],axis=1).drop(['SK_ID_BUREAU'],axis=1)
bu_group = bureau.groupby('SK_ID_CURR')
bureau = bu_group.sum().reset_index()
bu_tmp = bureau.drop(['SK_ID_CURR'],axis = 1).corr().unstack().reset_index()
bu_tmp[bu_tmp[0]!=1][bu_tmp[0]>=0.7]
delete = ['DAYS_CREDIT_UPDATE','CREDIT_CURRENCY_currency 1','CREDIT_TYPE_Consumer credit']
bureau = bureau.drop(delete,axis=1)


# In[20]:


##aggregate install

ins_tmp = install.drop(['SK_ID_CURR', 'SK_ID_PREV'],axis=1).corr().unstack().reset_index()
ins_tmp[ins_tmp[0]!=1][ins_tmp[0]>=0.7]
install = install.drop(['DAYS_ENTRY_PAYMENT','AMT_INSTALMENT','SK_ID_PREV'],axis = 1)


# In[21]:


ins_group = install.groupby('SK_ID_CURR')
install = ins_group.sum().reset_index()


# In[22]:


##aggregate pos
pos_balance = pos_balance[['SK_ID_CURR', 'MONTHS_BALANCE', 'CNT_INSTALMENT',
       'CNT_INSTALMENT_FUTURE', 'SK_DPD',
       'SK_DPD_DEF']]


# In[23]:


pos_tmp = pos_balance.drop(['SK_ID_CURR'],axis=1).corr().unstack().reset_index()
pos_tmp[pos_tmp[0]!=1][pos_tmp[0]>=0.7]


# In[24]:


pos_balance = pos_balance[['SK_ID_CURR', 'MONTHS_BALANCE', 'CNT_INSTALMENT', 'SK_DPD','SK_DPD_DEF']]
pos_group = pos_balance.groupby('SK_ID_CURR')


# In[25]:


pos_balance = pos_group.sum().reset_index()


# In[26]:


##aggregate credit card
card_balance.columns


# In[27]:


card_cont = card_balance.drop(['NAME_CONTRACT_STATUS'],axis = 1)
card_dummie = card_balance['NAME_CONTRACT_STATUS']


# In[28]:


c_tmp = card_cont.drop(['SK_ID_PREV', 'SK_ID_CURR'],axis=1).corr().unstack().reset_index()
c_tmp[c_tmp[0]!=1][c_tmp[0]>=0.7]


# In[29]:


delete = ['AMT_INST_MIN_REGULARITY','AMT_RECEIVABLE_PRINCIPAL','AMT_RECIVABLE','AMT_TOTAL_RECEIVABLE',
         'AMT_DRAWINGS_ATM_CURRENT','CNT_DRAWINGS_ATM_CURRENT','CNT_DRAWINGS_POS_CURRENT','AMT_PAYMENT_TOTAL_CURRENT']
card_cont = card_cont.drop(delete,axis=1)


# In[30]:


card_dum = pd.get_dummies(card_dummie).drop('Completed',axis=1)


# In[31]:


card = pd.concat([card_cont,card_dum],axis=1)


# In[32]:


card_group = card.groupby('SK_ID_CURR')
card = card_group.sum().reset_index().drop('SK_ID_PREV',axis = 1)


# In[33]:


c_tmp = card.drop(['SK_ID_CURR'],axis=1).corr().unstack().reset_index()
c_tmp[c_tmp[0]!=1][c_tmp[0]>=0.7]


# In[34]:


card = card.drop(['AMT_DRAWINGS_POS_CURRENT','AMT_PAYMENT_CURRENT','SK_DPD_DEF',
                  'CNT_INSTALMENT_MATURE_CUM'],axis = 1)


# In[35]:


###deal with current application
applications.columns


# In[36]:


applications.info()


# In[37]:


app_dum = pd.get_dummies(applications[['NAME_CONTRACT_TYPE', 'CODE_GENDER',
       'FLAG_OWN_CAR', 'FLAG_OWN_REALTY','NAME_TYPE_SUITE',
       'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS',
       'NAME_HOUSING_TYPE','WEEKDAY_APPR_PROCESS_START','ORGANIZATION_TYPE']])


# In[38]:


app_cont = applications.drop(['NAME_CONTRACT_TYPE', 'CODE_GENDER',
       'FLAG_OWN_CAR', 'FLAG_OWN_REALTY','NAME_TYPE_SUITE',
       'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS',
       'NAME_HOUSING_TYPE','WEEKDAY_APPR_PROCESS_START','ORGANIZATION_TYPE'],axis = 1)


# In[39]:


app = pd.concat([app_cont,app_dum],axis = 1)


# In[40]:


c_tmp = app.drop(['SK_ID_CURR','TARGET'],axis=1).corr().unstack().reset_index()
c_tmp[c_tmp[0]!=1][c_tmp[0]>=0.7]


# In[41]:


delete = ['CNT_CHILDREN','AMT_ANNUITY','AMT_GOODS_PRICE','NAME_INCOME_TYPE_Pensioner',
          'ORGANIZATION_TYPE_XNA','REGION_RATING_CLIENT_W_CITY','LIVE_REGION_NOT_WORK_REGION',
          'LIVE_CITY_NOT_WORK_CITY','OBS_60_CNT_SOCIAL_CIRCLE','DEF_60_CNT_SOCIAL_CIRCLE']
app = app.drop(delete,axis = 1)


# In[42]:


total_dt = app.merge(pre_app,how = 'inner', on = 'SK_ID_CURR').merge(bureau,how = 'inner', on = 'SK_ID_CURR').merge(install,how = 'inner', on = 'SK_ID_CURR').merge(pos_balance,how = 'inner', on = 'SK_ID_CURR').merge(card,how = 'inner', on = 'SK_ID_CURR')


# In[43]:


total_dt


# In[46]:


total_dt.to_csv("HCDR.csv",index = False)

