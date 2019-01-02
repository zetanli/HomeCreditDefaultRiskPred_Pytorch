import numpy as np
import pandas as pd 
import random

app_train='application_train.csv'
app_test='application_test.csv'
df_apTrain=pd.read_csv(app_train,chunksize=20000,low_memory=False)
df_apTest=pd.read_csv(app_test,chunksize=20000,low_memory=False)

##manipulate train data in sub-chunks
sumNa=[]
totalRows=0
times=0
onehot_train=pd.DataFrame()
other_train=pd.DataFrame()
while True:
    
    try:
        df_train=df_apTrain.get_chunk()
    except StopIteration:
        break
    ## get columns that need to be one hot encoded     
    onehot_train = onehot_train.append(df_train[['NAME_TYPE_SUITE','NAME_INCOME_TYPE','NAME_EDUCATION_TYPE',
                         'NAME_FAMILY_STATUS','NAME_HOUSING_TYPE','OCCUPATION_TYPE','ORGANIZATION_TYPE']])
    ## get other columns
    other_train = other_train.append(df_train[df_train.columns.difference(['NAME_TYPE_SUITE','NAME_INCOME_TYPE','NAME_EDUCATION_TYPE',
                         'NAME_FAMILY_STATUS','NAME_HOUSING_TYPE','OCCUPATION_TYPE','ORGANIZATION_TYPE'])])
    ##compute total na 
    sumNa.append(df_train.isnull().sum())
    totalRows=totalRows+df_train.shape[0]
    
naPerc=sumNa[0]
for i in range(1,len(sumNa)):
    naPerc+=sumNa[i]
naPerc=naPerc/totalRows
naPerc=naPerc[naPerc!=0]
naPerSorted=naPerc.sort_values(ascending=False)
dropCols=list(naPerSorted[naPerSorted>0.3].index)
onehotCols=list(onehot_train.columns)
otherCols=list(other_train.columns)
oneDrop=[x for x in onehotCols if x in dropCols]
otherDrop=[x for x in otherCols if x in dropCols]

onehot_train=onehot_train.drop(oneDrop[0],axis=1)
other_train=other_train.drop(otherDrop,axis=1)

##manipulate test data in sub-chunks

onehot_test=pd.DataFrame()
other_test=pd.DataFrame()
while True:
    
    try:
        df_test=df_apTest.get_chunk()
    except StopIteration:
        break
    ## get columns that need to be one hot encoded     
    onehot_test = onehot_test.append(df_test[['NAME_TYPE_SUITE','NAME_INCOME_TYPE','NAME_EDUCATION_TYPE',
                         'NAME_FAMILY_STATUS','NAME_HOUSING_TYPE','OCCUPATION_TYPE','ORGANIZATION_TYPE']])
    ## get other columns
    other_test = other_test.append(df_test[df_test.columns.difference(['NAME_TYPE_SUITE','NAME_INCOME_TYPE','NAME_EDUCATION_TYPE',
                         'NAME_FAMILY_STATUS','NAME_HOUSING_TYPE','OCCUPATION_TYPE','ORGANIZATION_TYPE'])])
    
    
onehot_test=onehot_test.drop(oneDrop,axis=1)
other_test=other_test.drop(otherDrop,axis=1)

##compute pearson correlation

contColIndex=['AMT_ANNUITY', 'AMT_CREDIT', 'AMT_GOODS_PRICE', 'AMT_INCOME_TOTAL',
       'AMT_REQ_CREDIT_BUREAU_DAY', 'AMT_REQ_CREDIT_BUREAU_HOUR',
       'AMT_REQ_CREDIT_BUREAU_MON', 'AMT_REQ_CREDIT_BUREAU_QRT',
       'AMT_REQ_CREDIT_BUREAU_WEEK', 'AMT_REQ_CREDIT_BUREAU_YEAR',
       'CNT_CHILDREN', 'CNT_FAM_MEMBERS', 'DAYS_BIRTH',
       'DAYS_EMPLOYED', 'DAYS_ID_PUBLISH', 'DAYS_LAST_PHONE_CHANGE',
       'DAYS_REGISTRATION', 'DEF_30_CNT_SOCIAL_CIRCLE',
       'DEF_60_CNT_SOCIAL_CIRCLE', 'EXT_SOURCE_2', 'EXT_SOURCE_3',
       'HOUR_APPR_PROCESS_START','OBS_30_CNT_SOCIAL_CIRCLE',
       'OBS_60_CNT_SOCIAL_CIRCLE', 'REGION_POPULATION_RELATIVE',
       'REGION_RATING_CLIENT', 'REGION_RATING_CLIENT_W_CITY'
        ]
contCols=other_train[contColIndex]
colCorrs=contCols.corr()
colinears=dict()
for i in contColIndex:
    if list(colCorrs[i][colCorrs[i] > 0.8][colCorrs[i] < 1].index)!=[]:
        colinears[i] = list(colCorrs[i][colCorrs[i] > 0.8][colCorrs[i] < 1].index)



##Thus column' s pairs have high correlations are AMT_ANNUITY AMT_CREDIT AMT_GOODS_PRICE, CNT_CHILDREN CNT_FAM_MEMBERS, DEF_60_CNT_SOCIAL_CIRCLE DEF_30_CNT_SOCIAL_CIRCLE, 
##OBS_60_CNT_SOCIAL_CIRCLE OBS_30_CNT_SOCIAL_CIRCLE, REGION_RATING_CLIENT_W_CITY REGION_RATING_CLIENT.
##Columns we could further drop are AMT GOODS PRICE, CNT_CHILDREN, DEF_30_CNT_SOCIAL_CIRCLE, OBS_30_CNT_SOCIAL_CIRCLE, REGION_RATING_CLIENT	

	
other_train=other_train.drop(['AMT_GOODS_PRICE', 'CNT_CHILDREN', 'DEF_30_CNT_SOCIAL_CIRCLE', 'OBS_30_CNT_SOCIAL_CIRCLE', 'REGION_RATING_CLIENT'],axis=1)
other_test=other_test.drop(['AMT_GOODS_PRICE', 'CNT_CHILDREN', 'DEF_30_CNT_SOCIAL_CIRCLE', 'OBS_30_CNT_SOCIAL_CIRCLE', 'REGION_RATING_CLIENT'],axis=1)

miscols=[i for i in other_train.isnull().sum().index if other_train.isnull().sum()[i] >0]
ot_nomiss=other_train.drop(miscols,axis=1)
ot_imputed=other_train[miscols].fillna(other_train[miscols].mean())

miscols_test=[i for i in other_test.isnull().sum().index if other_test.isnull().sum()[i] >0]
ot_nomiss_test=other_test.drop(miscols_test,axis=1)
ot_imputed_test=other_test[miscols_test].fillna(other_test[miscols_test].mean())

otherFullImputed_df=pd.concat([ot_nomiss,ot_imputed],axis=1)
apTrainFull_df=pd.concat([otherFullImputed_df,pd.get_dummies(onehot_train)],axis=1)
typess=apTrainFull_df.dtypes
objCols=list(typess[typess=='object'].index)
apTrainFull=pd.concat([apTrainFull_df.drop(objCols,axis=1),pd.get_dummies(apTrainFull_df[objCols])],axis=1)

otherFullImputed_df_test=pd.concat([ot_nomiss_test,ot_imputed_test],axis=1)
apTestFull_df=pd.concat([otherFullImputed_df_test,pd.get_dummies(onehot_test)],axis=1)
typess=apTestFull_df.dtypes
objCols=list(typess[typess=='object'].index)
apTestFull=pd.concat([apTestFull_df.drop(objCols,axis=1),pd.get_dummies(apTestFull_df[objCols])],axis=1)

##split origin train dataset into train and test
##75% of origin train dataset as train, 25% as test 
trainIdx=random.sample(range(apTrainFull.shape[0]), int(apTrainFull.shape[0]*0.75))
testIdx=list(set(range(apTrainFull.shape[0]))-set(trainIdx))
trainAp=apTrainFull.iloc[trainIdx,:]
testAp=apTrainFull.iloc[testIdx,:]

##oversampling

def_ins=trainAp[trainAp['TARGET']==1]
nondef=trainAp[trainAp['TARGET']==0]
defNum=def_ins.shape[0]
##train:test=1:1

random.seed(222)
nondef_idx=random.sample(list(nondef.index),defNum)
nondef2=nondef.loc[nondef_idx,:]

trainAp11=nondef2.append(def_ins)

##train:test=1:1.5

random.seed(222)
nondef_idx=random.sample(list(nondef.index),int(defNum*1.5))
nondef2=nondef.loc[nondef_idx,:]

trainAp115=nondef2.append(def_ins)

##train:test=1:2

random.seed(222)
nondef_idx=random.sample(list(nondef.index),defNum*2)
nondef2=nondef.loc[nondef_idx,:]

trainAp12=nondef2.append(def_ins)

##train:test=1:2.5

random.seed(222)
nondef_idx=random.choices(list(nondef.index),k=int(defNum*2.5))
nondef2=nondef.loc[nondef_idx,:]

trainAp125=nondef2.append(def_ins)



##train:test=1:3

random.seed(222)
nondef_idx=random.choices(list(nondef.index),k=defNum*3)
nondef2=nondef.loc[nondef_idx,:]

trainAp13=nondef2.append(def_ins)

##train:test=1:4

random.seed(222)
nondef_idx=random.choices(list(nondef.index),k=defNum*4)
nondef2=nondef.loc[nondef_idx,:]

trainAp14=nondef2.append(def_ins)

##train:test=1:5

random.seed(222)
nondef_idx=random.choices(list(nondef.index),k=defNum*5)
nondef2=nondef.loc[nondef_idx,:]

trainAp15=nondef2.append(def_ins)

##train:test=1:6

random.seed(222)
nondef_idx=random.choices(list(nondef.index),k=defNum*6)
nondef2=nondef.loc[nondef_idx,:]

trainAp16=nondef2.append(def_ins)