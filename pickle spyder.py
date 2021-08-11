# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 22:46:52 2021

@author: Dell
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
data=pd.read_csv(r'C:\Users\Dell\Desktop\cleaned.csv')
data1=pd.read_csv(r'C:\Users\Dell\Desktop\cleaned.csv')
data=data.dropna()
data1=data1.dropna()
df=data.copy()
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024**2
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                #if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                #    df[col] = df[col].astype(np.float16)
                #el
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        #else:
            #df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB --> {:.2f} MB (Decreased by {:.1f}%)'.format(
        start_mem, end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df
data=reduce_mem_usage(df)
label_encoder = preprocessing.LabelEncoder()
data=data.drop(['Unnamed: 0','DrugId','Date','Reviews'],axis=1)
x2=data['Drug'].unique()
x3=data['Condition'].unique()
x4=data['Age'].unique()
x5=data['Sex'].unique()
drug_enc = dict(enumerate(x2.flatten(), 1))
condition_enc = dict(enumerate(x3.flatten(), 1))
age_enc=dict(enumerate(x4.flatten(), 1))
sex_enc=dict(enumerate(x5.flatten(), 1))
def age_encode(x):
    for j in age_enc:
        if x==age_enc[j]:
            return j
def condition_encode(x):
    for k in condition_enc:
        if x==condition_enc[k]:
            return k
def drug_encode(x):
    for l in drug_enc:
        if x==drug_enc[l]:
            return l
def sex_encode(x):
    for m in sex_enc:
        if x==sex_enc[m]:
            return m
data['Age']=data['Age'].apply(lambda x:age_encode(x)).astype(str)
data['Condition']=data['Condition'].apply(lambda x:condition_encode(x)).astype(str)
data['Drug']=data['Drug'].apply(lambda x:drug_encode(x)).astype(str)
data['Sex']=data['Sex'].apply(lambda x:sex_encode(x)).astype(str)
data.drop('EaseofUse',
  axis='columns', inplace=True)
data.drop('Satisfaction',
  axis='columns', inplace=True)
data.drop('UsefulCount',
  axis='columns', inplace=True)
data.drop(['Sides','Effectiveness'],
  axis='columns', inplace=True)
X = data
Y = df['Sides']
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size = 0.3,random_state=1000)
DecisionTreePipeline=Pipeline([('decisiontree_classifier',DecisionTreeClassifier())])
KNeighborsPipeline=Pipeline([('KNeighbors_classifier',KNeighborsClassifier())])
mypipeline=[ DecisionTreePipeline,KNeighborsPipeline]
accuracy=0.0
classifier=0
pipeline=" "
PipeLineDict={0:'DecisionTree',1:'KNeighbors'}
#fit the pipeline
for mypipe in mypipeline:
    mypipe.fit(X_train,y_train)
for i,model in enumerate(mypipeline):
    print("{} TestAccuracy:{}".format(PipeLineDict[i],model.score(X_test,y_test)))
for i,model in enumerate(mypipeline):
    if model.score(X_test,y_test)>accuracy:
        accuracy=model.score(X_test,y_test)
        pipeline=model
        classifier=i
print('classifier with the best accuracy:{}'.format(PipeLineDict[classifier]))
import pickle
Pkl_Filename = "pipe_model.pkl"  
with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(mypipe, file)
with open(Pkl_Filename, 'rb') as file:  
    pipe_model = pickle.load(file)
print(pipe_model)