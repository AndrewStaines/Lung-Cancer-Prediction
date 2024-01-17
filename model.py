import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r'cancer patient data sets.csv')
df

df.info()

df.isnull().sum()

df.drop(['Patient Id'],axis=1,inplace=True)
df

df.Level.unique()

df['Level']=df['Level'].map({'Low':1, 'Medium':2, 'High':3}).astype('int64')

df.info()

x=df.iloc[:,1:-1]
x

y = df['Level']

from sklearn.model_selection import train_test_split
x_tr,x_te,y_tr,y_te = train_test_split(x,y,test_size=0.2)

from catboost import CatBoostClassifier

model=CatBoostClassifier(learning_rate=0.01)
model.fit(x_tr,y_tr)

model.save_model('Cancer_model',format='cbm')