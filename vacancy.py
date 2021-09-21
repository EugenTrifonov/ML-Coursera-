from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.feature_extraction import DictVectorizer
import pandas as pd
import numpy as np
from scipy.sparse import hstack
x_train = pd.read_csv('salary-train.csv')
x_test = pd.read_csv('salary-test.csv')
y_train = x_train['SalaryNormalized']
x_train.drop('SalaryNormalized', axis=1, inplace=True)
columns=x_train.columns
for i in columns:
        x_train[i].replace('[^a-zA-Z0-9]', ' ', regex=True)
for i in ['LocationNormalized', 'ContractTime']:
        x_train[i].fillna('nan',inplace=True)
for i in ['LocationNormalized', 'ContractTime']:
        x_train[i].fillna('NaN',inplace=True)
x_train['FullDescription'].str.lower()
#print(x_train['LocationNormalized'])
vectorizer=TfidfVectorizer(min_df=5)
x_train_vec=vectorizer.fit_transform(x_train['FullDescription'])
x_test_vec=vectorizer.transform(x_test['FullDescription'])
enc=DictVectorizer()
x_train_categ = enc.fit_transform(x_train[['LocationNormalized', 'ContractTime']].to_dict('records'))
x_test_categ = enc.transform(x_test[['LocationNormalized', 'ContractTime']].to_dict('records'))
x_for_train=hstack([x_train_vec,x_train_categ])
x_for_test=hstack([x_test_vec,x_test_categ])
ridge=Ridge(random_state=241,alpha=1)
ridge.fit(x_for_train,y_train)
prediction=ridge.predict(x_for_test)
print(prediction)

