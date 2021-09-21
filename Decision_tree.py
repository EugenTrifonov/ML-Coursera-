import pandas
import numpy as np
from sklearn.tree import DecisionTreeClassifier
data2=pandas.read_csv('titanic.csv',index_col='PassengerId')
columns=['Pclass','Fare','Age','Sex','Survived']
data=data2[columns]
data['Sex'].replace(['male', 'female'], [1, 0], inplace=True)
#print(data)
indexs=[]
for i in columns:
    index=0
    for j in data[i]:
        if np.isnan(j):
            indexs.append(index)
        index+=1

for i in range(len(indexs)):
    indexs[i]+=1
data.drop(indexs,axis=0,inplace=True)
y=data['Survived']
data.drop('Survived',axis=1,inplace=True)
print(data)
#print(y)
clf=DecisionTreeClassifier(random_state=241)
clf.fit(data,y)
importances=clf.feature_importances_
print(importances)