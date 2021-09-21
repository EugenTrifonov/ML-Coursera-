import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
data = pd.read_csv('classification.csv')
data2=pd.read_csv('scores.csv')
#print(data2)
TP=0
FN=0
FP=0
TN=0
#print(data)
for i in range(data.shape[0]):
    if data.iloc[i][0]==1:
        if data.iloc[i][1]==1:
            TP+=1
        if data.iloc[i][1]==0:
            FN+=1
    elif data.iloc[i][0]==0:
        if data.iloc[i][1] == 1:
            FP += 1
        if data.iloc[i][1] == 0:
            TN += 1
print('TP=',TP,'FN=',FN,'FP=',FP,'TN=',TN)
print(accuracy_score(data['true'],data['pred']))
print(precision_score(data['true'],data['pred']))
print(recall_score(data['true'],data['pred']))
print(f1_score(data['true'],data['pred']))
score=roc_auc_score(data['true'],data['pred'])
y_data=data2['true']
score_logreg=data2['score_logreg']
score_svm=data2['score_svm']
score_knn=data2['score_knn']
score_tree=data2['score_tree']
abc=[score_logreg,score_svm,score_knn,score_tree]
mass_auc=[]
scores=[]
for i in abc:
    mass_auc.append(roc_auc_score(y_data,i))
print(mass_auc)

data2['score_logreg']=data2['score_logreg'].apply(lambda i : 1 if i>=0.7 else 0)
data2['score_svm']=data2['score_svm'].apply(lambda i : 1 if i>=0.7 else 0)
data2['score_knn']=data2['score_knn'].apply(lambda i : 1 if i>=0.7 else 0)
data2['score_tree']=data2['score_tree'].apply(lambda i : 1 if i>=0.7 else 0)
score_logreg=data2['score_logreg']
score_svm=data2['score_svm']
score_knn=data2['score_knn']
score_tree=data2['score_tree']
abc=[score_logreg,score_svm,score_knn,score_tree]
for i in abc:
    scores.append(precision_score(data2['true'],i))
print(scores)