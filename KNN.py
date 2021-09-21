import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import scale

data = np.genfromtxt('wine.data', delimiter=',')
x = data[:,1:]
y = data[:,0]
tmp = list()
for i in range(50):
    kf = KFold(n_splits=5, random_state=42, shuffle=True)
    quality = cross_val_score(KNeighborsClassifier(n_neighbors=i+1),\
                              x, y, cv=kf, scoring='accuracy')
    tmp.append(np.mean(quality))

index = tmp.index(max(tmp))+1

tmp_sc = list()
x = scale(x)
for i in range(50):
    kf_sc = KFold(n_splits=5, random_state=42, shuffle=True)
    quality_sc = cross_val_score(KNeighborsClassifier(n_neighbors=i+1),\
                              x, y, cv=kf, scoring='accuracy')
    tmp_sc.append(np.mean(quality_sc))

index_sc = tmp_sc.index(max(tmp_sc))+1
print(index_sc)
print(index)
print(quality)
print(quality_sc)
print(max(tmp))
print(tmp_sc)