import numpy as np
from sklearn.metrics import roc_auc_score
import pandas as pd
data = pd.read_csv('data_logistic.csv', header=None)
#data=np.genfromtxt('data_logistic.csv',delimiter=',')
X = data.iloc[:,1:]
y = data.iloc[:,0]
w1 = 0
w2 = 0
k = 0.1
max_iter = 10000
evk_min = 0.00001
print(X)
print(y)


def logistic_reqression(X, y, w1=0, w2=0, c=0, k=0.1, max_iter=100, evk_min=0.00001):
    l = len(y)
    summa1 = 0
    summa2 = 0
    for j in range(max_iter):
        for i in range(l):
            s = 1 - 1 / (1 + np.exp(-y[i] * (w1 * X[1][i] + w2 * X[2][i])))
            summa1 += y[i] * X[1][i] * s
            summa2 += y[i] * X[2][i] * s
        w1new = w1 + k / l * summa1 - k * c * w1
        w2new = w2 + k / l * summa2 - k * c * w2
        evk = np.sqrt((w1new - w1) ** 2 + (w2new - w2) ** 2)
        if (evk < evk_min): break
        w1, w2 = w1new, w2new
    return w1, w2


def auc_roc(X, y, w1=0, w2=0):
    l = len(y)
    a = []
    for i in range(l):
        a.append(1 / (1 + np.exp(- w1 * X[1][i] - w2 * X[2][i])))
    return roc_auc_score(y, a)


ww1,ww2 = logistic_reqression(X, y, c=0)
print('w1=%.8f, w2=%.8f' % (ww1, ww2))

print(roc_auc_score(y, [1 / (1 + np.exp(-ww1 * X[1][i] - ww2 * X[2][i])) for i in range(len(y))]))

ww1, ww2 = logistic_reqression(X, y, c=10)
print('w1=%.8f, w2=%.8f' % (ww1, ww2))

print(roc_auc_score(y, [1 / (1 + np.exp(-ww1 * X[1][i] - ww2 * X[2][i])) for i in range(len(y))]))
