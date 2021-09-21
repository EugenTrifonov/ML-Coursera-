import  pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss
import numpy as np
data_pd=pd.read_csv('gbm-data.csv')
y_1=data_pd['Activity']
x_1=data_pd.iloc[:,1:]
X=x_1.to_numpy()
y=y_1.to_numpy()
X_train, X_test, y_train, y_test=train_test_split(X, y,test_size=0.8,random_state=241)
learning_rate=[1,0.5,0.3,0.2,0.1]
for i in learning_rate:
    print("Hello")
    Grad = GradientBoostingClassifier(n_estimators=250, verbose=True, random_state=241,learning_rate=i)
    Grad.fit(X_train,y_train)
    test_loss = np.empty(250)
    y_pred=Grad.staged_decision_function(X_test)
    for m, y_decision in enumerate(Grad.staged_decision_function(X_test)):
            y_pred_test = 1.0 / (1.0 + np.exp(-y_decision))
            test_loss[m] = log_loss(y_test, y_pred_test)
    print(test_loss.max())
    if i == 0.2:
        print(min(range(len(test_loss)), key=test_loss.__getitem__))
    print(test_loss.min())
    plt.figure()
    plt.plot(test_loss, 'r', linewidth=2)
    plt.show()
F=RandomForestClassifier(n_estimators=36,random_state=241)
F.fit(X_train,y_train)
y_pred=F.predict_proba(X_test)
test_loss_1=log_loss(y_test,y_pred)
print(test_loss_1)
