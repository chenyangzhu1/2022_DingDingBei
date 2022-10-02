import time
from sklearn import metrics

from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
data=pd.read_csv("标准化后数据集.csv",header=None)

data=data.values

x=data[:,0:7]
y=data[:,7]
# print(x,y)
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=1,train_size=0.7)


start = time.perf_counter()
model=linear_model.LogisticRegression(max_iter=1000)
model.fit(x_train,y_train)
end = time.perf_counter()
needtime = end - start

y_test_predict=model.predict(x_test)
score_test=model.score(x_test,y_test)
score_train=model.score(x_train,y_train)
score_total=model.score(x,y)
# print(score_test)
# print(score_train)
# print(score_total)

y_pre_train=model.predict(x_train)
p_train=precision_score(y_train,y_pre_train)



p_test=precision_score(y_test,y_test_predict)
r_test=recall_score(y_test,y_test_predict)
f1_test=f1_score(y_test,y_test_predict)
print(p_test,r_test,f1_test)

rocy=model.predict_proba(x_test)

fpr, tpr, thresholds = metrics.roc_curve(y_test, rocy[:,1], pos_label=1)

# print(fpr, '\n', tpr, '\n', thresholds)
print("auc-value")
print(metrics.auc(fpr, tpr))
print(needtime)
res = np.zeros([5])
res[0] = p_test
res[1] = r_test
res[2] = f1_test
res[3] = metrics.auc(fpr, tpr)
res[4] = needtime

np.savetxt("逻辑回归-results.csv", res, delimiter=',', header='precision,recall,f1-score,auc-value,time')
np.savetxt("逻辑回归-AUC-fpr.csv", fpr, delimiter=',')
np.savetxt("逻辑回归-AUC-tpr.csv", tpr, delimiter=',')
