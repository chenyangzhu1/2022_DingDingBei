import time
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score

from sklearn.model_selection import GridSearchCV
x_train=pd.read_csv("x_SMOTE.csv",header=None)
x_train=x_train.values
y_train=pd.read_csv("y_SMOTE.csv",header=None)
y_train=y_train.values
y_train=y_train.flatten()
x_test=pd.read_csv("x_test.csv",header=None)
x_test=x_test.values
y_test=pd.read_csv("y_test.csv",header=None)
y_test=y_test.values
y_test=y_test.flatten()

param = {
    'C': [0.001, 0.01, 0.1, 1, 10]
}
"""
如果GridSearchCV初始化时，refit=True（默认的初始化值）,
在交叉验证时，一旦发现最好的模型（estimator）,将会在整个训练集上重新训练，
这通常是一个好主意，因为使用更多的数据集会提升模型的性能。
"""
model=GridSearchCV(LogisticRegression(max_iter=1000),param_grid=param,n_jobs=-1,scoring='f1',cv=5,refit=True)
model.fit(x_train,y_train)



y_test_predict=model.predict(x_test)
p_test=precision_score(y_test,y_test_predict)
r_test=recall_score(y_test,y_test_predict)
f1_test=f1_score(y_test,y_test_predict)
print(p_test,r_test,f1_test)
rocy=model.predict_proba(x_test)

fpr, tpr, thresholds = metrics.roc_curve(y_test, rocy[:,1], pos_label=1)

# print(fpr, '\n', tpr, '\n', thresholds)
print("auc-value")
print(metrics.auc(fpr, tpr))
res = np.zeros([10])
res[0] = p_test
res[1] = r_test
res[2] = f1_test
res[3] = metrics.auc(fpr, tpr)
list=[model.best_params_]
canshu=np.array(list,dtype=str)
np.savetxt("逻辑回归调参-results.csv", res, delimiter=',', header='precision,recall,f1-score,auc-value')
np.savetxt("逻辑回归调参-AUC-fpr.csv", fpr, delimiter=',')
np.savetxt("逻辑回归调参-AUC-tpr.csv", tpr, delimiter=',')
np.savetxt("逻辑回归-参数.csv",canshu,delimiter=',',fmt='%s')

