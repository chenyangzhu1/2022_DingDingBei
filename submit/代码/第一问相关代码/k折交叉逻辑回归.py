# from sklearn.model_selection import KFold
#
# kf = KFold(n_splits = 5, shuffle=True, random_state=0)
# for train_index, test_index in kf.split(data):
#     clt = model.fit(data[train_index], four_label[train_index])
#     curr_score = curr_score + clt.score(data[test_index], four_label[test_index])
#     print(clt.score(data[test_index], four_label[test_index]))
#
# avg_score = curr_score / 5
# print("平均准确率为：", avg_score)
#
#
#
#
# from sklearn.model_selection import cross_val_score
#
# clf = svm.SVC(kernel='linear', C=1)
#
# scores = cross_val_score(clf, iris.data, iris.target, cv=5)
#
# print("scores:",scores)
#
# print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
#
# print("\n")

import time
from sklearn import metrics

from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
data=pd.read_csv("card_transdata.csv")
data=data.values

x=data[:,0:7]
y=data[:,7]
model=linear_model.LogisticRegression(max_iter=1000)