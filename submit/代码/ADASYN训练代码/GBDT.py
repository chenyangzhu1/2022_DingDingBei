from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
import time
# data=pd.read_csv("正态处理后的数据集.csv",header=None)
#
# data = data.values
#
# x = data[:, 0:7]
# y = data[:, 7]
# # print(x,y)
# x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, train_size=0.7)

x_train = pd.read_csv("x_ADASYN.csv", header=None)
x_train = x_train.values
y_train = pd.read_csv("y_ADASYN.csv", header=None)
y_train = y_train.values
y_train = y_train.flatten()
x_test = pd.read_csv("x_test.csv", header=None)
x_test = x_test.values
y_test = pd.read_csv("y_test.csv", header=None)
y_test = y_test.values
y_test = y_test.flatten()

start = time.perf_counter()
model = GradientBoostingClassifier(random_state=1)

# my_train=np.zeros([y_train.shape[0],2])
# for i in range(y_train.shape[0]):
#     my_train[i,int(y_train[i])]=1
#
# my_test=np.zeros([y_test.shape[0],2])
# for i in range(y_test.shape[0]):
#     my_train[i,int(y_test[i])]=1
#


'''
调参：
loss：损失函数。有deviance和exponential两种。deviance是采用对数似然，exponential是指数损失，后者相当于AdaBoost。
n_estimators:最大弱学习器个数，默认是100，调参时要注意过拟合或欠拟合，一般和learning_rate一起考虑。
learning_rate:步长，即每个弱学习器的权重缩减系数，默认为0.1，取值范围0-1，当取值为1时，相当于权重不缩减。较小的learning_rate相当于更多的迭代次数。
subsample:子采样，默认为1，取值范围(0,1]，当取值为1时，相当于没有采样。小于1时，即进行采样，按比例采样得到的样本去构建弱学习器。这样做可以防止过拟合，但是值不能太低，会造成高方差。
init：初始化弱学习器。不使用的话就是第一轮迭代构建的弱学习器.如果没有先验的话就可以不用管

由于GBDT使用CART回归决策树。以下参数用于调优弱学习器，主要都是为了防止过拟合
max_feature：树分裂时考虑的最大特征数，默认为None，也就是考虑所有特征。可以取值有：log2,auto,sqrt
max_depth：CART最大深度，默认为None
min_sample_split：划分节点时需要保留的样本数。当某节点的样本数小于某个值时，就当做叶子节点，不允许再分裂。默认是2
min_sample_leaf：叶子节点最少样本数。如果某个叶子节点数量少于某个值，会同它的兄弟节点一起被剪枝。默认是1
min_weight_fraction_leaf：叶子节点最小的样本权重和。如果小于某个值，会同它的兄弟节点一起被剪枝。一般用于权重变化的样本。默认是0
min_leaf_nodes：最大叶子节点数
'''

model.fit(x_train, y_train)
end = time.perf_counter()
needtime = end - start
# score_test = model.score(x_test, y_test)
# score_train = model.score(x_train, y_train)
# score_total = model.score(x, y)
# print(score_test)
# print(score_train)
# print(score_total)

# y_pre_train = model.predict(x_train)
# p_train = precision_score(y_train, y_pre_train)


y_test_predict = model.predict(x_test)
# y_test_predict = np.argmax(y_test_predict,axis=0)

ysum = np.sum(y_test_predict)
# print(ysum)
# for i in range(y_test_predict.shape[0]):
#     if(y_test_predict[i]<0.3):
#         y_test_predict[i]=0
#     else:
#         y_test_predict[i]=1

ysum = np.sum(y_test_predict)

print(ysum)
p_test = precision_score(y_test, y_test_predict)
r_test = recall_score(y_test, y_test_predict)
f1_test = f1_score(y_test, y_test_predict)
print(p_test, r_test, f1_test)

# fpr, tpr, thresholds = metrics.roc_curve(y_test, y_test_predict, pos_label=1)
# print(fpr, '\n', tpr, '\n', thresholds)
# print(metrics.auc(fpr, tpr))
#
# y_train_predict = model.predict(x_train)
# p_train = precision_score(y_train, y_train_predict)
# r_train = recall_score(y_train, y_train_predict)
# f1_train = f1_score(y_train, y_train_predict)
# print(p_train, r_train, f1_train)

rocy=model.predict_proba(x_test)
# print(rocy)
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

np.savetxt("GBDT-results.csv", res, delimiter=',', header='precision,recall,f1-score,auc-value,time')
np.savetxt("GBDT-AUC-fpr.csv", fpr, delimiter=',')
np.savetxt("GBDT-AUC-tpr.csv", tpr, delimiter=',')
print(metrics.classification_report(y_test,y_test_predict))
