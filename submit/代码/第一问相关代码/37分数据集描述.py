import time
from sklearn import metrics

from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
data=pd.read_csv("card_transdata.csv")
data=data.values

x=data[:,0:7]
y=data[:,7]
# print(x,y)
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=1,train_size=0.7)

np.savetxt("三七分训练集标签.csv",y_train,delimiter=',')
np.savetxt("三七分测试集标签.csv",y_test,delimiter=',')

train=np.zeros([700000,8])
test=np.zeros([300000,8])

for i in range(700000):
    train[i,0:7]=x_train[i,:]
    train[i,7]=y_train[i]

for i in range(300000):
    test[i,0:7]=x_test[i,:]
    test[i,7]=y_test[i]


sb.set()
ax1=sb.histplot(train[:,0],kde=True)
# plt.title("distance_from_home",fontsize=20)
ax2=sb.histplot(test[:,0],kde=True,color='red')
plt.show()
bx1=sb.histplot(train[:,1],kde=True)
# plt.title("distance_from_last_transaction",fontsize=20)
bx2=sb.histplot(test[:,1],kde=True,color='red')
plt.show()
cx1=sb.histplot(train[:,2],kde=True)
# plt.title("ratio_to_median_purchase_price",fontsize=20)
cx2=sb.histplot(test[:,2],kde=True,color='red')
plt.show()





