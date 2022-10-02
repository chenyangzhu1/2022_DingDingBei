import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import seaborn as sb

data = pd.read_csv("card_transdata.csv")

data = data.values
for i in range(data.shape[0]):
    data[i, 0] = math.log(data[i, 0])
    data[i, 1] = math.log(data[i, 1])
    data[i, 2] = math.log(data[i, 2])

mean1 = np.mean(data[:, 0])
std1 = np.std(data[:, 0])

upper1 = mean1 + 3 * std1
lower1 = mean1 - 3 * std1
rmlist = []
for i in range(data.shape[0]):
    if (data[i, 0] > upper1 or data[i, 0] < lower1):
        if i in rmlist:
            continue
        else:
            rmlist.append(i)

mean1 = np.mean(data[:, 1])
std1 = np.std(data[:, 1])

upper1 = mean1 + 3 * std1
lower1 = mean1 - 3 * std1
for i in range(data.shape[0]):
    if (data[i, 1] > upper1 or data[i, 1] < lower1):
        if i in rmlist:
            continue
        else:
            rmlist.append(i)


mean1 = np.mean(data[:, 2])
std1 = np.std(data[:, 2])
upper1 = mean1 + 3 * std1
lower1 = mean1 - 3 * std1
for i in range(data.shape[0]):
    if (data[i, 2] > upper1 or data[i, 2] < lower1):
        if i in rmlist:
            continue
        else:
            rmlist.append(i)

data=np.delete(data,rmlist,axis=0)


for i in range(data.shape[0]):
    data[i, 0] = np.exp(data[i, 0])
    data[i, 1] = np.exp(data[i, 1])
    data[i, 2] = np.exp(data[i, 2])
np.savetxt("正态处理并还原后的数据集.csv",data,delimiter=',')
sb.set()
ax1=sb.histplot(data[:,0],kde=True)
plt.show()

