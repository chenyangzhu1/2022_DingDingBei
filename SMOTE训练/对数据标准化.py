import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import seaborn as sb


data = pd.read_csv("正态处理后的数据集.csv",header=None)


data = data.values

for i in range(3):
    mean=np.mean(data[:,i])
    std=np.std(data[:,i])
    for j in range(data.shape[0]):
        data[j,i]=(data[j,i]-mean)/std

np.savetxt("标准化后数据集.csv",data,delimiter=',')