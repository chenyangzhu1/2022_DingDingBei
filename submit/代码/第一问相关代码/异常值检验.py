import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import math
plt.rcParams['font.sans-serif']=['FangSong']
data=pd.read_csv("card_transdata.csv")


data=data.values

for i in range(data.shape[0]):
    data[i,0]=math.log(data[i,0])
    data[i,1]=math.log(data[i,1])
    data[i,2]=math.log(data[i,2])
# data=pd.DataFrame(data)
sb.set()
ax=sb.histplot(data[:,0],kde=True)
# plt.title("distance_from_home",fontsize=20)
plt.show()
bx=sb.histplot(data[:,1],kde=True)
# plt.title("distance_from_last_transaction",fontsize=20)
plt.show()
cx=sb.histplot(data[:,2],kde=True)
# plt.title("ratio_to_median_purchase_price",fontsize=20)
plt.show()