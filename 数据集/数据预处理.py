import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
data=pd.read_csv("card_transdata.csv")
#1000000条数据
print(data['ratio_to_median_purchase_price'].describe())


# print(data['distance_from_home'].nunique())

# sns.set()
# sns.histplot(data['distance_from_home'],kde=True)
# plt.show()