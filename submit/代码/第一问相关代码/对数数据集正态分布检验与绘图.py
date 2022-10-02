import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import math
import scipy.stats as stats
import matplotlib.pyplot as plt
from geneview.gwas import qqplot
# plt.rcParams['font.sans-serif'] = ['FangSong']
data = pd.read_csv("card_transdata.csv")

data = data.values

for i in range(data.shape[0]):
    data[i, 0] = math.log(data[i, 0])
    data[i, 1] = math.log(data[i, 1])
    data[i, 2] = math.log(data[i, 2])

# data=pd.DataFrame(data[:,0])
def self_JBtest(y):
    # 样本规模n
    n = y.size
    y_ = y - y.mean()
    """
    M2:二阶中心钜
    skew 偏度 = 三阶中心矩 与 M2^1.5的比
    krut 峰值 = 四阶中心钜 与 M2^2 的比
    """
    M2 = np.mean(y_ ** 2)
    skew = np.mean(y_ ** 3) / M2 ** 1.5
    krut = np.mean(y_ ** 4) / M2 ** 2

    """
    计算JB统计量，以及建立假设检验
    """
    JB = n * (skew ** 2 / 6 + (krut - 3) ** 2 / 24)
    pvalue = 1 - stats.chi2.cdf(JB, df=2)
    print("偏度：", stats.skew(y), skew)
    print("峰值：", stats.kurtosis(y) + 3, krut)
    print("JB检验：", stats.jarque_bera(y))
    return np.array([JB, pvalue])


print(self_JBtest(data[:, 0]))
print(self_JBtest(data[:, 1]))
print(self_JBtest(data[:, 2]))
# ax = qqplot(data[:,0],ax=1)
# fig = plt.figure()
# res = stats.probplot(data[:,0], plot=plt)
# plt.show()
#
# fig = plt.figure()
# res = stats.probplot(data[:,1], plot=plt)
# plt.show()
#
# fig = plt.figure()
# res = stats.probplot(data[:,2], plot=plt)
# plt.show()


