import numpy as np
import scipy.stats as stats
import pandas as pd


def self_JBtest(y):
    # 样本规模n
    n = y.size
    y_ = y - y.mean()
    """
    M2:二阶中心钜
    skew 偏度 = 三阶中心矩 与 M2^1.5的比
    krut 峰值 = 四阶中心钜 与 M2^2 的比
    """
    M2 = np.mean(y_**2)
    skew =  np.mean(y_**3)/M2**1.5
    krut = np.mean(y_**4)/M2**2

    """
    计算JB统计量，以及建立假设检验
    """
    JB = n*(skew**2/6 + (krut-3 )**2/24)
    pvalue = 1 - stats.chi2.cdf(JB,df=2)
    print("偏度：",stats.skew(y),skew)
    print("峰值：",stats.kurtosis(y)+3,krut)
    print("JB检验：",stats.jarque_bera(y))
    return np.array([JB,pvalue])

data=pd.read_csv("card_transdata.csv")
data=data.values
print(self_JBtest(data[:,0]))