from fitter import Fitter
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
data=pd.read_csv("card_transdata.csv",header=0)

data=data.values

f=Fitter(data[:,0],timeout=100)
f.fit()
f.hist()
plt.show()
print(f.summary())
print(f.get_best())

