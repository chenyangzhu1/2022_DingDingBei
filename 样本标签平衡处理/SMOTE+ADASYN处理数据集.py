from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np
from imblearn.ensemble import EasyEnsembleClassifier
from imblearn.over_sampling import ADASYN
from sklearn.model_selection import train_test_split

data = pd.read_csv("正态处理后的数据集.csv", header=None)
data = data.values
x = data[:, 0:7]
y = data[:, 7]
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, train_size=0.7)

smote = SMOTE(random_state=1, n_jobs=-1, sampling_strategy='auto')
ada = ADASYN(random_state=1, n_jobs=-1)
x_new, y_new = smote.fit_resample(x_train, y_train)
sumy = np.sum(y_new)
np.savetxt("x_SMOTE.csv", x_new, delimiter=',')
np.savetxt("y_SMOTE.csv", y_new, delimiter=',')
np.savetxt("x_test.csv", x_test, delimiter=',')
np.savetxt("y_test.csv", y_test, delimiter=',')
