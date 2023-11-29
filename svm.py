import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.metrics import RocCurveDisplay, auc
from sklearn.model_selection import StratifiedKFold

random_state = np.random.RandomState(0)
classifier = svm.SVC(kernel="linear", probability=True, random_state=random_state)
classifier.fit(X, y)
