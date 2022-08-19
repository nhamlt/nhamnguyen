# -*- coding: utf-8 -*-
"""
Created on Wed May 18 07:45:38 2022

@author: admin
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import (TimeSeriesSplit, KFold, ShuffleSplit,StratifiedKFold,GroupShuffleSplit, GroupKFold, StratifiedShuffleSplit)



>>> X, y = datasets.load_iris(return_X_y=True)
>>> X.shape, y.shape
>>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
>>> X_train.shape, y_train.shape
>>> X_test.shape, y_test.shape
>>> clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)

clf.score(X_test, y_test)
cv = KFold(n_splits=4,X,y,random_state=None, shuffle=True)
cv.split(X)
X_train = X[train_index]

>>> for train_index, test_index in cv.split(X):
...     print("TRAIN:", train_index, "TEST:", test_index)
...     X_train, X_test = X[train_index], X[test_index]
...     y_train, y_test = y[train_index], y[test_index]

clf = svm.SVC(kernel='linear', C=1, random_state=42)
scores = cross_val_score(clf, X, y, cv=5)

a= 'nhamco tinh viet them vao day haha'
