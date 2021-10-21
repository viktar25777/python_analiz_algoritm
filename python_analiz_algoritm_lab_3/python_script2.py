import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, roc_auc_score, precision_score, f1_score
X, y = make_classification(n_classes=2, class_sep=1.5, weights=[0.9, 0.1], n_features=20, n_samples=1000, random_state=10)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
regressor = LogisticRegression(class_weight="balanced")
print(regressor.fit(X_train, y_train))
W = 0.25
print(W)
y_preds = np.where(regressor.predict_proba(X_test)[:,1] > W, 1, 0)
y_true = preds = np.where(regressor.predict_proba(X_test)[:,1] > W, 1, 0)
print(accuracy_score(y_true, y_preds))
print(accuracy_score(y_true, y_preds, normalize=False))
print(confusion_matrix(y_true, y_preds))
print(confusion_matrix(y_true, y_preds, normalize='all'))
tn, fp, fn, tp = confusion_matrix(y_true, y_preds).ravel()
print(tn, fp, fn, tp)
print(precision_score(y_true, y_preds))
print(recall_score(y_true, y_preds))
print(f1_score(y_true, y_preds))



