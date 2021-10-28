import numpy as np
import pandas as pd
import scipy
import sklearn
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
X, y = make_regression(n_samples=100, n_features=2, n_informative=1)
X = StandardScaler().fit_transform(X)
y = StandardScaler().fit_transform(y.reshape(len(y),1))[:,0]
print(X.T)
y_pred1 = 35 * np.ones(100) + X.T[1]*5
y_pred2 = 40 * np.ones(100) + X.T[1]*7.5
print(y_pred1, y_pred2)
err1 = np.sum(y - y_pred1)
err2 = np.sum(y - y_pred2)
print(err1, err2)

mae_1 = np.sum(np.abs(y - y_pred1)) / 100
mae_2 = np.sum(np.abs(y - y_pred2)) / 100
print(mae_1, mae_2)
mse_1 = np.mean((y - y_pred1) ** 2)
mse_2 = np.mean((y - y_pred2) ** 2)
print(mse_1, mse_2)
print(X)
print(X.shape)
print(X.T)
print(X.T.shape)
print(all(X.T @ y == np.dot(X.T, y)))
W = np.linalg.inv(np.dot(X.T, X)) @X.T @ y
print(W)

y_pred3 = W[0] * X[0] + W[1] * X[1]
print(y_pred3)
def calc_mae(y, y_pred):
    err = np.mean(np.abs(y - y_pred))
    return err

def calc_mse(y, y_pred):
    err = np.mean((y- y_pred) ** 2) # <=> 1/n * np.sum((y_pred - y) ** 2)
    return err

print(calc_mae(y, y_pred1), calc_mse(y, y_pred1))
print(calc_mae(y, y_pred2), calc_mse(y, y_pred2))
plt.plot(X, y)
print(plt.show())
W = np.array([1, 0.5])
print(W)
gradient_form_direct = 1/100 * 2 * np.sum(X[0] * W[0] - y[0])
print(gradient_form_direct)
print(W[0] - gradient_form_direct)


