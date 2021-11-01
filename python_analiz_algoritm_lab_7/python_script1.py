import numpy as np
import pandas as pd
import scipy
import sklearn
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from itertools import product
iris = load_iris()
X, y = iris.data[:, :2], iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
knn = KNeighborsClassifier(n_neighbors=1, weights="uniform")
print(knn.fit(X_train, y_train))
y_hat = knn.predict(X_test)
print(classification_report(y_test, y_hat))
def plot_decision_boundary(model, X, y):
    color = ["r", "g", "b"]
    marker = ["o", "v", "x"]
    class_label = np.unique(y)
    cmap = ListedColormap(color[: len(class_label)])
    x1_min, x2_min = np.min(X, axis=0)
    x1_max, x2_max = np.max(X, axis=0)
    x1 = np.arange(x1_min - 1, x1_max + 1, 0.02)
    x2 = np.arange(x2_min - 1, x2_max + 1, 0.02)
    X1, X2 = np.meshgrid(x1, x2)
    Z = model.predict(np.c_[X1.ravel(), X2.ravel()])
    Z = Z.reshape(X1.shape)
    plt.contourf(X1, X2, Z, cmap=cmap, alpha=0.5)
    for i, class_ in enumerate(class_label):
        plt.scatter(x=X[y == class_, 0], y=X[y == class_, 1],
                c=cmap.colors[i], label=class_, marker=marker[i])
    plt.legend()

print(plt.figure(figsize=(18, 10)))
weights = ['uniform', 'distance']
ks = [2, 15]
for i, (w, k) in enumerate(product(weights, ks), start=1):
    plt.subplot(2, 2, i)
    plt.title(f"Значение K: {k} вес: {w}")
    knn = KNeighborsClassifier(n_neighbors=k, weights=w)
    knn.fit(X, y)
    plot_decision_boundary(knn, X_train, y_train)

print(plt.show())



