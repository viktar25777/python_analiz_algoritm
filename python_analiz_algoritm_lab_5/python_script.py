import numpy as np
import pandas as pd
import scipy
import sklearn
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
classification_data, classification_labels = make_classification(n_samples=100, n_features=1, n_informative=1, n_classes=2, n_redundant=0, n_clusters_per_class=1)
X_train, X_test, y_train, y_test = train_test_split(classification_data, classification_labels, test_size=0.5)
clf = DecisionTreeClassifier(random_state=0)
rfc = RandomForestClassifier(random_state=0)
clf = clf.fit(X_train, y_train)
rfc = rfc.fit(X_train, y_train)
score_c = clf.score(X_test, y_test)
score_r = rfc.score(X_test, y_test)
print("Single Tree:{}".format(score_c)
      ,"Random Forest:{}".format(score_r)
     )
classification_data, classification_labels = make_classification(n_samples=100, n_features=3, n_informative=1, n_classes=2, n_redundant=0, n_clusters_per_class=1)
X_train, X_test, y_train, y_test = train_test_split(classification_data, classification_labels, test_size=0.5)
clf_1 = DecisionTreeClassifier(random_state=0)
rfc_1 = RandomForestClassifier(random_state=0)
clf_1 = clf_1.fit(X_train, y_train)
rfc_1 = rfc_1.fit(X_train, y_train)
score_c_1 = clf_1.score(X_test, y_test)
score_r_1 = rfc_1.score(X_test, y_test)
print("Single Tree:{}".format(score_c_1)
      ,"Random Forest:{}".format(score_r_1)     )
classification_data, classification_labels = make_classification(n_samples=100, n_features=10, n_informative=1, n_classes=2, n_redundant=0, n_clusters_per_class=1)
X_train, X_test, y_train, y_test = train_test_split(classification_data, classification_labels, test_size=0.5)
clf_2 = DecisionTreeClassifier(random_state=0)
rfc_2 = RandomForestClassifier(random_state=0)
clf_2 = clf_2.fit(X_train, y_train)
rfc_2 = rfc_2.fit(X_train, y_train)
score_c_2 = clf_2.score(X_test, y_test)
score_r_2 = rfc_2.score(X_test, y_test)
print("Single Tree:{}".format(score_c_2)
      ,"Random Forest:{}".format(score_r_2)
     )
classification_data, classification_labels = make_classification(n_samples=100, n_features=50, n_informative=1, n_classes=2, n_redundant=0, n_clusters_per_class=1)
X_train, X_test, y_train, y_test = train_test_split(classification_data, classification_labels, test_size=0.5)
clf_3 = DecisionTreeClassifier(random_state=0)
rfc_3 = RandomForestClassifier(random_state=0)
clf_3 = clf_3.fit(X_train, y_train)
rfc_3 = rfc_3.fit(X_train, y_train)
score_c_3 = clf_3.score(X_test, y_test)
score_r_3 = rfc_3.score(X_test, y_test)
print("Single Tree:{}".format(score_c_3)
      ,"Random Forest:{}".format(score_r_3)
     )
plt.scatter(score_c, score_r)
plt.scatter(score_c_1, score_r_1)
plt.scatter(score_c_2, score_r_2)
plt.scatter(score_c_3, score_r_3)
print(plt.show())

