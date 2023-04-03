import copy

import numpy as np

from Read_DataSet import *
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, confusion_matrix
import matplotlib.pyplot as plt
from copy import deepcopy


def cross_validation(model, x_data, y_data, k):
    param_grid = {'max_features': ['auto', 'sqrt', 'log2'],
                  'ccp_alpha': [0.1, .01, .001],
                  'max_depth': [5, 6, 7, 8, 9],
                  'criterion': ['gini', 'entropy']
                  }
    gridsearch = GridSearchCV(model, param_grid=param_grid, cv=k, verbose=True)
    gridsearch.fit(x_data, y_data)
    final_model = gridsearch.best_estimator_
    best_params = gridsearch.best_params_
    return final_model, best_params


def make_plot(x_label, y_label, plot_title, data):
    plt.figure(figsize=(12, 6))
    labels = ["Accuracy", "Precision", "Recall", "F1_Score"]
    X_axis = np.arange(len(labels))
    ax = plt.gca()
    plt.ylim(0.40000, 1)
    plt.bar(X_axis - 0.2, data, 0.4, color='green', label='Test')
    plt.title(plot_title, fontsize=30)
    plt.xticks(X_axis, labels)
    plt.xlabel(x_label, fontsize=14)
    plt.ylabel(y_label, fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.show()


def make_random_forest():
    random_forest = RandomForestClassifier(criterion="entropy", random_state=0, min_samples_split=5, n_estimators=10)
    estimator, params = cross_validation(random_forest, x_train, y_train, k=5)
    estimator.fit(x_train, y_train)
    result_test = estimator.predict(x_test)
    result_train = estimator.predict(x_train)
    acc_test = accuracy_score(y_test, result_test)
    acc_train = accuracy_score(y_train, result_train)
    pre_test = precision_score(y_test, result_test, average='weighted')
    pre_train = precision_score(y_train, result_train, average='weighted')
    rec_test = recall_score(y_test, result_test, average='weighted')
    rec_train = recall_score(y_train, result_train, average='weighted')
    f1_test = f1_score(y_test, result_test, average='weighted')
    f1_train = f1_score(y_train, result_train, average='weighted')
    con_test = confusion_matrix(y_test, result_test, labels=[1, 2, 3, 4, 5])
    con_train = confusion_matrix(y_train, result_train, labels=[1, 2, 3, 4, 5])
    test_metrics = [acc_test, pre_test, rec_test, f1_test]
    test_metrics = np.array(test_metrics)
    train_metrics = [acc_train, pre_train, rec_train, f1_train]
    train_metrics = np.array(train_metrics)
    make_plot("Decision Tree", "Metrics", "Test DataSet", test_metrics)
    make_plot("Decision Tree", "Metrics", "Train DataSet", train_metrics)
    print("Test Confusion Matrix")
    print(con_test)
    print("Train Confusion Matrix")
    print(con_train)
    print("Best Parameters")
    print(params)


make_random_forest()
