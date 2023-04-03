from Read_DataSet import *
from sklearn.model_selection import cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, confusion_matrix
import matplotlib.pyplot as plt
from copy import deepcopy


def cross_validation(model, x_data, y_data, k):
    scores = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
    answer = cross_validate(estimator=model, X=x_data, y=y_data,
                            cv=k, scoring=scores, return_train_score=True, return_estimator=True)
    results_train = {"Training Accuracy": answer['train_accuracy'],
                     "Mean of Training Accuracy": answer['train_accuracy'].mean(),
                     "Training Precision": answer['train_precision_weighted'],
                     "Mean of Training Precision": answer['train_precision_weighted'].mean(),
                     "Training Recall": answer['train_recall_weighted'],
                     "Mean of Training Recall": answer['train_recall_weighted'].mean(),
                     "Training F1_Score": answer['train_f1_weighted'],
                     "Mean of Training F1_Score": answer['train_f1_weighted'].mean()}
    results_test = {"Validation Accuracy": answer['test_accuracy'],
                    "Mean of Validation Accuracy": answer['test_accuracy'].mean(),
                    "Validation Precision": answer['test_precision_weighted'],
                    "Mean of Validation Precision": answer['test_precision_weighted'].mean(),
                    "Validation Recall": answer['test_recall_weighted'],
                    "Mean of Validation Recall": answer['test_recall_weighted'].mean(),
                    "Validation F1_Score": answer['test_f1_weighted'],
                    "Mean of Validation F1_Score": answer['test_f1_weighted'].mean()}
    estimators = answer['estimator']
    return results_train, results_test, estimators


def make_plot(x_label, y_label, plot_title, train_data, val_data):
    plt.figure(figsize=(12, 6))
    labels = ["1st Fold", "2nd Fold", "3rd Fold", "4th Fold", "5th Fold"]
    X_axis = np.arange(len(labels))
    ax = plt.gca()
    plt.ylim(0.40000, 1)
    plt.bar(X_axis - 0.2, train_data, 0.4, color='blue', label='Training')
    plt.bar(X_axis + 0.2, val_data, 0.4, color='red', label='Validation')
    plt.title(plot_title, fontsize=30)
    plt.xticks(X_axis, labels)
    plt.xlabel(x_label, fontsize=14)
    plt.ylabel(y_label, fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.show()


def make_plot_test(x_label, y_label, plot_title, data):
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


def make_decision_tree():
    decision_tree = DecisionTreeClassifier(criterion="entropy", random_state=0, min_samples_split=5)
    decision_tree_train_result, decision_tree_test_result, estimators = cross_validation(decision_tree, x_train,
                                                                                         y_train, k=5)
    # Test DataSet
    results = []  # Consist all 5 estimators results
    for i in range(len(estimators)):
        temp = estimators[i].predict(x_test)
        results.append(deepcopy(temp))
    temp_acc = []
    temp_pre = []
    temp_rec = []
    temp_f1 = []
    confusion_matrix_test = []
    for i in range(len(results)):
        temp_acc.append(accuracy_score(y_test, results[i]))
        temp_pre.append(precision_score(y_test, results[i], average='weighted'))
        temp_rec.append(recall_score(y_test, results[i], average='weighted'))
        temp_f1.append(f1_score(y_test, results[i], average='weighted'))
        confusion_matrix_test.append(confusion_matrix(y_test, results[i]))
    temp_acc = np.array(temp_acc)
    temp_pre = np.array(temp_pre)
    temp_rec = np.array(temp_rec)
    temp_f1 = np.array(temp_f1)
    print(confusion_matrix_test)
    # Confusion Matrix of Train DataSet
    train_results = []
    confusion_matrix_train = []
    for i in range(len(estimators)):
        train_results.append(estimators[i].predict(x_train))
    for i in range(len(train_results)):
        confusion_matrix_train.append(confusion_matrix(y_train, train_results[i]))
    print(confusion_matrix_train)
    test_results = np.array([temp_acc.mean(), temp_pre.mean(), temp_rec.mean(), temp_f1.mean()])
    return decision_tree_train_result, decision_tree_test_result, test_results


def show_plot():
    number = 2
    decision_tree_train_result, decision_tree_test_result, test_results = make_decision_tree()
    model_name = "Decision Tree"
    scores = ['Accuracy', 'Precision', 'Recall', 'F1_Score']
    scores_descriptions = ["Accuracy in 5 Folds", "Precision in 5 Folds",
                           "Recall in 5 Folds", "F1_Score in 5 Folds"]
    train_keys = ["Training Accuracy", "Training Precision",
                  "Training Recall", "Training F1_Score"]
    test_keys = ["Validation Accuracy", "Validation Precision",
                 "Validation Recall", "Validation F1_Score"]
    make_plot(model_name, scores[number], scores_descriptions[number],
              decision_tree_train_result[train_keys[number]],
              decision_tree_test_result[test_keys[number]])
    make_plot_test(model_name, "Metrics", "Test Results", test_results)


show_plot()
