from Read_DataSet import *
from sklearn.model_selection import cross_validate
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt


def cross_validation(model, x_data, y_data, k):
    scores = ['accuracy', 'precision_micro', 'recall_micro', 'f1_micro']
    answer = cross_validate(estimator=model, X=x_data, y=y_data,
                            cv=k, scoring=scores, return_train_score=True)
    results_train = {"Training Accuracy": answer['train_accuracy'],
                     "Mean of Training Accuracy": answer['train_accuracy'].mean(),
                     "Training Precision": answer['train_precision_micro'],
                     "Mean of Training Precision": answer['train_precision_micro'].mean(),
                     "Training Recall": answer['train_recall_micro'],
                     "Mean of Training Recall": answer['train_recall_micro'].mean(),
                     "Training F1_Score": answer['train_f1_micro'], "Mean of Training F1_Score": answer['train_f1_micro'].mean()}
    results_test = {"Validation Accuracy": answer['test_accuracy'],
                    "Mean of Validation Accuracy": answer['test_accuracy'].mean(),
                    "Validation Precision": answer['test_precision_micro'],
                    "Mean of Validation Precision": answer['test_precision_micro'].mean(),
                    "Validation Recall": answer['test_recall_micro'],
                    "Mean of Validation Recall": answer['test_recall_micro'].mean(),
                    "Validation F1_Score": answer['test_f1_micro'],
                    "Mean of Validation F1_Score": answer['test_f1_micro'].mean()}
    return results_train, results_test


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


def make_decision_tree():
    decision_tree = DecisionTreeClassifier(criterion="entropy", random_state=0, min_samples_split=5)
    decision_tree_train_result, decision_tree_test_result = cross_validation(decision_tree, x_train, y_train, k=5)
    print(decision_tree_train_result)
    print()
    print(decision_tree_test_result)
    return decision_tree_train_result, decision_tree_test_result


def show_plot():
    number = 2
    decision_tree_train_result, decision_tree_test_result = make_decision_tree()
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


show_plot()