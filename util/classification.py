# This cell contains code form earlier notebooks, should be placed in util
from sklearn.model_selection import cross_val_predict
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np


def plot_roc_curve(fpr, tpr, label, title):
    plt.figure()
    plt.plot(fpr, tpr, color='green', lw=2, label=label)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()


def get_training_data_roc_cv(clf, x, y):
    # ROC curve  on the training data
    predictions_rf = cross_val_predict(clf, x, y, cv=2, method="predict_proba")
    fpr, tpr, thresholds = metrics.roc_curve(y, predictions_rf[:, 1], pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)
    plot_roc_curve(fpr, tpr, label=f'Random Forest (AUC = {roc_auc:.3f})', title="GILGFVFTL")
    print(f"ROC AUC: {roc_auc}")


def fix_test(x_test, train_columns):
    # x_test.fillna(0, inplace=True)
    for col in train_columns:
        if col not in x_test.columns:
            # only columns starting with pos_0 are allowed to be missing, the rest should already exist (be sure you use the test version of the onehot encoder if this isn't the case)
            assert col.startswith('alfa_pos_') or col.startswith('beta_pos_') or col.endswith('_count'), f'Column {col} not in test set'

            x_test[col] = np.nan  # TODO: NaN geven
            # print(f'Column {col} not in test set, added with NaN values')
    # remove all columns from x_test that are not in x
    x_test = x_test[train_columns]
    return x_test


def calculate_auc_and_plot(y_test, y_pred):
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)

    plot_roc_curve(fpr, tpr, label=f'ROC curve (area = {roc_auc:.3f})', title='ROC curve')
    return roc_auc
