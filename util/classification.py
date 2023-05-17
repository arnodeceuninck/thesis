# This cell contains code form earlier notebooks, should be placed in util
from sklearn.model_selection import cross_val_predict, KFold, cross_val_score, train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from util import get_features, get_columns_starting_with


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
    cols_to_add = []
    for col in train_columns:
        if col not in x_test.columns:
            # only columns starting with pos_0 are allowed to be missing, the rest should already exist (be sure you use the test version of the onehot encoder if this isn't the case)
            assert col.startswith('alpha_pos_') or col.startswith('beta_pos_') or col.endswith(
                '_count') or col in ['beta_J', 'beta_V', 'alpha_J',
                                     'alpha_V'], f'Column {col} not in test set'  # Don't know whether col in ['beta_J', 'beta_V', 'alpha_J','alpha_V'] should be aloowed, was required for alpha beta knn
            cols_to_add.append(col)

            # line below raises performance error, which is why I add them all at once using cols to add
            # https://stackoverflow.com/questions/68292862/performancewarning-dataframe-is-highly-fragmented-this-is-usually-the-result-o
            # x_test[col] = 0 #np.nan  # TODO: NaN of 0? currently kept it at 0 to make clear it's not because the chain was missing

            # print(f'Column {col} not in test set, added with NaN values')
    # x_test = x_test.copy()
    # create a dataframe with all columns to add, set to 0
    df = pd.DataFrame(np.zeros((x_test.shape[0], len(cols_to_add))), columns=cols_to_add)
    # add the new columns to the test set
    x_test = pd.concat([x_test, df], axis=1)
    # x_test[cols_to_add] = 0

    # remove all columns from x_test that are not in x
    x_test = x_test[train_columns]
    return x_test


def calculate_auc_and_plot(y_test, y_pred, plot=True):
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)

    plot_roc_curve(fpr, tpr, label=f'ROC curve (area = {roc_auc:.3f})', title='ROC curve') if plot else None
    return roc_auc


def evaluate(clf, x, y):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(clf, x, y, cv=kf, scoring='roc_auc')
    print(scores)
    print(f"ROC: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")


def evaluate_no_cv(clf, x, y, x_test, y_test, model_imputer=None):
    if model_imputer is not None:
        x = model_imputer.fit_transform(x)

    clf.fit(x, y)
    y_pred = clf.predict_proba(x_test)[:, 1]
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)
    # print(f"ROC AUC: {roc_auc:.3f}")
    return roc_auc

class TestSetSmallException(Exception):
    pass

def get_train_test(df, seed, drop_train_na=False):
    train, test = train_test_split(df, test_size=0.2, random_state=seed)

    test.dropna(inplace=True)
    if drop_train_na:
        train.dropna(inplace=True)

    if len(test) < 10:
        raise TestSetSmallException(f'Test set too small after dropping NaNs. Test set samples left: {len(test)}')

    x = get_features(train)
    y = train['reaction']

    x_test = get_features(test, test=True)
    x_test = fix_test(x_test, x.columns)
    y_test = test['reaction']

    return x, y, x_test, y_test


def evaluate_cv_no_nan_test(models_to_evaluate, df, folds=5):
    scores = pd.DataFrame()

    for i, seed in enumerate(range(folds)):
        # print(f'Fold {i + 1}/{folds}')

        for j, model in enumerate(models_to_evaluate):
            # print("HIT")
            try:
                # print(f"Running {model['name']} ({j + 1}/{len(models_to_evaluate)})")

                drop_train_na = model.get('drop_train_na', False)

                x, y, x_test, y_test = get_train_test(df, seed, drop_train_na=drop_train_na)

                if not model.get('seperate_chains', False):
                    if 'imputer' in model:
                        x = model['imputer'].fit_transform(x)

                    auc = evaluate_no_cv(model['model'], x, y, x_test, y_test)
                else:
                    # raise NotImplementedError('seperate chains not implemented yet (gave errors, evaluat function needs df)')
                    imputer = model.get('imputer', None)
                    auc = evaluate_seperate_chains(model['model_alpha'], model['model_beta'], x, y, x_test, y_test, imputer)

                print(f"AUC(m={model['name']}, s={seed}, train_na={int(not drop_train_na)}): {auc}")

                index = len(scores)
                scores.loc[index, 'model'] = model['name']
                scores.loc[index, 'auc'] = auc
                scores.loc[index, 'drop_train_na'] = drop_train_na
            except TestSetSmallException as e:
                print(e)
                continue
            except Exception as e:
                print(f"Error: {e}")
                raise e
                continue

    return scores


def evaluate_seperate_chains(clf1, clf2, x, y, x_test, y_test, imputer=None):
    # Keep only the columns starting with 'alpha_'
    x_alpha = get_columns_starting_with(x, 'alpha_')
    x_beta = get_columns_starting_with(x, 'beta_')

    # remember those, since the imputer removes them
    x_alpha_columns = x_alpha.columns
    x_beta_columns = x_beta.columns

    if imputer is not None:
        x_alpha = imputer.fit_transform(x_alpha)
        x_beta = imputer.fit_transform(x_beta)

    # print(f'starting')
    clf1.fit(x_alpha, y)
    # print('Alpha fitting don')
    clf2.fit(x_beta, y)

    x_test_alpha = get_columns_starting_with(x_test, 'alpha_')
    x_test_beta = get_columns_starting_with(x_test, 'beta_')

    x_test_alpha = fix_test(x_test_alpha, x_alpha_columns)
    x_test_beta = fix_test(x_test_beta, x_beta_columns)

    y_pred1 = clf1.predict_proba(x_test_alpha)[:, 1]
    y_pred2 = clf2.predict_proba(x_test_beta)[:, 1]

    y_pred = (y_pred1 + y_pred2) / 2
    auc = calculate_auc_and_plot(y_test, y_pred, plot=False)

    # print(f"ROC AUC: {auc:.3f}")

    return auc

# Not working
# class NoNanInTestKFold():
#     def __init__(self, df_original, n_splits=5, shuffle=False, random_state=None):
#         self.n_splits = n_splits
#         self.shuffle = shuffle
#         self.random_state = random_state
#         self.df_original = df_original # The original dataframe the x features are generated from, without everything changed to feature (since features contain a lot of NaNs)
#
#     def split(self, X, y=None, groups=None):
#         kf = KFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state)
#         for train_index, test_index in kf.split(X, y, groups):
#             # remove all rows from test_index that have any NaN values in X
#             test_index_new = test_index[~np.isnan(self.df_original.iloc[test_index]).any(axis=1)]
#             yield train_index, test_index_new
#
#
#     def get_n_splits(self, X, y, groups=None):
#         return self.n_splits
#
#
# def evaluate_no_nan_test(clf, x, y, df_original):
#     # Doesn't work because of inputations
#     kf = NoNanInTestKFold(df_original=df_original, n_splits=5, shuffle=True, random_state=42)
#     scores = cross_val_score(clf, x, y, cv=kf, scoring='roc_auc')
#     print(scores)
#     print(f"ROC: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
