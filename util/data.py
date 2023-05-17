import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import numpy as np


def get_train_dataset(vdjdb=False):
    """
    :return: The train part of the combined positive and negative dataset that was made in data.ipybn
    """
    if vdjdb:
        return pd.read_csv('data/generated_combined_dataset_train_vdjdb.csv')
    else:
        return pd.read_csv('data/generated_combined_dataset_train.csv')


def get_test_dataset(vdjdb=False):
    """
    :return: The test part of the combined positive and negative dataset that was made in data.ipybn
    """
    if vdjdb:
        return pd.read_csv('data/generated_combined_dataset_test_vdjdb.csv')
    else:
        return pd.read_csv('data/generated_combined_dataset_test.csv')


def sample_and_drop(df, n):
    """Sample n rows from df and drop them from df"""
    # src: https://stackoverflow.com/questions/39835021/pandas-random-sample-with-remove
    df_subset = df.sample(n)
    df.drop(df_subset.index, inplace=True)
    return df_subset


def process_gene(g):
    # Code based on code snippet sent by Ceder (on Slack)
    """
    Extracts the gene family, version and allele from a gene in the VDJdb.
    Forward slashes are removed (to make e.g. TRAV14/DV4 and TRAV14DV4 equal), make sure only 1 gene is given as input
    The allele is set to 01 if it is not specified.

    If g is not a string, this functions returns (g, g). This can be used for missing chains when g is None or NaN

    :param g: TRAV, TRAJ, TRBV or TRBJ gene
    :return: the gene formatted as {family}-{version}*{allele} and the family
    """
    # Example:
    # TRAV38-2/DV8*01
    # Get's converted to:
    # TRAV38 2DV8 (and the allele is removed, since it isn't in the negative dataset)

    # If the gene is None or NaN, return the same as the family
    if not isinstance(g, str):
        return g, g

    g_old = g

    # Remove all forward slashes to make e.g. TRAV14/DV4 and TRAV14DV4 equal.
    # g = g.replace('/', '') # TODO?

    # Extract the allele (and set to 01 if none given)
    g_allele = g.split('*')
    if len(g_allele) == 1:
        allele = '01'
    elif len(g_allele) == 2:
        g = g_allele[0]
        allele = g_allele[1]
    else:
        raise RuntimeError(f"splitting full gene on '*' resulted in >2 items: {g_old}")

    # Extract the version (and set to 1 if none given)
    group_version = g.split('-')
    if len(group_version) == 1:
        version = '1'
    elif len(group_version) == 2:
        g = group_version[0]
        version = group_version[1]
    else:
        if len(group_version) == 3:
            # ASKED: What should I actually do here? TRAV15-2/DV6-2*01 -> Is okay as I dit this
            g = group_version[0]
            version = group_version[2]

            # print(f"Warning: error splitting gene {g_old}, fixed to {g}-{version}")
            return g, version

        raise RuntimeError(f"splitting full gene on '-' resulted in >2 items: {g_old}")

    return g, version


def split_gene_in_columns(df):
    # Process the genes
    for col in ['TRAJ', 'TRAV', 'TRBV', 'TRBJ']:
        # process gene returns the gene and the version, create two new columns for this
        df[col + '_family'], df[col + '_version'] = zip(*df[col].apply(process_gene))
        # # Remove the original column
        # combined_dataset.drop(col, axis=1, inplace=True)


def plot_scores(scores, baseline_name, y_label="AUC", x_label="Model", rotate_x=False):

    if baseline_name is not None:
        baseline = scores[baseline_name]

        diff = [scores[score] - baseline for score in scores]
    else:
        diff = [scores[score] for score in scores]

    graph = sns.barplot(x=list(scores.keys()), y=diff)

    if baseline_name is not None:
        # draw a horizontal line at y=baseline and label it
        graph.axhline(0, color="k", clip_on=False)

        y_ticks = graph.get_yticks()
        graph.set_yticklabels([f"{baseline+y:.2f}" for y in y_ticks])

    if rotate_x:
        graph.set_xticklabels(graph.get_xticklabels(), rotation=90)

    graph.set_ylabel(y_label)
    graph.set_xlabel(x_label)

    plt.show()

    # clear the graph for future plots
    plt.clf()


def readDataframeToListDataset(features_df, labels_df=None, label_col_name=None):
    """"
    Read a pandas dataframe to a ListDataset
    :param features_df: pandas dataframe with features (and labels if labels_df is None)
    :param labels_df: pandas dataframe with labels
    :param label_col_name: name of the column with the labels (if labels_df is None)
    :return: ListDataset
    """

    from dataStructures import ListDataset

    if labels_df is None:
        assert label_col_name is not None, "label_col_name must be specified if labels_df is None"
        labels_df = features_df[label_col_name]
        features_df = features_df.drop(label_col_name, axis=1)

    assert len(features_df) == len(labels_df), "features_df and labels_df must have the same length"
    assert np.isnan(labels_df).sum() == 0, "NaN values in labels_df"

    df = features_df.reset_index(drop=True)
    assert 'target' not in df.columns, "'target' column already exists in features_df. Please rename it or specify using label_col_name."
    df['target'] = labels_df.reset_index(drop=True)

    start = time.time()
    dataset = ListDataset.ListDataset()

    for index, row in df.iterrows():
        series = row.values[:-1]
        label = row.values[-1]
        if label not in [0, 1]:
            raise ValueError("Label must be 0 or 1")
        assert not np.isnan(label), "Label cannot be NaN"
        dataset.add_series(label, series)

    end = time.time()
    elapsed = end - start
    print(f"Parsing process finished in {elapsed} seconds")

    return dataset
