import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def get_train_dataset():
    """
    :return: The train part of the combined positive and negative dataset that was made in data.ipybn
    """

    return pd.read_csv('data/generated_combined_dataset_train.csv')


def get_test_dataset():
    """
    :return: The test part of the combined positive and negative dataset that was made in data.ipybn
    """

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
    g = g.replace('/', '')

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
            # TODO: TOASK: What should I actually do here? TRAV15-2/DV6-2*01
            g = group_version[0]
            version = group_version[2]

            print(f"Warning: error splitting gene {g_old}, fixed to {g}-{version}")
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
