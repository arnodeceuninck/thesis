import pandas as pd


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