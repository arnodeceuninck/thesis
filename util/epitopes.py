import pandas as pd
from util import split_gene_in_columns


def get_vdjdb():
    df_vdjdb = pd.read_csv('data/vdjdb-2022-03-30/vdjdb_full.txt', sep='\t')

    rename_columns = {
        'antigen.epitope': 'epitope',
        'cdr3.alpha': 'CDR3_alfa',
        'v.alpha': 'TRAV',
        'j.alpha': 'TRAJ',
        'cdr3.beta': 'CDR3_beta',
        'v.beta': 'TRBV',
        'j.beta': 'TRBJ',
    }

    df_vdjdb = df_vdjdb.rename(columns=rename_columns)
    df_vdjdb = df_vdjdb[rename_columns.values()]

    return df_vdjdb


# Remove items from negative_samples that are in positive_samples
def remove_items_occuring_in_other_column(df1, column_to_remove, df2, column_to_check):
    """Remove all items from column_to_remove that occur in column_to_check"""
    items_to_remove = df2[column_to_check].unique()

    # remove NaN values
    items_to_remove = items_to_remove[~pd.isnull(items_to_remove)]

    df = df1[~df1[column_to_remove].isin(items_to_remove)]
    return df


def remove_negative_positive_cdr3_overlap(negative_samples, positive_samples):
    previous_len = len(negative_samples)

    negative_samples = remove_items_occuring_in_other_column(negative_samples, 'CDR3_alfa', positive_samples,
                                                             'CDR3_alfa')
    negative_samples = remove_items_occuring_in_other_column(negative_samples, 'CDR3_beta', positive_samples,
                                                             'CDR3_beta')

    if len(negative_samples) != previous_len:
        print(f'Number of negative samples changed from {previous_len} to {len(negative_samples)}')

    return negative_samples


# get the counts of positive dataset containing alpha, beta and both (so the number of columns where it's not NaN)
def filter_df(df, alpha_not_nan, beta_not_nan):
    alpha_condition = df['CDR3_alfa'].notna() if alpha_not_nan else df['CDR3_alfa'].isna()
    beta_condition = df['CDR3_beta'].notna() if beta_not_nan else df['CDR3_beta'].isna()
    return df[alpha_condition & beta_condition]


def get_counts(positive_samples):
    alpha_only_count_pos = len(filter_df(positive_samples, alpha_not_nan=True, beta_not_nan=False))
    beta_only_count_pos = len(filter_df(positive_samples, alpha_not_nan=False, beta_not_nan=True))
    both_count_pos = len(filter_df(positive_samples, alpha_not_nan=True, beta_not_nan=True))
    non_count_pos = len(filter_df(positive_samples, alpha_not_nan=False, beta_not_nan=False))
    return alpha_only_count_pos, beta_only_count_pos, both_count_pos, non_count_pos


def get_negative_subsets(negative_samples):
    negative_alpha_only = filter_df(negative_samples, alpha_not_nan=True, beta_not_nan=False)
    negative_beta_only = filter_df(negative_samples, alpha_not_nan=False, beta_not_nan=True)
    negative_both = filter_df(negative_samples, alpha_not_nan=True, beta_not_nan=True)
    negative_none = filter_df(negative_samples, alpha_not_nan=False, beta_not_nan=False)
    return negative_alpha_only, negative_beta_only, negative_both, negative_none


# Create a custom exception for when there are not enough samples
class NotEnoughSamplesException(Exception):
    pass


def get_negative_df(negative_alpha_only, negative_beta_only, negative_both, negative_none,
                    alpha_only_count_pos, beta_only_count_pos, both_count_pos, non_count_pos):
    try:
        # Sample the negative samples to the same size as the positive samples
        negative_df_alpha_only = negative_alpha_only.sample(n=alpha_only_count_pos, random_state=42)
        negative_df_beta_only = negative_beta_only.sample(n=beta_only_count_pos, random_state=42)
        negative_df_both = negative_both.sample(n=both_count_pos, random_state=42)
        negative_df_none = negative_none.sample(n=non_count_pos, random_state=42)
    except ValueError:
        print('Not enough negative samples to sample from')
        raise NotEnoughSamplesException()

    negative_df = pd.concat([negative_df_alpha_only, negative_df_beta_only, negative_df_both, negative_df_none])
    return negative_df


def combine_and_shuffle(positive_samples, negative_df, current_epitope):
    # Combine positive_samples and negative_df and shuffle
    df = pd.concat([positive_samples, negative_df])
    df['reaction'] = df['epitope'].apply(lambda x: 1 if x == current_epitope else 0)
    df = df.drop(columns=['epitope'])
    df = df.sample(frac=1, random_state=42)
    return df


def get_epitope_df(epitope, silent=False):
    df_vdjdb = get_vdjdb()

    positive_samples = df_vdjdb[df_vdjdb['epitope'] == epitope]
    negative_samples = df_vdjdb[df_vdjdb['epitope'] != epitope]

    negative_samples = remove_negative_positive_cdr3_overlap(negative_samples, positive_samples)

    alpha_only_count_pos, beta_only_count_pos, both_count_pos, non_count_pos = get_counts(positive_samples)
    print(
        f'Positive samples: alpha only: {alpha_only_count_pos}, beta only: {beta_only_count_pos}, both: {both_count_pos}, none: {non_count_pos}') if not silent else None

    negative_alpha_only, negative_beta_only, negative_both, negative_none = get_negative_subsets(negative_samples)
    print(
        f'Negative samples: alpha only: {len(negative_alpha_only)}, beta only: {len(negative_beta_only)}, both: {len(negative_both)}, none: {len(negative_none)}') if not silent else None

    negative_df = get_negative_df(negative_alpha_only, negative_beta_only, negative_both, negative_none,
                                  alpha_only_count_pos, beta_only_count_pos, both_count_pos, non_count_pos)

    df = combine_and_shuffle(positive_samples, negative_df, epitope)

    split_gene_in_columns(df)

    return df
