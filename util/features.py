from sklearn import feature_extraction
import pandas as pd
import numpy as np
from pyteomics import parser, mass, electrochem

basicity = {'A': 206.4, 'B': 210.7, 'C': 206.2, 'D': 208.6, 'E': 215.6, 'F': 212.1, 'G': 202.7,
            'H': 223.7, 'I': 210.8, 'K': 221.8, 'L': 209.6, 'M': 213.3, 'N': 212.8, 'P': 214.4,
            'Q': 214.2, 'R': 237.0, 'S': 207.6, 'T': 211.7, 'V': 208.7, 'W': 216.1, 'X': 210.2,
            'Y': 213.1, 'Z': 214.9}

hydrophobicity = {'A': 0.16, 'B': -3.14, 'C': 2.50, 'D': -2.49, 'E': -1.50, 'F': 5.00, 'G': -3.31,
                  'H': -4.63, 'I': 4.41, 'K': -5.00, 'L': 4.76, 'M': 3.23, 'N': -3.79, 'P': -4.92,
                  'Q': -2.76, 'R': -2.77, 'S': -2.85, 'T': -1.08, 'V': 3.02, 'W': 4.88, 'X': 4.59,
                  'Y': 2.00, 'Z': -2.13}

helicity = {'A': 1.24, 'B': 0.92, 'C': 0.79, 'D': 0.89, 'E': 0.85, 'F': 1.26, 'G': 1.15, 'H': 0.97,
            'I': 1.29, 'K': 0.88, 'L': 1.28, 'M': 1.22, 'N': 0.94, 'P': 0.57, 'Q': 0.96, 'R': 0.95,
            'S': 1.00, 'T': 1.09, 'V': 1.27, 'W': 1.07, 'X': 1.29, 'Y': 1.11, 'Z': 0.91}

mutation_stability = {'A': 13, 'C': 52, 'D': 11, 'E': 12, 'F': 32, 'G': 27, 'H': 15, 'I': 10,
                      'K': 24, 'L': 34, 'M': 6, 'N': 6, 'P': 20, 'Q': 10, 'R': 17, 'S': 10,
                      'T': 11, 'V': 17, 'W': 55, 'Y': 31}

PHYSCHEM_PROPERTIES = {'basicity': basicity, 'hydrophobicity': hydrophobicity,
                       'helicity': helicity, 'mutation_stability': mutation_stability}


def get_basicity(amino_acid):
    return basicity[amino_acid]


def get_hydrophobicity(amino_acid):
    return hydrophobicity[amino_acid]


def get_helicity(amino_acid):
    return helicity[amino_acid]


def get_mutation_stability(amino_acid):
    return mutation_stability[amino_acid]


ONEHOT_ENCODER = {"alpha": None, "beta": None}
ALPHA_OR_BETA = None  # Should be set to "alpha" or "beta"


def onehot_encode_test(df):
    return onehot_encode(df, test=True)


def onehot_encode(df, test=False):
    global ONEHOT_ENCODER, ALPHA_OR_BETA
    assert ALPHA_OR_BETA in ["alpha",
                             "beta"], f"ALPHA_OR_BETA must be set to 'alpha' or 'beta', is currently set to {ALPHA_OR_BETA}"

    # One hot encode the columns (creates a new column per unique value here and fills it with 1 or 0)
    # get all columns in df with the name 'V' or 'J' or that starts with 'V_' or 'J_'
    onehot_cols = [col for col in df.columns if col.startswith('V_') or col.startswith('J_') or col in ['V', 'J']]
    if not test:
        ONEHOT_ENCODER[ALPHA_OR_BETA] = feature_extraction.DictVectorizer(sparse=False)
        encodings = ONEHOT_ENCODER[ALPHA_OR_BETA].fit_transform(df[onehot_cols].to_dict(orient='records'))
        # print(f"encoder {ALPHA_OR_BETA} fitted")
    else:
        assert ONEHOT_ENCODER[ALPHA_OR_BETA] is not None
        encodings = ONEHOT_ENCODER[ALPHA_OR_BETA].transform(df[onehot_cols].to_dict(orient='records'))
        # print(f"only transformed {ALPHA_OR_BETA}")

    # Get the columns of the onehot encoder
    columns = ONEHOT_ENCODER[ALPHA_OR_BETA].get_feature_names_out()
    # print(columns)

    onehot_df = pd.DataFrame(encodings, columns=ONEHOT_ENCODER[ALPHA_OR_BETA].get_feature_names_out())

    for col in columns:
        assert col in onehot_df.columns, f'Column {col} not in dataframe'

    return onehot_df


def get_length(sequence):
    if not isinstance(sequence, str):
        # It's probably NaN
        # return 0
        return np.nan
    else:
        return parser.length(sequence)


def cdr3_length(df):
    return df['CDR3'].apply(get_length).to_frame('CDR3_length')


def get_amino_acid_composition(sequence):
    if not isinstance(sequence, str):
        # It's probably NaN
        return {}  # will be converted to NaN anyway
    else:
        composition = parser.amino_acid_composition(sequence)
        return composition


def aa_occurances(df):
    composition = [get_amino_acid_composition(sequence) for sequence in df['CDR3']]
    # aa_alpha_counts = pd.DataFrame.from_records(composition).fillna(0)
    aa_alpha_counts = pd.DataFrame.from_records(composition).fillna(0)  # Filling na with 0, since they weren't counted
    aa_alpha_counts.columns = [f'{column}_count' for column in aa_alpha_counts.columns]
    return aa_alpha_counts


def get_property(sequence, prop_lookup):
    if not isinstance(sequence, str):
        # It's probably NaN
        # return 0
        return np.nan
    else:
        return np.mean(list(prop_lookup[aa] for aa in sequence))


def physchem_properties(df):
    physchem_df = pd.DataFrame()
    for prop_name, prop_lookup in PHYSCHEM_PROPERTIES.items():
        physchem_df[prop_name] = df['CDR3'].apply(get_property, args=(prop_lookup,))
    return physchem_df


def get_mass(sequence):
    if not isinstance(sequence, str):
        # It's probably NaN
        # return 0
        return np.nan
    else:
        return mass.fast_mass(sequence)


def peptide_mass(df):
    return df['CDR3'].apply(get_mass).to_frame('peptide_mass')


def get_pi(sequence):
    if not isinstance(sequence, str):
        # return 0
        return np.nan
    else:
        return electrochem.pI(sequence)


def pi_feature(df):
    return df['CDR3'].apply(get_pi).to_frame('pi')


def pos_features(df):
    features_list = []
    pos_aa, pos_basicity, pos_hydro, pos_helicity, pos_mutation, pos_pI = [[] for _ in range(6)]

    for sequence in df['CDR3']:
        if not isinstance(sequence, str):
            # It's probably NaN, can't just continue, since we'll have less rows then
            # nan_value = 0
            nan_value = np.nan
            pos_aa.append({f'pos_0_A': nan_value})
            pos_basicity.append({f'pos_0_basicity': nan_value})
            pos_hydro.append({f'pos_0_hydrophobicity': nan_value})
            pos_helicity.append({f'pos_0_helicity': nan_value})
            pos_mutation.append({f'pos_0_mutation_stability': nan_value})
            pos_pI.append({f'pos_0_pi': nan_value})
            continue

        length = get_length(sequence)

        start_pos = -1 * (length // 2)

        # Ranges are averaged around 0, so if mod 2 = 1, we need to include 0, else not
        if length % 2 == 1:
            pos_range = list(range(start_pos, start_pos + length))
        else:
            pos_range = list(range(start_pos, 0)) + list(range(1, start_pos + length + 1))

        # bool 1 or 0 if amino acid is present at position
        pos_aa.append({f'pos_{pos}_{aa}': 1 for pos, aa in zip(pos_range, sequence)})

        pos_basicity.append({f'pos_{pos}_basicity': get_basicity(aa) for pos, aa in zip(pos_range, sequence)})
        pos_hydro.append({f'pos_{pos}_hydrophobicity': get_hydrophobicity(aa) for pos, aa in zip(pos_range, sequence)})
        pos_helicity.append({f'pos_{pos}_helicity': get_helicity(aa) for pos, aa in zip(pos_range, sequence)})
        pos_mutation.append(
            {f'pos_{pos}_mutation_stability': get_mutation_stability(aa) for pos, aa in zip(pos_range, sequence)})

        pos_pI.append({f'pos_{pos}_pI': electrochem.pI(aa) for pos, aa in zip(pos_range, sequence)})

    features_list.append(pd.DataFrame.from_records(pos_aa))
    features_list.append(pd.DataFrame.from_records(pos_basicity))
    features_list.append(pd.DataFrame.from_records(pos_hydro))
    features_list.append(pd.DataFrame.from_records(pos_helicity))
    features_list.append(pd.DataFrame.from_records(pos_mutation))
    features_list.append(pd.DataFrame.from_records(pos_pI))

    return pd.concat(features_list, axis=1).fillna(0)


def get_baseline_feature_functions(test, columns_to_use):

    feature_functions = []

    ohe_columns = ["V", "J", "V_family", "J_family", "V_version", "J_version"]
    if any([column in columns_to_use for column in ohe_columns]):
        ohe = onehot_encode if not test else onehot_encode_test
        feature_functions.append(ohe)

    if "CDR3" in columns_to_use:
        feature_functions.extend([cdr3_length, aa_occurances, physchem_properties, peptide_mass, pi_feature, pos_features])

    return feature_functions


def get_baseline_sequence_features(df, test, columns_to_use):
    features_list = []
    for feature_function in get_baseline_feature_functions(test, columns_to_use):
        features = feature_function(df)
        if features.shape[0] == 0:
            # No features extracted (e.g. in the case of aa_occurances of all NaN)
            # Should I just skip it, or set this feature to 0? the fix_test function solves this, so we can just skip
            continue
        assert features.shape[0] == df.shape[
            0], f'Feature function {feature_function} returned {features.shape[0]} rows, expected {df.shape[0]}'
        features_list.append(features.reset_index(drop=True))

    # Create one large dataframe, consisting of all the features (number of rows remains the same)
    features_in_one_df = pd.concat(features_list, axis=1)
    assert features_in_one_df.shape[0] == df.shape[
        0], f'Feature functions returned {features_in_one_df.shape[0]} rows, expected {df.shape[0]}'

    # set all values to nan if that row is also nan in the original df
    df.reset_index(drop=True, inplace=True)
    features_in_one_df[df['CDR3'].isna()] = np.nan

    return features_in_one_df


def update_alpha_or_beta(new_value):
    """
    A seperate function for updating the ALPHA_OR_BETA variable, since global variables act different in notebooks
    :param new_value:
    :return:
    """
    global ALPHA_OR_BETA
    ALPHA_OR_BETA = new_value


def get_features(df, test=False, columns=None, cdr_only=False, extract_features=True):
    if cdr_only:
        assert columns is None, 'Columns should be None when cdr_only is True'
        columns = ['CDR3']

    if columns is None:
        columns = ['CDR3', 'V', 'J']

    df_num_rows = df.shape[0]

    global ALPHA_OR_BETA

    chains = {"alpha": 'A', "beta": 'B'}
    all_chains_features = []
    for chain in chains:
        ALPHA_OR_BETA = chain

        short = chains[chain]
        rename_map = {f"CDR3_{chain}": "CDR3",
                      f"TR{short}V": "V",
                      f"TR{short}J": "J",
                      f"TR{short}V_family": "V_family",
                      f"TR{short}J_family": "J_family",
                      f"TR{short}V_version": "V_version",
                      f"TR{short}J_version": "J_version"}
        # remove columns from rename_map that are not in df
        rename_map = {k: v for k, v in rename_map.items() if k in df.columns}

        renamed = df[rename_map.keys()].rename(columns=rename_map)

        # if renamed is empty, skip this chain
        if renamed.shape[0] == 0:
            continue

        # only keep the columns that are in columns
        renamed = renamed[columns]

        if extract_features:
            chain_features = get_baseline_sequence_features(renamed, test, columns)
        else:
            chain_features = renamed

        chain_features = chain_features.add_prefix(f"{chain}_")

        all_chains_features.append(chain_features)

        chain_features_num_rows = chain_features.shape[0]

        assert df_num_rows == chain_features_num_rows, f"Number of rows in df ({df_num_rows}) and chain_features ({chain_features_num_rows}) must be the same"

    ALPHA_OR_BETA = None

    return pd.concat(all_chains_features, axis=1) if len(all_chains_features) > 0 else pd.DataFrame()


def get_features_old_deprecated(df, test=False, cdr_only=False):
    df_num_rows = df.shape[0]

    global ALPHA_OR_BETA

    ALPHA_OR_BETA = 'beta'
    if not cdr_only:
        beta_renamed = df[['CDR3_beta', 'TRBV', 'TRBJ']].rename(columns={'CDR3_beta': 'CDR3', 'TRBV': 'V', 'TRBJ': 'J'})
    else:
        beta_renamed = df[['CDR3_beta']].rename(columns={'CDR3_beta': 'CDR3'})
    try:
        beta_features = get_baseline_sequence_features(beta_renamed, test, cdr_only).add_prefix('beta_')

        beta_features_num_rows = beta_features.shape[0]

        if beta_features_num_rows != df_num_rows:
            raise ValueError(
                f'Number of rows in beta_features ({beta_features_num_rows}, {beta_features.shape[1]}) does not match number of rows in '
                f'df ({df_num_rows}, {df.shape[1]})')

    except Exception as e:
        print(f'Error while extracting beta features: {e}')
        beta_features = pd.DataFrame()

    ALPHA_OR_BETA = 'alpha'
    if not cdr_only:
        alpha_renamed = df[['CDR3_alpha', 'TRAV', 'TRAJ']].rename(
            columns={'CDR3_alpha': 'CDR3', 'TRAV': 'V', 'TRAJ': 'J'})
    else:
        alpha_renamed = df[['CDR3_alpha']].rename(columns={'CDR3_alpha': 'CDR3'})
    try:
        alpha_features = get_baseline_sequence_features(alpha_renamed, test, cdr_only).add_prefix('alpha_')

        alpha_features_num_rows = alpha_features.shape[0]

        # assert df_num_rows == beta_features_num_rows == alpha_features_num_rows, 'Number of rows in dataframes do not match'
    except Exception as e:
        print(f'Error while extracting alpha features: {e}')
        alpha_features = pd.DataFrame()

    ALPHA_OR_BETA = None

    return pd.concat([beta_features, alpha_features], axis=1)


def get_columns_starting_with(df, prefix):
    return df[df.columns[df.columns.str.startswith(prefix)]]


from numpy.random import randint
from sklearn.base import BaseEstimator, TransformerMixin


class MissingIndicator(BaseEstimator, TransformerMixin):
    """Add an attribute that is True if the sample contains missing values, False otherwise."""

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        X["missing"] = X.isnull().any(axis=1)
        return X
