import numpy as np
import pandas as pd
from tcrdist.repertoire import TCRrep


def calculate_tcr_dist2(seq1, seq2, nan_distance=0):
    if (seq1 == seq2).all():
        return 0

    # create a dataframe with the two sequences

    # seq1 and seq2 are two rows in the tDataFrame
    # the columns are cdr3_a_aa, v_a_gene, j_a_gene, cdr3_b_aa, v_b_gene, j_b_gene

    df = pd.DataFrame([seq1, seq2], columns=['cdr3_a_aa', 'v_a_gene', 'j_a_gene', 'cdr3_b_aa', 'v_b_gene', 'j_b_gene'])
    df['count'] = 1

    # contains the df a nan value?
    if df.isnull().values.any():
        return nan_distance

    # create a TCRrep object
    tr = TCRrep(cell_df=df,
                organism='human',
                chains=['alpha', 'beta'],
                db_file='alphabeta_gammadelta_db.tsv'
                )
    # compute the distances
    tr.compute_distances()
    # get the computed distances
    # return tr.pw_cdr3_a_aa[0][1], tr.pw_cdr3_b_aa[0][1] # TODO: normalize the distances
    # if pw_alpha is 1x1, return it's only value (probably 0) # todo: this is the case because the family is not recognized, must be fixed
    if tr.pw_alpha.shape == (1, 1):
        return tr.pw_alpha[0][0]
    return tr.pw_alpha[0][1] + tr.pw_beta[0][1] + tr.pw_cdr3_a_aa[0][1] + tr.pw_cdr3_b_aa[0][1]

from util import get_train_dataset, ProximityTreeClassifier, calculate_tcr_dist_multiple_chains
from sklearn.model_selection import train_test_split

x_columns = ['CDR3_alfa', 'TRAV', 'TRAJ', 'CDR3_beta', 'TRBV', 'TRBJ']
# x_columns = ['CDR3_alfa', 'CDR3_beta']
y_column = 'reaction'

df = get_train_dataset()
df = df[x_columns + [y_column]]
df = df.sample(100)

for col in ['TRAV', 'TRAJ', 'TRBV', 'TRBJ']:
    # only append if not nan
    df[col] = df[col].apply(lambda x: x + '*01' if not pd.isna(x) else x)
    df[col] = df[col].apply(lambda x: x.replace('2DV8', '2/DV8') if not pd.isna(x) else x)
    df[col] = df[col].apply(lambda x: x.replace('14DV4', '14/DV4') if not pd.isna(x) else x)
    df[col] = df[col].apply(lambda x: x.replace('23DV6', '23/DV6') if not pd.isna(x) else x)

train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# create numpy arrays for the train and test data
train_X = train_df[x_columns].to_numpy()
train_y = train_df[y_column].to_numpy()

val_X = val_df[x_columns].to_numpy()
val_y = val_df[y_column].to_numpy()
#%%
from util import ProximityForestClassifier
from sklearn.metrics import accuracy_score

model = ProximityForestClassifier(reduce_features=False, distance_measure=calculate_tcr_dist2,
                                  distance_kwargs={"nan_distance": 0}, multithreaded=True, n_trees=20)
model.fit(train_X, train_y)

predictions = model.predict(val_X)

accuracy_score(val_y, predictions)
#%%
from sklearn import metrics
from util import plot_roc_curve

predictions = model.predict_proba(val_X)
fpr, tpr, thresholds = metrics.roc_curve(val_y, predictions[:, 1], pos_label=1)
roc_auc = metrics.auc(fpr, tpr)
plot_roc_curve(fpr, tpr, label=f'Random Forest (AUC = {roc_auc:.3f})', title="GILGFVFTL")
print(f"ROC AUC: {roc_auc}")