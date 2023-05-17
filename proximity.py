from util import get_train_dataset, ProximityTreeClassifier, calculate_tcr_dist_multiple_chains
from sklearn.model_selection import train_test_split
from util import ProximityForestClassifier, calculate_tcr_dist2, calculate_tcr_dist2_cached
from sklearn.metrics import accuracy_score
from sklearn import metrics
from util import plot_roc_curve


x_columns = ['CDR3_alfa', 'TRAV', 'TRAJ', 'CDR3_beta', 'TRBV', 'TRBJ']
# x_columns = ['CDR3_alfa', 'CDR3_beta']
y_column = 'reaction'

df = get_train_dataset(vdjdb=True)
df = df[x_columns + [y_column]]
df = df.sample(100) # (of the 12000)
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# create numpy arrays for the train and test data
train_X = train_df[x_columns].to_numpy()
train_y = train_df[y_column].to_numpy()

val_X = val_df[x_columns].to_numpy()
val_y = val_df[y_column].to_numpy()


model = ProximityForestClassifier(reduce_features=False, distance_measure=calculate_tcr_dist2,
                                  distance_kwargs={"nan_distance": 0}, multithreaded=False)
model.fit(train_X, train_y)


predictions = model.predict(val_X)

print(accuracy_score(val_y, predictions))


predictions = model.predict_proba(val_X)
fpr, tpr, thresholds = metrics.roc_curve(val_y, predictions[:, 1], pos_label=1)
roc_auc = metrics.auc(fpr, tpr)
plot_roc_curve(fpr, tpr, label=f'Random Forest (AUC = {roc_auc:.3f})', title="GILGFVFTL")
print(f"ROC AUC: {roc_auc}")