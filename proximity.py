# Seperate file to analyze the execution time profile and see which functions can be optimized using numba

import sys
sys.path.append('Pforest-dtw')

from trees import ProximityForest
from core import FileReader
from sklearn.model_selection import train_test_split
from util import get_train_dataset, get_test_dataset
from dataStructures import ListDataset
import pandas as pd
import time
from util import readDataframeToListDataset
from util import calculate_tcr_dist_multiple_chains


train_df = get_train_dataset()
# only keep the CDR3 columns
train_df = train_df[['CDR3_alfa', 'CDR3_beta', 'reaction']]
# train_df.dropna(inplace=True) # TODO: remove this, distance function should handle this
train_df = train_df.sample(1000)

train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)

train_dataset = readDataframeToListDataset(train_df, label_col_name='reaction')
val_dataset = readDataframeToListDataset(val_df, label_col_name='reaction')

Pforest = ProximityForest.ProximityForest(1, n_trees=3, n_candidates=5, distance_measure=calculate_tcr_dist_multiple_chains, distance_kwargs={'nan_distance': 5})  #todo: 100  trees instead of 3

Pforest.train(train_dataset)
results = Pforest.test(val_dataset)
print(results.accuracy)