from collections import defaultdict
from functools import cache  # TODO: from methodtools import lru_cache? Should work with unhashable types

import pwseqdist as pwsd
import numpy as np
import numba
import numpy as np
import pandas as pd
from tcrdist.repertoire import TCRrep

### Implementation 1

def calculate_tcr_dist_no_nan(seq1, seq2):
    drect_dots2: np.ndarray = pwsd.apply_pairwise_rect(pwsd.metrics.nb_vector_tcrdist, seqs1=[seq1], seqs2=[seq2], use_numba=True,
                                           ncpus=1, uniqify=False, ntrim=0, ctrim=0) # ntrim and ctrim to 0 since we're not always working with cdr's (see nb_vector_tcrdist documentation)
    return drect_dots2[0][0]


def calculate_tcr_dist(seq1, seq2, nan_distance):
    # if one of the two is NaN, return nan_distance
    if (isinstance(seq1, float) and np.isnan(seq1)) or \
            (isinstance(seq2, float) and np.isnan(seq2)):
        return nan_distance
    else:
        return calculate_tcr_dist_no_nan(seq1, seq2)

def calculate_tcr_dist_multiple_chains(seq1, seq2, nan_distance=5, print_dist=False):
    # seq1 is e.g. ['ABCD', 'EFGH']: the alpha and beta chains
    assert len(seq1) == len(seq2)
    dist = 0
    for i in range(len(seq1)):
        dist += calculate_tcr_dist(seq1[i], seq2[i], nan_distance=nan_distance) # TODO: weight?
    if print_dist: print(f"Dist between {seq1} and {seq2} is {dist}")
    return dist

### Implementation 2
def calculate_tcr_dist2(seq1, seq2, nan_distance=0, organism='human'):
    if (seq1 == seq2).all():
        return 0

    df = pd.DataFrame([seq1, seq2], columns=['cdr3_a_aa', 'v_a_gene', 'j_a_gene', 'cdr3_b_aa', 'v_b_gene', 'j_b_gene'])
    df['count'] = 1

    # contains the df a nan value?
    if df.isnull().values.any():
        return nan_distance

    # create a TCRrep object
    tr = TCRrep(cell_df=df,
                organism=organism,
                chains=['alpha', 'beta'],
                db_file='alphabeta_gammadelta_db.tsv'
                )

    tr.compute_distances()

    if tr.pw_alpha.shape == (1, 1):
        print("Warning: shape 1x1")
        return tr.pw_alpha[0][0]

    distance_sum = tr.pw_alpha[0][1] + tr.pw_beta[0][1] # + tr.pw_cdr3_a_aa[0][1] + tr.pw_cdr3_b_aa[0][1] # cdr distances are already in pw
    # print(f"Distance between {seq1} and {seq2} is {distance_sum}")
    return distance_sum


# @cache # doesn't work, since numpy arrays are not hashable
CACHE_DICT = {}
CACHE_COUNTER = defaultdict(int)
def calculate_tcr_dist2_cached(seq1, seq2, nan_distance=0, organism='human'):
    id = str(seq1) + str(seq2) # might slow down a lot
    if id in CACHE_DICT:
        CACHE_COUNTER[id] += 1
        # print(f"Cache hit for {id} (hit {CACHE_COUNTER[id]} times)")
        return CACHE_DICT[id]
    else:
        res = calculate_tcr_dist2(seq1, seq2, nan_distance=nan_distance, organism=organism)
        CACHE_DICT[id] = res
        return res

def get_cache_counter():
    return CACHE_COUNTER
