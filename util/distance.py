import pwseqdist as pwsd
import numpy as np
import numba

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
def calculate_tcr_dist_2(seq1, seq2):
     pass