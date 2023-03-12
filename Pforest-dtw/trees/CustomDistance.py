import pwseqdist as pwsd

def calculate_tcr_dist(seq1, seq2):
    drect_dots2 = pwsd.apply_pairwise_rect(pwsd.metrics.nb_vector_tcrdist, seqs1=[seq1], seqs2=[seq2], use_numba=True, ncpus=1, uniqify=False)
    return drect_dots2[0][0]

def calculate_tcr_dist_multiple_chains(seq1, seq2):
    assert len(seq1) == len(seq2)
    dist = 0
    for i in range(len(seq1)):
        dist += calculate_tcr_dist(seq1[i], seq2[i]) # TODO: weight?
    return dist
