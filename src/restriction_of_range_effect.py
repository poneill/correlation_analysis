"""The purpose of this script is to explore the hypothesis that
correlations are due to the restriction of range effect.  To test this
hypothesis we sample GLE matrices and select for the binding energy to
exceed a certain threshold, then measure MI."""

import numpy as np
from linear_gaussian_ensemble import *
from gle_gini_analysis import sample_motif_neglect_fg, naive_spoof
from utils import sample_until, score_seq, total_motif_mi

def ror_experiment():
    L = 10
    n = 100
    sigmas = np.linspace(0.1,10,10)
    alphas = np.linspace(0,1,10)
    for sigma in sigmas:
        for alpha in alphas:
            theta = - alpha * sigma * L
            matrix = sample_matrix(L,sigma)
            sampler = lambda : sample_motif_neglect_fg(matrix,1,Ne=2)[0]
            motif = sample_until(lambda site:score_seq(matrix,site) < theta,sampler,n)
            print sigma, alpha, total_motif_mi(motif)
        
def threshold(matrix,motif):
    eps = [score_seq(matrix,site) for site in motif]
    return max(eps)
