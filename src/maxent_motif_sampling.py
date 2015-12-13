"""This script contains functions for sampling motifs from the maxent
distribution on entropy.

"""
# Maxent sampling methods
from utils import bisect_interval, h, fac, prod, normalize, inverse_cdf_sample, inverse_cdf_sampler, log_fac
from utils import permute, concat, transpose
from math import exp, log
from collections import Counter
from tqdm import *
import random
import numpy as np

def maxent_motif_with_ic_ref(n,L,desired_ic,tolerance=10**-2,beta=None):
    """sample motif from max ent distribution with mean desired_ic"""
    # first we adjust the desired ic upwards so that when motif_ic is
    # called with 1st order correction, we get the desired ic.
    if beta is None:
        correction_per_col = 3/(2*log(2)*n)
        desired_ic += L * correction_per_col
        beta = find_beta_for_mean_motif_ic(n,L,desired_ic,tolerance=tolerance)
    ps = count_ps_from_beta(n,beta)
    counts = [inverse_cdf_sample(enumerate_counts(n),ps) for i in range(L)]
    cols = [sample_col_from_count(count) for count in counts]
    return map(lambda site:"".join(site),transpose(cols))

def maxent_motif_with_ic(n,L,desired_ic,tolerance=10**-2,beta=None):
    """sample motif from max ent distribution with mean desired_ic"""
    # first we adjust the desired ic upwards so that when motif_ic is
    # called with 1st order correction, we get the desired ic.
    if beta is None:
        correction_per_col = 3/(2*log(2)*n)
        desired_ic += L * correction_per_col
        beta = find_beta_for_mean_motif_ic(n,L,desired_ic,tolerance=tolerance)
    ps = count_ps_from_beta(n,beta)
    count_sampler = inverse_cdf_sampler(enumerate_counts(n),ps)
    counts = [count_sampler() for i in range(L)]
    cols = [sample_col_from_count(count) for count in counts]
    return map(lambda site:"".join(site),transpose(cols))

def maxent_motifs_with_ic_ref(n,L,desired_ic,num_motifs,tolerance=10**-2):
    """sample motif from max ent distribution with mean desired_ic"""
    # first we adjust the desired ic upwards so that when motif_ic is
    # called with 1st order correction, we get the desired ic.
    correction_per_col = 3/(2*log(2)*n)
    desired_ic += L * correction_per_col
    beta = find_beta_for_mean_motif_ic(n,L,desired_ic,tolerance=tolerance)
    return [maxent_motif_with_ic(n,L,desired_ic,tolerance=tolerance,beta=beta) for i in trange(num_motifs)]

def maxent_motifs_with_ic(n, L, desired_ic, num_motifs, tolerance=10**-2,beta=None):
    if beta is None:
        correction_per_col = 3/(2*log(2)*n)
        desired_ic += L * correction_per_col
        beta = find_beta_for_mean_motif_ic(n,L,desired_ic,tolerance=tolerance)
    ps = count_ps_from_beta(n,beta)
    count_sampler = inverse_cdf_sampler(enumerate_counts(n),ps)
    def sample():
        counts = [count_sampler() for i in range(L)]
        cols = [sample_col_from_count(count) for count in counts]
        return map(lambda site:"".join(site),transpose(cols))
    return [sample() for _ in trange(num_motifs)]

def find_beta_for_mean_col_ic_ref(n,desired_ic_per_col,tolerance=10**-2):
    ic_from_beta = lambda beta:2-mean_col_ent(n,beta)
    f = lambda beta:ic_from_beta(beta) - desired_ic_per_col
    #print "finding beta to tol:",tolerance
    ub = 1000 # hackish, upped in order to deal with CRP
    return bisect_interval(f,-10,ub,verbose=False,tolerance=tolerance)

def find_beta_for_mean_col_ic(n, desired_ic_per_col,tolerance=10**-2):
    """find beta such that entropy*exp(-beta*entropy)/Z = des_ent"""
    counts = enumerate_counts(n)
    entropies = np.array(map(entropy_from_counts, counts))
    cols = np.array(map(counts_to_cols, counts))
    def f(beta):
        phats = cols*(np.exp(-beta*entropies))
        return 2 - entropies.dot(phats)/np.sum(phats) - desired_ic_per_col
    ub = 1000
    return bisect_interval(f,-10,ub,verbose=False,tolerance=tolerance)

def find_beta_for_mean_motif_ic(n,L,desired_ic,tolerance=10**-2):
    desired_ic_per_col = desired_ic/L
    return find_beta_for_mean_col_ic(n,desired_ic_per_col,tolerance)

def count_ps_from_beta(n,beta):
    ws = [counts_to_cols(count)*exp(-beta*entropy_from_counts(count)) for count in enumerate_counts(n)]
    return normalize(ws)

def counts_to_cols(counts):
    """return numer of cols associated given counts"""
    N = sum(counts)
    # all_cols = 4**N
    metacounts = Counter(counts)
    counts_to_bases = fac(4)/prod(fac(multiplicity) for multiplicity in metacounts.values())
    if N <= 170:
        bases_to_pos = fac(N)/prod(fac(count) for count in counts)
    else:
        #print "Warning: possible numerical issues in counts_to_cols" 
        bases_to_pos = round(exp(log_fac(N) - sum(log_fac(count) for count in counts)))
    return counts_to_bases * bases_to_pos

def enumerate_counts(N):
    return list(partitionfunc(N,4,l=0))

def partitionfunc(n,k,l=1):
    '''n is the integer to partition, k is the length of partitions, l is the min partition element size'''
    if k < 1:
        raise StopIteration
    if k == 1:
        if n >= l:
            yield (n,)
        raise StopIteration
    for i in range(l,n+1):
        for result in partitionfunc(n-i,k-1,i):
            yield (i,)+result

def mean_col_ent(n,beta):
    """compute mean ic of maxent distribution p(col) ~ exp(-beta*H(col))"""
    counts = enumerate_counts(n)
    Z = 0
    expectation = 0
    for count in counts:
        entropy = entropy_from_counts(count)
        w = counts_to_cols(count)*exp(-beta*entropy)
        expectation += entropy*w
        Z += w
    return expectation/Z

def entropy_from_counts(counts):
    N = float(sum(counts))
    ps = [c/N for c in counts]
    return h(ps)

def sample_col_from_count_ref(count):
    return permute(concat([[base]*n for base,n in zip(permute("ACGT"),count)]))        

def sample_col_from_count(count):
    col = concat([[base]*n for base,n in zip(permute("ACGT"),count)])
    random.shuffle(col)
    return col
