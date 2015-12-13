"""This script contains functions for sampling motifs from the uniform
distribution wrt IC.  The two principal algorithms are:

- Uniform via IMH w/ maxent proposal
- Uniform MH w/ random walk proposal
"""

from maxent_motif_sampling import maxent_motif_with_ic, find_beta_for_mean_motif_ic, count_ps_from_beta
from maxent_motif_sampling import enumerate_counts, sample_col_from_count
from utils import mutate_motif_p, inverse_cdf_sample, inverse_cdf_sampler
from utils import transpose, motif_ic, mh, sample_until, gelman_rubin, mmap
from math import log, exp, ceil
import random
from tqdm import *

def uniform_motif_with_ic_imh_ref(n,L,desired_ic,epsilon=0.1,iterations=None,verbose=False,num_chains=8):
    correction_per_col = 3/(2*log(2)*n)
    desired_ic_for_beta = desired_ic + L * correction_per_col
    beta = find_beta_for_mean_motif_ic(n,L,desired_ic_for_beta)
    ps = count_ps_from_beta(n,beta)
    count_sampler = inverse_cdf_sampler(enumerate_counts(n),ps)
    def Q(motif):
        counts = [count_sampler() for i in range(L)]
        cols = [sample_col_from_count(count) for count in counts]
        motif_p = map(lambda site:"".join(site),transpose(cols))
        return motif_p
    def log_dQ(motif_p, motif):
        return (beta*motif_ic(motif_p))
    def log_f(motif):
        in_range = abs(motif_ic(motif)-desired_ic) < epsilon
        return 0 if in_range else -10.0**100
    if iterations:
        x0 = sample_until(lambda x:log_f(x) > -1,lambda: Q(None),1)[0]
        chain = mh(log_f,proposal=Q,dprop=log_dQ,x0=x0,iterations=iterations,use_log=True,verbose=False)
        return chain
    else: #use gelman rubin criterion
        x0s = sample_until(lambda x:log_f(x) > -1,lambda: Q(None),num_chains)
        iterations = 100
        converged = False
        chains = [[] for _ in range(num_chains)]
        while not converged:
            for chain,x0 in zip(chains,x0s):
                chain.extend(mh(log_f,proposal=Q,dprop=log_dQ,x0=x0,
                                 iterations=iterations,use_log=True,verbose=False))
            ic_chains = mmap(motif_ic,chains)
            R_hat, neff = gelman_rubin(ic_chains)
            if R_hat < 1.1:
                return chains
            else:
                x0s = [chain[-1] for chain in chains]
                iterations *= 2

def uniform_motif_with_ic_imh(n,L,desired_ic,epsilon=0.1,iterations=None,verbose=False,beta=None,num_chains=8):
    if beta is None:
        correction_per_col = 3/(2*log(2)*n)
        desired_ic_for_beta = desired_ic + L * correction_per_col
        beta = find_beta_for_mean_motif_ic(n,L,desired_ic_for_beta)
    ps = count_ps_from_beta(n,beta)
    count_sampler = inverse_cdf_sampler(enumerate_counts(n),ps)
    def Q(motif):
        counts = [count_sampler() for i in range(L)]
        cols = [sample_col_from_count(count) for count in counts]
        motif_p = map(lambda site:"".join(site),transpose(cols))
        return motif_p
    def log_dQ(motif_p, motif):
        return (beta*motif_ic(motif_p))
    def log_f(motif):
        in_range = abs(motif_ic(motif)-desired_ic) < epsilon
        return 0 if in_range else -10.0**100
    
    x0 = sample_until(lambda x:log_f(x) > -1,lambda: Q(None),1)[0]
    # first, determine probability of landing in range
    ar = 0
    iterations = 100
    while ar == 0:
        ar = mh(log_f,proposal=Q,dprop=log_dQ,x0=x0,iterations=iterations,use_log=True,verbose=False,return_ar=True)
        iterations *= 2
    iterations = int(1.0/ar * 10)
    chain = mh(log_f,proposal=Q,dprop=log_dQ,x0=x0,iterations=iterations,use_log=True,verbose=False)
    return chain

def uniform_motif_imh_tv(n,L,desired_ic,beta=None,epsilon=None,tv=0.01):
    """run uniform imh to within total variation bound tv"""
    correction_per_col = 3/(2*log(2)*n)
    desired_ic_for_beta = desired_ic + L * correction_per_col
    if beta == None:
        beta = find_beta_for_mean_motif_ic(n,L,desired_ic_for_beta)
    if epsilon == None:
        epsilon = 1.0/(2*beta)
        print "maximally efficient epsilon:",epsilon
    ps = count_ps_from_beta(n,beta)
    count_sampler = inverse_cdf_sampler(enumerate_counts(n),ps)
    def Qp(motif):
        counts = [count_sampler() for i in range(L)]
        cols = [sample_col_from_count(count) for count in counts]
        motif_p = map(lambda site:"".join(site),transpose(cols))
        return motif_p
    def Q(motif):
        return sample_until(lambda m:abs(motif_ic(m) - desired_ic) < epsilon,lambda:Qp(None),1)[0]
    def log_dQ(motif_p, motif):
        return (beta*motif_ic(motif_p))
    def log_f(motif):
        in_range = abs(motif_ic(motif)-desired_ic) < epsilon
        return 0 if in_range else -10.0**100
    alpha = exp(-2*beta*epsilon)
    iterations = int(ceil(log(tv)/log(1-alpha)))
    print "iterations:",iterations
    x0 = sample_until(lambda x:log_f(x) > -1,lambda: Q(None),1)[0]
    # first, determine probability of landing in range
    chain = mh(log_f,proposal=Q,dprop=log_dQ,x0=x0,iterations=iterations,use_log=True,verbose=False)
    return chain

def uniform_motifs_imh_tv(n,L,desired_ic,num_motifs,beta=None,epsilon=None,tv=0.01):
    correction_per_col = 3/(2*log(2)*n)
    desired_ic_for_beta = desired_ic + L * correction_per_col
    if beta == None:
        beta = find_beta_for_mean_motif_ic(n,L,desired_ic_for_beta)
    if epsilon == None:
        epsilon = 1.0/(2*beta)
        print "maximally efficient epsilon:",epsilon
    return [uniform_motif_imh_tv(n,L,desired_ic,beta=beta,epsilon=epsilon,tv=tv) for i in trange(num_motifs)]
    
def uniform_motifs_with_ic_rw_harmonic(n, L, desired_ic, num_motifs, epsilon=0.1):
    correction_per_col = 3/(2*log(2)*n)
    desired_ic_for_beta = desired_ic + L * correction_per_col
    beta = find_beta_for_mean_motif_ic(n,L,desired_ic_for_beta)
    return [uniform_motif_with_ic_rw(n,L, desired_ic,beta=beta,epsilon=epsilon,iterations="harmonic")[-1]
            for _ in range(num_motifs)]
    
def uniform_motif_with_ic_rw(n,L,desired_ic,epsilon=0.1,p=None,iterations=None,num_chains=8,x0=None,beta=None):
    if p is None:
        p = 2.0/(n*L)
    def Q(motif):
        return mutate_motif_p(motif,p)
    def f(motif):
        return abs(motif_ic(motif)-desired_ic) < epsilon
    if type(iterations) is int:
        if x0 is None:
            x0 = uniform_motif_with_ic_imh(n,L,desired_ic,epsilon=epsilon,iterations=1,beta=beta)[0]
        chain = mh(f,proposal=Q,x0=x0,iterations=iterations)
        return chain
    elif iterations == "harmonic":
        ar = 1.0/5
        iterations = int(n*L*harmonic(n*L)/ar)
        print "iterations:",iterations
        if x0 is None:
            x0 = uniform_motif_with_ic_imh(n,L,desired_ic,epsilon=epsilon,iterations=1)[0]
        chain = mh(f,proposal=Q,x0=x0,iterations=iterations)
        return chain
    else: #use gelman rubin criterion
        x0s = [uniform_motif_with_ic_imh(n,L,desired_ic,epsilon=epsilon,iterations=1)[0]
              for i in range(num_chains)]
        iterations = 100
        converged = False
        chains = [[] for _ in range(num_chains)]
        while not converged:
            for chain,x0 in zip(chains,x0s):
                chain.extend(mh(f,proposal=Q,x0=x0,
                                 iterations=iterations,verbose=False))
            ic_chains = mmap(motif_ic,chains)
            R_hat, neff = gelman_rubin(ic_chains)
            if R_hat < 1.1:
                return chains
            else:
                x0s = [chain[-1] for chain in chains]
                iterations *= 2

def uniform_motif_accept_reject(n,L,desired_ic,epsilon=0.1,beta=None,ps=None,count_sampler=None):
    correction_per_col = 3/(2*log(2)*n)
    desired_ic_for_beta = desired_ic + L * correction_per_col
    if beta is None:
        beta = find_beta_for_mean_motif_ic(n,L,desired_ic_for_beta)
    if ps is None:
        ps = count_ps_from_beta(n,beta)
    if count_sampler is None:
        count_sampler = inverse_cdf_sampler(enumerate_counts(n),ps)
    def rQ_raw():
        counts = [count_sampler() for i in range(L)]
        cols = [sample_col_from_count(count) for count in counts]
        motif_p = map(lambda site:"".join(site),transpose(cols))
        return motif_p
    def rQ():
        return sample_until(lambda M:inrange(M,desired_ic,epsilon),rQ_raw,1,progress_bar=False)[0]
    def dQhat(motif):
        return exp(beta*motif_ic(motif))
    Imin = desired_ic - epsilon
    Imax = desired_ic + epsilon
    log_M = -beta*Imin
    def dQ(motif):
        return exp(beta*motif_ic(motif) + log_M)
    def AR(motif):
        return 1.0/dQ(motif)
    M = exp(-beta*(desired_ic - epsilon)) # which ic? +/- correction
    trials = 0
    while True:
        trials +=1
        motif = rQ()
        r = random.random()
        if r < AR(motif):
            return motif

def uniform_motifs_accept_reject(n,L,desired_ic,num_motifs,epsilon=0.1,beta=None):
    if beta is None:
        correction_per_col = 3/(2*log(2)*n)
        desired_ic_for_beta = desired_ic + L * correction_per_col
        beta = find_beta_for_mean_motif_ic(n,L,desired_ic_for_beta)
    ps = count_ps_from_beta(n,beta)
    count_sampler = inverse_cdf_sampler(enumerate_counts(n),ps)
    return [uniform_motif_accept_reject(n,L,desired_ic,epsilon=epsilon,beta=beta,
                                        ps=ps,count_sampler=count_sampler) for i in trange(num_motifs)]
    
def inrange(M,I,epsilon):
    return abs(motif_ic(M)-I) < epsilon

def ar_validation_plot(n,L,desired_ic,epsilon,filename=None):
    beta = find_beta_for_mean_motif_ic(n,L,desired_ic_for_beta)
    ar_motifs = [uniform_motif_accept_reject(n,L,desired_ic,beta=beta) for _ in tqdm(chain)]
    chain = uniform_motif_imh_tv(n,L,desired_ic,beta=beta,epsilon=0.1)
    qqplot(map(motif_ic, ar_motifs),map(motif_ic, chain))
    plt.xlabel("Rejection Sampling IC (bits)")
    plt.ylabel("IMH Sampling IC (bits)")
    plt.title("Quantile-Quantile Plot for Rejection vs IMH Sampling")
    maybesave(filename)

def harmonic(n):
    return sum(1.0/i for i in range(1,n+1))
