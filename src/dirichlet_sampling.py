from utils import simplex_sample, h, inverse_cdf_sample, transpose, qqplot, motif_ic, mean
import random
from formosa import maxent_motifs
from matplotlib import pyplot as plt
from tqdm import *
from math import exp, log
import numpy as np

def rvector(beta, K=4):
    #trials = 0
    while True:
        #trials += 1
        p = simplex_sample(K)
        if random.random() < exp(-beta*h(p)):
            #print "acceptance rate:",1/float(trials)
            return p

def rcol(beta, N):
    ps = rvector(beta, K=4)
    col = [inverse_cdf_sample("ACGT",ps) for _ in xrange(N)]
    return col

def rmotif(N, L, beta):
    cols = [rcol(beta, N) for _ in xrange(L)]
    return ["".join(row) for row in transpose(cols)]
    
def find_beta(N, L, des_ic, iterations=100, verbose=False):
    #des_ic_per_col = des_ic / float(L)
    beta = 1
    alpha = 1.0
    beta_hist = []
    ic_hist = []
    for i in range(1,iterations+1):
        #beta = beta + alpha/i * (des_ic_per_col - motif_ic(rmotif(N, 1, beta)))
        obs_ic = motif_ic(rmotif(N, L, beta))
        beta_hist.append(beta)
        ic_hist.append(obs_ic)
        beta = beta + alpha/i * (des_ic - obs_ic)
        if verbose:
            print i, beta, obs_ic
    return beta, beta_hist, ic_hist

def find_beta2(N, L, des_ic, iterations=100, verbose=False):
    #des_ic_per_col = des_ic / float(L)
    beta = 1
    alpha = 1.0
    beta_hist = []
    def f(beta):
        #return motif_ic(rmotif(N, 1, beta)) - des_ic_per_col
        return motif_ic(rmotif(N, L, beta)) - des_ic
    for i in range(1,iterations+1):
        beta = beta + alpha/floati * (f(beta + 1) - f(beta - 1))
        beta_hist.append(beta)
        if verbose:
            print i, beta
    return beta, beta_hist

def find_beta3(N, L, des_ic, iterations=100):
    K = 1/(2*log(2)*N)
    def f(beta):
        return (2 - (h(rvector(beta)) + K))*L
    alpha = 1.0
    beta = 1
    beta_hist = []
    ic_hist = []
    for i in range(1, iterations):
        obs_ic = f(beta)
        beta_hist.append(beta)
        ic_hist.append(obs_ic)
        beta = beta + alpha/i * (des_ic - obs_ic)
    return beta, beta_hist, ic_hist
    

def comparison():
    N = 100
    L = 10
    des_ic = 10
    beta, beta_hist, ic_hist = find_beta3(N, L, des_ic, iterations=10000)
    dir_motifs = [rmotif(N, L, beta) for _ in trange(1000)]
    max_motifs = maxent_motifs(N, L, des_ic, num_motifs=1000)
    dir_ics = map(motif_ic, dir_motifs)
    max_ics = map(motif_ic, max_motifs)
    print "dir mean:", mean(dir_ics)
    print "max mean:", mean(max_ics)
    plt.subplot(1,2,1)
    plt.hist(dir_ics,bins=np.linspace(0,20,50), alpha=0.5, color='b')
    plt.hist(max_ics,bins=np.linspace(0,20,50), alpha=0.5, color='r')
    plt.subplot(1,2,2)
    qqplot(max_ics, dir_ics)
    plt.show()

def entropy_from_ps(ps, N):
    K = len(ps)
    ns = [0] * K
    xs = range(K)
    for i in xrange(N):
        j = inverse_cdf_sample(xs, ps)
        ns[j] += 1
    return h([n/float(N) for n in ns])

def cross_entropy(ps, qs):
    return -sum(p*log(q) for (p,q) in zip(ps,qs))
