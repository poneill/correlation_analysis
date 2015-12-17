import numpy as np
from utils import prod, score_seq, mutate_motif, random_motif, mh, argmin, argmax, prod, random_site, score_seq
from utils import mean, variance, choose2
from math import exp,log
from pwm_utils import ringer_motif
import random
from tqdm import *

G = 5 * 10**6

def sample_Zb_terms(L,sigma,trials=10000):
    matrix = sample_matrix(L,sigma)
    return [score_seq(matrix,random_site(L)) for i in xrange(trials)]

def sample_Zb(L,sigma,G):
    return Zb_from_matrix(sample_matrix(L,sigma),G)

def predict_Zb_stats(L,sigma,G):
    """return mean, variances of Zb"""
    return G*exp(L*sigma**2/2.0), G**2*((exp(sigma**2)*(4*exp(sigma**2) + 12)/16)**L - exp(L*sigma**2/2.0))
    
def Zb_from_matrix_ref(matrix,G):
    L = len(matrix)
    eps = np.array([score_seq(matrix,random_site(L)) for i in trange(G)])
    return np.sum(np.exp(-eps))
    
def fitness_additive(matrix,motif,G):
    eps = [score_seq(matrix,site) for site in motif]
    fg = sum(exp(-ep) for ep in eps)
    Zb = Zb_from_matrix(matrix,G)
    return fg/(fg + Zb)

def fitness(matrix,motif,G):
    """multiplicative fitness of occupancy over all sites"""
    eps = [score_seq(matrix,site) for site in motif]
    fgs = [exp(-ep) for ep in eps]
    Zb = Zb_from_matrix(matrix,G)
    Z = sum(fgs) + Zb
    return prod(fg/Z for fg in fgs)

def log_fitness(matrix,motif,G):
    """multiplicative fitness of occupancy over all sites"""
    n = len(motif)
    eps = [score_seq(matrix,site) for site in motif]
    fgs = [exp(-ep) for ep in eps]
    Zf = sum(fgs)
    Zb = Zb_from_matrix(matrix,G)
    Z = Zf + Zb
    return -sum(eps) - n*log(Z)

def log_fitness_approx3(matrix,motif,G):
    n = len(motif)
    eps = [score_seq(matrix,site) for site in motif]
    fgs = [exp(-ep) for ep in eps]
    Zf = sum(fgs)
    Zb = Zb_from_matrix(matrix,G)
    Z = Zf + Zb
    print Zf,Zb,Zf/Zb
    good_approximation = -sum(eps) - n*(log(Zf))
    Zf_hat = mean(fgs)
    Zf_resids = [fg - Zf_hat for fg in fgs]
    worse_approximation = -sum(eps) - n*(log(n) + log(Zf_hat))
    print good_approximation, worse_approximation
    return good_approximation
    
# Do we know that higher order terms even matter?  Seems to agree well
# with log_fitness_approx2!  For LexA, first and second order terms are six OOM greater than zeroth!
def log_fitness_approx(matrix,motif,G,terms=2):
    n = len(motif)
    eps = [score_seq(matrix,site) for site in motif]
    fgs = [exp(-ep) for ep in eps]
    Zf = sum(fgs)
    Zb = Zb_from_matrix(matrix,G)
    Z = Zf + Zb
    zeroth_term = log(n+Zb) * (terms >= 0)
    first_term = (-1/(n+Zb)*sum(eps)) * (terms >= 1)
    second_term = 1/2.0*1/(n+Zb)**2*((n + Zb - 1)*sum(ep**2 for ep in eps) -
                                     sum(epi*epj for epi,epj in choose2(eps))) * (terms >= 2)
    print zeroth_term,first_term,second_term
    # first_order = -sum(eps) - n*(log(n+Zb) + (-1/(n+Zb)*sum(eps)))
    # second_order = -sum(eps) - n*(log(n+Zb) + (-1/(n+Zb)*sum(eps)) + 1/2.0*1/(n+Zb)**2*((n)))
    return -sum(eps) - n*(zeroth_term + first_term + second_term)

def log_fitness_approx2(matrix,motif,G):
    """approximate fitness by neglecting competition from other functional sites, i.e. Zb"""
    eps = [score_seq(matrix,site) for site in motif]
    Zb = Zb_from_matrix(matrix,G)
    return -sum(eps) - n*log(Zb)
    
def Z_approx(matrix,n,Ne,G=5*10**6):
    """use log fitness approximation to compute partition function"""
    nu = Ne - 1
    sigma_sq = sum(map(lambda xs:variance(xs,correct=False),matrix))
    Zb = Zb_from_matrix(matrix,G)
    
def anti_ringer_motif(matrix,n):
    worst_site = "".join(["ACGT"[argmax(col)] for col in matrix])
    worst_motif = [worst_site]*n
    return worst_motif

def max_fitness(matrix,n,G):
    best_motif = ringer_motif(matrix,n)
    return fitness(matrix,best_motif,G)
