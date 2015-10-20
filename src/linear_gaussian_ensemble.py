import numpy as np
from utils import prod, score_seq, mutate_motif, random_motif, mh, argmin, argmax, prod, random_site, score_seq
from utils import mean, variance
from math import exp,log
import random
from tqdm import *

G = 5 * 10**6

def sample_matrix(L,sigma):
    return [[random.gauss(0,sigma) for j in range(4)] for i in range(L)]

def sample_Zb_terms(L,sigma,trials=10000):
    matrix = sample_matrix(L,sigma)
    return [score_seq(matrix,random_site(L)) for i in xrange(trials)]

def Zb_from_matrix(matrix,G):
    L = len(matrix)
    Zb_hat = prod(sum(exp(-ep) for ep in col) for col in matrix)/(4**L)
    return G * Zb_hat

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
    eps = [score_seq(matrix,site) for site in motif]
    fgs = [exp(-ep) for ep in eps]
    Zb = Zb_from_matrix(matrix,G)
    Z = sum(fgs) + Zb
    return sum(log(fg/Z) for fg in fgs)



def ringer_motif(matrix,n):
    best_site = "".join(["ACGT"[argmin(col)] for col in matrix])
    best_motif = [best_site]*n
    return best_motif

def anti_ringer_motif(matrix,n):
    worst_site = "".join(["ACGT"[argmax(col)] for col in matrix])
    worst_motif = [worst_site]*n
    return worst_motif

def max_fitness(matrix,n,G):
    best_motif = ringer_motif(matrix,n)
    return fitness(matrix,best_motif,G)
