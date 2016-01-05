"""utilities for pwm matrices"""

from utils import normalize, h, inverse_cdf_sample, transpose, make_pssm, prod, argmin, mean, sd
import random
from math import exp, log, sqrt, pi

def sample_matrix(L,sigma):
    """sample Gaussian PWM"""
    return [[random.gauss(0,sigma) for j in range(4)] for i in range(L)]

def Zb_from_matrix(matrix,G):
    """calculate partition function"""
    L = len(matrix)
    Zb_hat = prod(sum(exp(-ep) for ep in col) for col in matrix)/(4**L)
    return G * Zb_hat

def psfm_from_matrix(matrix):
    """calculate IC assuming that matrix has log-odds weights"""
    return [normalize([exp(-ep) for ep in col]) for col in matrix]

def psfm_from_motif(motif,pc=1):
    n = float(len(motif))
    cols = transpose(motif)
    return [[(col.count(b) + pc)/(n+4*pc) for b in "ACGT"] for col in cols]

def pssm_from_motif(motif, pc=1):
    psfm = psfm_from_motif(motif, pc)
    return [[log(f/0.25,2) for f in row] for row in psfm]

def ic_from_matrix(matrix):
    psfm = psfm_from_matrix(matrix)
    return sum(2 - h(col) for col in psfm)

def sample_from_psfm(psfm):
    return "".join([inverse_cdf_sample("ACGT",ps) for ps in psfm])

def site_mu_from_matrix(matrix):
    return sum(map(mean,matrix))

def site_sigma_from_matrix(matrix):
    """return sd of site energies from matrix"""
    return sqrt(sum(map(lambda xs:variance(xs,correct=False),matrix)))

def sigma_from_matrix(matrix):
    """given a GLE matrix, estimate standard deviation of cell weights,
    correcting for bias of sd estimate.  See:
    https://en.wikipedia.org/wiki/Unbiased_estimation_of_standard_deviation
    """
    c = 2*sqrt(2/(3*pi))
    return mean(map(lambda x:sd(x,correct=True),matrix))/c

def matrix_from_motif(seqs,pc=1):
    cols = transpose(seqs)
    N = float(len(seqs))
    raw_mat = [[-log((col.count(b)+pc)/(N+4*pc)) for b in "ACGT"] for col in cols]
    # now normalize each column by the average value
    avg = mean(map(mean,raw_mat))
    return [[x-avg for x in row] for row in raw_mat]

def sample_matrix(L,sigma):
    return [[random.gauss(0,sigma) for j in range(4)] for i in range(L)]

def ringer_motif(matrix,n):
    best_site = "".join(["ACGT"[argmin(col)] for col in matrix])
    best_motif = [best_site]*n
    return best_motif

def predict_ic_from_mean_ep(matrix,ep):
    """given ep, find <IC> of motifs with that ep"""
    pass

def approx_mu(matrix, copies, G=5*10**6):
    Zb = Zb_from_matrix(matrix, G)
    return log(copies) - log(Zb)

def Zb_matrix_lamb(matrix, lamb):
    return prod([sum(exp(-lamb*ep) for ep in row) for row in matrix])

def dZb_matrix_lamb(matrix,lamb):
    print lamb
    return sum([(sum(-ep*exp(-lamb*ep) for ep in row)/
                 sum(exp(-lamb*ep) for ep in row))
                for row in matrix])

def minimize_Zq(matrix):
    f = lambda lamb:dZb_matrix_lamb(matrix,lamb)
    return secant_interval(f,-10,10)
