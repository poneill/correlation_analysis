"""utilities for pwm matrices"""

from utils import normalize, h, inverse_cdf_sample, transpose, make_pssm, prod, argmin, mean, sd
from utils import variance, score_seq
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

def psfm_from_matrix(matrix, lamb=1):
    """calculate IC assuming that matrix has log-odds weights"""
    return [normalize([exp(-lamb*ep) for ep in col]) for col in matrix]

def psfm_from_motif(motif,pc=1):
    N = float(len(motif))
    cols = transpose(motif)
    return [[(col.count(b) + pc)/(N+4*pc) for b in "ACGT"] for col in cols]

def self_scores(motif, pc=1):
    pssm = pssm_from_motif(motif, pc=pc)
    return [score_seq(pssm, seq) for seq in motif]

def pssm_from_motif(motif, pc=1):
    psfm = psfm_from_motif(motif, pc)
    return [[log(f/0.25,2) for f in row] for row in psfm]

def ic_from_matrix(matrix):
    psfm = psfm_from_matrix(matrix)
    return sum(2 - h(col) for col in psfm)

def sample_from_psfm(psfm):
    return "".join([inverse_cdf_sample("ACGT",ps) for ps in psfm])

def spoof_psfm(motif, pc=1):
    psfm = psfm_from_motif(motif, pc=pc)
    return [sample_from_psfm(psfm) for _ in motif]
    
def site_mu_from_matrix(matrix):
    return sum(map(mean,matrix))

def site_sigma_from_matrix(matrix,correct=False):
    """return sd of site energies from matrix""" # agrees with estimate_site_sigma
    return sqrt(sum(map(lambda xs:variance(xs,correct=correct),matrix)))

def estimate_site_sigma_from_matrix(matrix,n=1000):
    L = len(matrix)
    return sd([score_seq(matrix,random_site(L)) for i in xrange(n)])

def site_params_from_matrix(matrix):
    return site_mu_from_matrix(matrix),site_sigma_from_matrix(matrix)
    
def sigma_from_matrix(matrix):
    """given a GLE matrix, estimate standard deviation of cell weights,
    correcting for bias of sd estimate.  See:
    https://en.wikipedia.org/wiki/Unbiased_estimation_of_standard_deviation
    """
    c = 2*sqrt(2/(3*pi))
    return mean(map(lambda x:sd(x,correct=True),matrix))/c

def site_sigma_from_sigma(sigma, L=10):
    """given GLE param sigma, find expected site sigma"""
    return sqrt(3/4.0*L)*sigma

def estimate_site_sigma_from_sigma(sigma,L=10,n=1000):
    return mean(site_sigma_from_matrix(sample_matrix(L,sigma)) for _ in trange(n))

def test_site_sigma_from_sigma(sigma,L=10,n=1000):
    return site_sigma_from_sigma(sigma,L),estimate_site_sigma_from_sigma(sigma, L, n)

    
def on_off_matrix_with_sigma(L,sigma):
    """given desired sigma, return on-off matrix with matching sigma"""
    #ep = 4*sqrt(sigma**2/(3.0*L))
    ep = sigma*(4/sqrt(3))
    mean_ep = ep/4
    return [[-ep + mean_ep,mean_ep,mean_ep,mean_ep] for i in range(L)]

def on_off_matrix_with_site_sigma(L,site_sigma):
    """given desired sigma, return on-off matrix with matching sigma"""
    #ep = 4*sqrt(sigma**2/(3.0*L))
    ep = sigma*(4/sqrt(3))
    mean_ep = ep/4
    return [[-ep + mean_ep,mean_ep,mean_ep,mean_ep] for i in range(L)]

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
