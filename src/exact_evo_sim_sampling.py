from utils import random_site, inverse_cdf_sample, inverse_cdf_sampler, motif_ic, mean, mutate_site, mh
from utils import score_seq, argmin
from gle_chem_pot import approximate_mu
from why_linear_recognition_sanity_check import sample_matrix
from math import exp, log
import random
from matplotlib import pyplot as plt
from tqdm import *

def sample_site(E, L, mu, Ne, ringer_site):
    """Sample site of length L from stationary fitness distribution under
    E(s) at effective population Ne, chemical potential mu"""
    nu = Ne - 1
    def phat(s):
        return (1 + exp(E(s) - mu))**(-nu)
    phat_max = phat(ringer_site)
    while True:
        site = random_site(L)
        if random.random() < phat(site)/phat_max:
            return site

def sample_site_mh(E, L, mu, Ne, ringer_site, iterations=1000):
    nu = Ne - 1
    def phat(s):
        return (1 + exp(E(s) - mu))**(-nu)
    return mh(f=phat,proposal=mutate_site,x0=ringer_site, iterations=iterations)

def sample_motif_with_ic(n,L):
    matrix = sample_matrix(L,sigma=1)
    E = lambda s:score_seq(matrix, s)
    ringer_site = "".join(["ACGT"[argmin(col)] for col in matrix])
    mu = approximate_mu(matrix,10*n,G=5*10**6)
    Nes = range(2,10)
    trials = 10
    motifs = [[sample_motif_mh(E, L, mu, Ne, ringer_site, n) for t in range(trials)] for Ne in tqdm(Nes)]

def sample_motif(E, L, mu, Ne, ringer_site, n):
    return [sample_site(E, L, mu, Ne, ringer_site) for i in xrange(n)]

def sample_motif_mh(E, L, mu, Ne, ringer_site, n, iterations=1000):
    return [sample_site_mh(E, L, mu, Ne, ringer_site,iterations=1000)[-1] for i in xrange(n)]


def test():
    E = lambda site:-site.count("A")
    L = 10
    ringer_site = "A"*L
    n = 10
    trials = 10
    mus = range(-9,1,1)
    Nes = range(2,11,1)
    ics = [[mean(motif_ic(sample_motif(E, L, show(mu), Ne, ringer_site, n)) for i in range(trials)) for mu in tqdm(mus)]
           for Ne in Nes]
    plt.contourf(mus,Nes,ics)
    plt.colorbar(label="IC")
    plt.xlabel("Mu")
    plt.ylabel("Nes")
