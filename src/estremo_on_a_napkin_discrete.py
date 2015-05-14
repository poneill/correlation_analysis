import numpy as np
import random
from simplex_sampling_experiment import make_kmers,h_np
from estremo_on_a_napkin import sample_eps,boltzmann
from utils import random_site,inverse_cdf_sample,mean,motif_ic
from tqdm import tqdm,trange

n = 16
L = 5
beta = 1
K = 4**L
G = 5*10**6
site_mu = 0.01
rec_mu = 0.001

def make_idx_of_word():
    return {w:i for i,w in enumerate(make_kmers(L))}

idx_of_word = make_idx_of_word()

def sample_sites(n=n,L=L):
    return [random_site(L) for i in range(n)]

def occs_ref(sites,eps):
    site_eps = np.array([eps[idx_of_word[site]] for site in sites])
    backround_eps = np.random.choice(eps,G)
    all_eps = np.hstack((site_eps,backround_eps))
    occs = boltzmann(all_eps)
    return occs[:len(sites)]

def occs(sites,eps,mode='binary'):
    site_eps = np.array([eps[idx_of_word[site]] for site in sites])
    Zb = background_Z(eps)
    site_exps = np.exp(-site_eps)
    occs = site_exps/(np.sum(site_exps) + Zb)
    return occs
    
def background_Z(eps):
    exps = np.exp(-eps)
    Z = np.sum(exps)
    return Z * G/float(len(eps))

def sample_Z(eps):
    backround_eps = np.random.choice(eps,G)
    return np.sum(np.exp(-backround_eps))
    
def test_background_Z():
    eps = sample_eps()
    sampled_Z = sample_Z(eps)
    pred_Z = background_Z(eps)
    return pred_Z,sampled_Z
    
def fitness((sites,eps)):
    return np.sum(occs(sites,eps))

def mutate((sites,eps)):
    new_sites = mutate_sites(sites,site_mu)
    new_eps = eps + np.random.normal(0,rec_mu,len(eps))
    return (new_sites,new_eps)

def mutate_char(b,mu_prob):
    return b if random.random() > mu_prob else random.choice([c for c in "ACGT" if not c == b])
    
def mutate_site(site,mu_prob):
    return "".join(mutate_char(b,mu_prob) for b in site)

def mutate_sites(sites,mu_prob):
    return [mutate_site(site,mu_prob) for site in sites]

def sample_species():
    return (sample_sites(),sample_eps())

def moran_process(N=1000,turns=10000,init=sample_species,mutate=mutate,fitness=fitness,pop=None):
    if pop is None:
        pop = [(lambda spec:(spec,fitness(spec)))(sample_species())
               for _ in trange(N)]
    hist = []
    for turn in xrange(turns):
        fits = [f for (s,f) in pop]
        #print fits
        birth_idx = inverse_cdf_sample(range(N),fits,normalized=False)
        death_idx = random.randrange(N)
        #print birth_idx,death_idx
        mother,f = pop[birth_idx]
        daughter = mutate(mother)
        #print "mutated"
        pop[death_idx] = (daughter,fitness(daughter))
        mean_fits = mean(fits)
        hist.append((f,mean_fits))
        if turn % 10 == 0:
            mean_dna_ic = mean([motif_ic(sites,correct=False) for ((sites,eps),_) in pop])
            mean_rec_h = mean([h_np(boltzmann(eps)) for ((dna,eps),_) in pop])
            print turn,"sel_fit:",f,"mean_fit:",mean_fits,"mean_dna_ic:",mean_dna_ic,"mean_rec_h:",mean_rec_h
    return pop
