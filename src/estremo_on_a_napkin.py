import numpy as np
from simplex_sampling_experiment import simplexify_sample,mutate_ref,mutation_matrix_ref,h_np
from utils import inverse_cdf_sample,mean,normalize
import random
from math import exp,log

L = 5
K = 4**L
#mu = 0#1.0/K
beta = 1
#ep_sigma = 10**-4

#Mut = mutation_matrix_ref(mu,L,mode='discrete')

def fitness((qs,eps)):
    exps = qs*np.exp(-beta*eps)
    Z = np.sum(exps)
    ps = exps/Z
    return 4*log(L,2)-(h_np(ps))

def experimental_entropy((qs,eps),G):
    exps = np.exp(-beta*eps)
    Z = G*np.sum(qs*exps)
    return -sum(G*q*exp(-beta*ep)/Z*log(exp(-beta*ep)/Z) for q,ep in zip(qs,eps))
    
def sample_qs(sigma=1):
    return simplexify_sample(K,sigma=sigma)

def sample_eps(sigma=1):
    return np.random.normal(0,sigma,K)

def sample_species():
    return (sample_qs(),sample_eps())

def mutate((qs,eps),mu,ep_sigma):
    Mut = mutation_matrix_ref(mu,L,mode='discrete',stochastic=True)
    new_qs = Mut.dot(qs)
    new_eps = eps + np.random.normal(0,ep_sigma,K)
    return (new_qs,new_eps)

def moran_process(N=1000,turns=10000,mu=10**-3,ep_sigma=10**-3):
    ringer = (np.array([1]+[0]*(K-1)),sample_eps())
    # pop = [(lambda spec:(spec,fitness(spec)))(sample_species())
    #        for _ in xrange(N)]
    pop = [(ringer,fitness(ringer)) for _ in xrange(N)]
    hist = []
    for turn in xrange(turns):
        fits = [f for (s,f) in pop]
        #print fits
        birth_idx = inverse_cdf_sample(range(N),fits,normalized=False)
        death_idx = random.randrange(N)
        #print birth_idx,death_idx
        mother,f = pop[birth_idx]
        daughter = mutate(mother,mu,ep_sigma)
        #print "mutated"
        pop[death_idx] = (daughter,fitness(daughter))
        mean_fits = mean(fits)
        hist.append((f,mean_fits))
        if turn % 100 == 0:
            mean_dna_h = mean([h_np(dna) for ((dna,eps),f) in pop])
            mean_rec_h = mean([h_np(boltzmann(eps)) for ((dna,eps),f) in pop])
            print turn,"mean_fitness:",f,mean_fits,mean_dna_h,mean_rec_h
    return hist

def boltzmann(eps):
    exps = np.exp(eps)
    return exps/np.sum(exps)
