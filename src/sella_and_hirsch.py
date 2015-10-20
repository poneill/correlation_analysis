"""Test Sella and Hirsch's Boltzmann formula for distribution over
ancestor in Moran process, using a simple toy binary locus model"""

import random
from moran_process import moran_process
from math import exp
from utils import choose

M = 1000
N = 5
mu = 0.01/N
nu = N - 1

def predictions(M,N):
    nu = N - 1
    Z = sum(exp(-nu*j)*choose(M,j) for j in range(M+1))
    return [exp(-nu*j)*choose(M,j)/Z for j in range(M+1)]
    
def fitness(g):
    """fitness is multiplicatively proportional to number of zeros in g"""
    return exp(-sum(g))

def mutate(g):
    if random.random() > mu:
        return g
    else:
        mut_g = g[:]
        i = random.randrange(M)
        mut_g[i] = 1 - g[i]
        return mut_g

def init_species():
    return [random.choice([0,1]) for i in range(M)]

def ringer():
    return [0]*M
    
def experiment():
    """works beautifully!"""
    # make sure to adjust turns to achieve convergence
    pop = moran_process(fitness,mutate,init_species,N,turns=1000000,diagnostic_modulus=10000)
    plt.hist([sum(g) for (g,f) in pop])
    plt.plot(predictions(M,N))

