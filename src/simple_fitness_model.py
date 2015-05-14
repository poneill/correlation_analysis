N = 100
mu = 0.001

import random
from moran_process import moran_process
import numpy as np

def sample_species():
    return np.random.randint(0,2,N)

def fitness(x):
    return sum(x)

def mutate(x):
    xp = np.copy(x)
    muts = np.random.random(N) < mu
    for i,mut in enumerate(muts):
        if mut:
            xp[i] = 1-xp[i]
    return xp
    
def test_fitness_model():
    pop = moran_process(fitness,mutate,sample_species)

