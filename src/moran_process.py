from tqdm import trange
from utils import inverse_cdf_sample,mean
import random

def moran_process(fitness,mutate,init_species, N=1000,turns=10000,pop=None,
                  diagnostic_modulus=100,diagnostics=lambda pop:mean([f for (s,f) in pop])):
    if pop is None:
        pop = [(lambda spec:(spec,fitness(spec)))(init_species())
               for _ in trange(N)]
    for turn in xrange(turns):
        fits = [f for (s,f) in pop]
        birth_idx = inverse_cdf_sample(range(N),fits,normalized=False)
        death_idx = random.randrange(N)
        mother,f = pop[birth_idx]
        daughter = mutate(mother)
        pop[death_idx] = (daughter,fitness(daughter))
        mean_fits = mean(fits)
        if turn % diagnostic_modulus == 0:
            print turn,diagnostics(pop)
    return pop
