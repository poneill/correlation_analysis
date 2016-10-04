"""what is the probability of obtaining a motif w/ I bits of IC by chance?"""

from utils import random_motif, mutate_motif, motif_ic
from tqdm import *

def flux_prob(motif, a, b, trials=10000):
    """determine probability of mutation pushing motif out of interval [a,b]"""
    ic = motif_ic(motif)
    lesser = 0
    same = 0
    greater = 0
    for i in trange(trials):
        icp = motif_ic(mutate_motif(motif))
        if icp < a:
            lesser += 1
        elif a <= icp < b:
            same += 1
        else:
            greater += 1
    return lesser, same, greater
