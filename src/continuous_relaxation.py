from utils import concat
import numpy as np

G = 5*10**6

def binarize_base(b):
    return [int(b == c) for c in "ACGT"]

def binarize_site(site):
    return concat([binarize_base(b) for b in site])

def binarize_motif(motif):
    return concat([binarize_site(site) for site in motif])

def log_binary_fitness(matrix,x,n,L,G):
    E = np.hstack(concat([matrix]*n))
    
