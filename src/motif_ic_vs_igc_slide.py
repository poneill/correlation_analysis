from tqdm import *
import numpy as np
from utils import motif_ic, motif_gini, concat, maybesave
from formosa import maxent_motifs
from matplotlib import pyplot as plt


def make_motifs(bins=100):
    motifs = [maxent_motifs(100, 10, ic, 10) for ic in tqdm(np.linspace(0, 19, bins))]
    return motifs
    
def main(motifs, fname=None):    
    plt.scatter(map(motif_ic, concat(motifs)), map(motif_gini, concat(motifs)))
    plt.xlabel("IC (bits)")
    plt.ylabel("IGC")
    plt.ylim(0,0.6)
    plt.xlim(-0.5,20)
    plt.legend()
    maybesave(fname)
    
