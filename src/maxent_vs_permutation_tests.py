from utils import permute, transpose, sample, mean, sorted_indices, rslice
from mi_analysis import get_motifs
from formosa import spoof_maxent_motifs, spoof_uniform_motifs
from formosa_utils import motif_ic, motif_mi
from tqdm import *

def perm_motif(motif):
    """permute columns of motif"""
    cols = transpose(motif)
    colsp = map(permute, cols)
    return ["".join(row) for row in transpose(colsp)]

def main():
    prok_motifs, euk_motifs = get_motifs()
    prok_motifs = [sample(200, motif, replace=False) if len(motif) > 200 else motif for motif in tqdm(prok_motifs)]
    mis = map(motif_mi, prok_motifs)
    js = sorted_indices(mis)
    maxent_mis = [mean(map(motif_mi, spoof_maxent_motifs(motif, 1000))) for motif in tqdm(prok_motifs)]
    uniform_mis = [mean(map(motif_mi, spoof_uniform_motifs(motif, 1000))) for motif in tqdm(prok_motifs)]
    perm_mis = [mean(map(motif_mi, [perm_motif(motif) for _ in xrange(1000)])) for motif in tqdm(prok_motifs)]
    plt.plot(rslice(mis, js))
    plt.plot(rslice(maxent_mis, js))
    plt.plot(rslice(perm_mis, js))
