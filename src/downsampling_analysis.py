from mi_analysis import get_motifs
from utils import motif_ic, motif_mi, percentile
from tqdm import *

def experiment1():
    """Does downsampling preserve percentile statistics?"""
    motif = (prok_motifs[11])
    downsamples = [sample(int(len(motif)/10), motif,replace=False) for i in range(100)]
    maxent_spoofs = spoof_maxent_motifs(motif, 1000, verbose=True)
    down_spoofs = [spoof_maxent_motifs(dm, 100) for dm in tqdm(downsamples)]
    true_mi, spoof_mis = motif_mi(motif), map(motif_mi, tqdm(maxent_spoofs))
    down_mis, down_spoof_mis = map(motif_mi, downsamples), [map(motif_mi, spoofs) for spoofs in tqdm(down_spoofs)]
    true_percentile = percentile(true_mi, spoof_mis)
    down_percentiles = [percentile(down_mi, ds_mis) for (down_mi, ds_mis) in zip (down_mis, down_spoof_mis)]
