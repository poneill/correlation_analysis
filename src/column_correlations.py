from utils import sample, choose2, mi, percentile, fdr, count, choose, transpose, permute, dna_mi
import cPickle
import sys
sys.path.append("/home/pat/motifs")
#from parse_merged_data import tfdf
from parse_tfbs_data import tfdf
sys.path.append("/home/pat/jaspar")
from parse_jaspar import jaspar_motifs
from tqdm import *
import numpy as np

def get_motifs():
    prok_motifs = [getattr(tfdf,tf) for tf in tfdf.tfs]
    # euk_motifs = [motif if len(motif) <= 200 else sample(200,motif,replace=False)
    #               for motif in jaspar_motifs]
    euk_motifs = jaspar_motifs
    return prok_motifs, euk_motifs

def analyze_motif(motif, trials=1000):
    cols = transpose(motif)
    L = len(cols)
    ps = []
    for col1, col2 in (choose2(cols)):
        actual_mi = dna_mi(col1,col2)
        perm_mis = [dna_mi(col1,permute(col2)) for i in xrange(trials)]
        p = percentile(actual_mi, perm_mis)
        #print p
        ps.append(p)
    q = fdr(ps)
    correlated_pairs = [(i,j) for (i,j),p in zip(choose2(range(L)),ps) if p < q]
    num_correlated = len(correlated_pairs)
    print "correlated column pairs:", num_correlated, "%1.2f" % ((num_correlated)/choose(L,2))
    return correlated_pairs

def analyze_collection(prok_motifs, euk_motifs):
    prok_correlated_pairses = map(analyze_motif,tqdm(prok_motifs,desc='motifs'))
    with open("prok_correlated_pairses.pkl",'w') as f:
        cPickle.dump(f,prok_correlated_pairses)
    euk_correlated_pairses = map(analyze_motif,tqdm(euk_motifs,desc='motifs'))
    with open("euk_correlated_pairses.pkl",'w') as f:
        cPickle.dump(f,euk_correlated_pairses)
    prok_corrs = np.array(map(len,prok_correlated_pairses))
    euk_corrs = np.array(map(len,euk_correlated_pairses))
    prok_depths = np.array([len(motif) for motif in prok_motifs])
    euk_depths = np.array([len(motif) for motif in euk_motifs])
    prok_lens = np.array([len(motif[0]) for motif in prok_motifs])
    euk_lens = np.array([len(motif[0]) for motif in euk_motifs])
    prok_lc2s = np.array([choose(L,2) for L in prok_lens])
    euk_lc2s = np.array([choose(L,2) for L in euk_lens])
