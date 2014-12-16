"""Compute the pairwise mutual information for each column pair of
each motif.  Report positions and number of significant column pairs
for each motif, and assess significance."""
import sys
AT_LAB = True
from matplotlib import rc
#rc('text',usetex=True)
sys.path.append("/home/poneill/python_utils")
from utils import *
from collections import defaultdict
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import binom_test
from tqdm import tqdm
from scipy.stats import mannwhitneyu,wilcoxon
from parse_tfbs_data import tfbss
from motifs import Escherichia_coli

def read_prok_motifs():
    """load prokaryotic motifs into a dictionary of form {motif_name:[site]}"""
    prok_motif_dict = defaultdict(list)
    with open("../data/prok_sites/all.sites.prok") as f:
        for line in f:
            motif,site = line.strip().split(",")
            if not motif.startswith("MX"): #ignore first line: Motif,site
                continue
            prok_motif_dict[motif].append(site)
    return prok_motif_dict

def compute_motif_pairwise_mis(motif,pval=True):
    """For each column pair, return the mi and return them as a
    dictionary of the form: {(i,j):mi}"""
    replicates = 1000
    w = len(motif[0])
    cols = transpose(motif)
    mi_dict = {}
    for (i,j) in tqdm(choose2(range(w))):
        xs,ys = cols[i],cols[j]
        if pval:
            mi_dict[(i,j)] = (mi(xs,ys,correct=False),mi_permute(xs,ys,n=replicates,p_value=True,
                                                             mi_method=lambda x,y:mi(x,y,correct=False)))
        else:
            mi_dict[(i,j)] = mi(xs,ys,correct=True)
    return mi_dict

def compute_motif_pairwise_mis_properly(motif):
    replicates = 1000
    w = len(motif[0])
    cols = transpose(motif)
    mi_dict = {(i,j):mi(cols[i],cols[j],correct=False) for (i,j) in choose2(range(w))}
    perm_dict = defaultdict(list)
    for rep in tqdm(xrange(replicates),total=replicates):
        perm_cols = [permute(col) for col in transpose(motif)]
        for (i,j) in choose2(range(w)):
            xs,ys = perm_cols[i],perm_cols[j]
            perm_dict[(i,j)].append(mi(xs,ys,correct=False))
    return {(i,j):(mi_dict[(i,j)],len(filter(lambda m:m > mi_dict[(i,j)],perm_dict[(i,j)]))/float(replicates))
            for (i,j) in choose2(range(w))}
    
def all_pairwise_mis(properly=True,organism=Escherichia_coli):
    all_mis = {}
    for tf in organism.tfs:
        sites = getattr(organism,tf)
        if properly:
            all_mis[tf] = compute_motif_pairwise_mis_properly(sites)
        else:
            all_mis[tf] = compute_motif_pairwise_mis(sites)
    return all_mis

def fdrs_from_all_mis(all_mis,organism=Escherichia_coli):
    """given all_mis = all_pairwise_mis(),
    Compute false discovery rates for all motifs"""
    for tf in organism.tfs:
        pvals = [p for (m,p) in all_mis[tf].items()]
        print tf,"False Discovery Rate:",fdr(pvals)
    
    
def generate_distance_distribution(pmd):
    from accession2tf_name import accession2tf_name
    #pmd = read_prok_motifs()
    distance_dict = defaultdict(list)
    replicates = 1000
    for mx, motif in pmd.items():
        if mx in accession2tf_name:
            tf_name = accession2tf_name[mx]
        else:
            tf_name = mx
        print tf_name
        cols = transpose(motif)
        w = len(motif[0])
        for (i,j) in choose2(range(w)):
            xs,ys = cols[i],cols[j]
            distance = j - i
            distance_dict[distance].append(mi_permute(xs,ys,n=replicates,p_value=True,
                                                      mi_method=lambda x,y:mi(x,y,correct=False)))
    return distance_dict
                
def plot_distance_distribution(distance_dict,cutoff=50,filename=None):
    alpha = 0.05
    js = [j for j in sorted(distance_dict.keys()) if len(distance_dict[j]) > cutoff]
    plt.scatter(*transpose([(j,how_many(lambda x:x<0.05,distance_dict[j])/float(len(distance_dict[j])))
                            for j in js]))
    fdrs = [fdr(distance_dict[j]) for j in js]
    prop_tests = [binom_test(how_many(lambda x:x<0.05,distance_dict[j]),len(distance_dict[j]),p=0.05)
                  for j in js]
    # plt.plot(js,fdrs)
    # plt.plot(js,prop_tests)
    plt.plot(js,[alpha]*len(js),linestyle='--')
    plt.xlabel("Distance between bases (bp)")
    plt.ylabel("Discovery rate at $\\alpha$ = 0.05")
    plt.title("Discovery Rate for MI Permutation Tests in PRODORIC motifs, by distance")
    maybesave(filename)

def plot_distance_distribution2(distance_dict,filename=None):
    js = sorted(distance_dict.keys())
    fdrs = [fdr(distance_dict[j]) for j in js]
    plt.scatter(*transpose([(j,p) for j in distance_dict.keys() for p in distance_dict[j]]))
    plt.plot(js,fdrs)
    maybesave(filename)

def plot_distance_distribution3(all_mis,organism=Escherichia_coli):
    for tf,mi_dict in all_mis.items():
        sites = getattr(organism,tf)
        w = len(sites[0])
        distance_dict = defaultdict(list)
        for i,j in choose2(range(w)):
            m,pval = mi_dict[(i,j)]
            distance_dict[j-i].append(m)
        #print distance_dict
        plt.plot(range(1,w),[mean(distance_dict[i]) for i in range(1,w)])




            
def analyze_pairwise_mi_dict(mi_dict):
    """Given an mi_dict as returned by compute_motif_pairwise_mis,
    pretty print the results"""
    motif_width = max(j for (i,j) in mi_dict) + 1
    tests = len(mi_dict)
    positives = 0
    adjacents = 0
    positive_adjacents = 0
    obs_p_dict = {}
    adjacent_mis = []
    non_adjacent_mis = []
    for i,j in sorted(mi_dict):
        mi_obs,p_val = mi_dict[(i,j)]
        positive = p_val < 0.05
        positives += positive
        adjacent = (j == i + 1)
        adjacents += adjacent
        if adjacent:
            adjacent_mis.append(mi_obs)
        else:
            non_adjacent_mis.append(mi_obs)
        positive_adjacents += positive * adjacent
        mi_test_string = "POSITIVE" if positive else "negative"
        obs_p_dict[(i,j)] = (mi_obs,p_val)
        #print i,j,mi_obs,p_val,mi_test_string,("adjacent" if adjacent else "")
    print "Motif had width:",motif_width
    print "tests:",tests
    print "positives:",positives
    print "positive_rate:",positives/float(tests)
    print "adjacents:",adjacents
    print "positive_adjacents",positive_adjacents
    print "positive_adjacents/positives:",positive_adjacents/float(positives) if positives else 0
    print "adjacencts/tests:",adjacents/float(tests)
    print "Adjacent mis higher:",mean(adjacent_mis),mean(non_adjacent_mis),mannwhitneyu(adjacent_mis,non_adjacent_mis)
    return obs_p_dict

def test_adjacency_hypothesis_for_all_tfs(all_mis):
    adjacent_means = []
    non_adjacent_means = []
    for tf,mi_dict in all_mis.items():
        adjacent_mis = []
        non_adjacent_mis = []
        for i,j in sorted(mi_dict):
            mi_obs,p_val = mi_dict[(i,j)]
            if j == i + 1:
                adjacent_mis.append(mi_obs)
            else:
                non_adjacent_mis.append(mi_obs)
        adjacent_means.append(mean(adjacent_mis))
        non_adjacent_means.append(mean(non_adjacent_mis))
    print mean(adjacent_means),mean(non_adjacent_means),wilcoxon(adjacent_means,non_adjacent_means)
    return adjacent_means,non_adjacent_means

def plot_adj_hypothesis(filename=None):
    adj_means,non_adj_means = test_adjacency_hypothesis_for_all_tfs(all_mis)
    # for adj,non_adj in zip(adj_means,non_adj_means):
    #     plt.plot([adj,non_adj])
    plt.scatter(adj_means,non_adj_means)
    plt.xlabel("Mean Adjacent MI")
    plt.ylabel("Mean Non-Adjacent MI")
    plt.plot([0,0.5],[0,0.5],linestyle='--')
    maybesave(filename)

        
def plot_ic_vs_mi(tf_or_motif,filename=None,organism=Escherichia_coli,plotting=True):
    if type(tf_or_motif) is str:
        tf = tf_or_motif
        sites = getattr(organism,tf)
    else:
        sites = tf_or_motif
    ics = columnwise_ic(sites,correct=False)
    mis = [mi(colx,coly,correct=False) for colx,coly in pairs(transpose(sites))]
    if plotting:
        plt.plot(ics,label="Information Content")
        plt.plot([0.5+i for i in range(len(mis))],mis,label="Adjacent MI")
        plt.ylabel("bits")
        plt.xlabel("column")
        plt.legend(loc=0)
        if type(tf_or_motif) is str:
            plt.title("Information Content and Mutual Information in %s Binding Motif" % tf)
        maybesave(filename)
    return pearsonr(ics[:-1] + ics[1:],mis + mis)
    
def main():
    pmd = read_prok_motifs()
    motif_stats = []
    for i,(name,motif) in enumerate(pmd.items()):
        print "Motif:",i,name
        motif_stat = analyze_pairwise_mi_dict(compute_motif_pairwise_mis(motif))
        motif_stats.append(motif_stat)
        print
    return motif_stats

def make_heat_map(tf_name,motif=None,savefig=True,mi_method=mi,organism=Escherichia_coli):
    if motif is None:
        motif = getattr(organism,tf_name)
    cols = transpose(motif)
    mis = [[mi_method(x,y) if i + 1 < j else np.nan
            for i,x in enumerate(cols)]
           for j,y in enumerate(cols)]
    cax = plt.matshow(np.matrix(mis))
    #plt.colorbar(cax)
    plt.title(tf_name)
    plt.xlabel("First Column")
    plt.ylabel("Second Column")
    plt.colorbar(cax,label="MI (bits)")
    if savefig:
        plt.savefig(tf_name + "_mi_heatmap.png",dpi=400)

def heat_maps_for_nsf_grant_report():
    #Wed Apr 23 19:31:50 EDT 2014
    from accession2tf_name import accession2tf_name
    pmd = read_prok_motifs()
    for mx, motif in pmd.items():
        if mx in accession2tf_name:
            tf_name = accession2tf_name[mx]
            print tf_name
            bf_alpha = 0.05#/len(choose2(motif[0]))
            make_heat_map(tf_name,motif,mi_method=lambda x,y:mi_permute(x,y,p_value=True,n=1000,correct=False)
                          < bf_alpha)
