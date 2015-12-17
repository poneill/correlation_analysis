from maxent_motif_sampling import maxent_motif_with_ic
from uniform_motif_sampling import uniform_motif_with_ic_imh, uniform_motif_with_ic_rw
from uniform_motif_sampling import uniform_motif_accept_reject
from matplotlib import pyplot as plt
import seaborn as sns
from utils import motif_ic, motif_gini, maybesave, concat, mmap, total_motif_mi
import numpy as np
from tqdm import *
from math import exp, log
from scipy import stats

def plot(filename=None):
    L = 10
    n = 50
    iterations = 1000
    trials = 30
    ic_range = np.linspace(2,2*L-1,10)
    maxent_motifs = [[maxent_motif_with_ic(n,L,des_ic) for trial in range(trials)]
                     for des_ic in tqdm(ic_range)]
    uniform_imh_motifs = [[chain[-1]
                           for chain in uniform_motif_with_ic_imh(n,L,des_ic,iterations=None,num_chains=trials)]
                          for des_ic in tqdm(ic_range)]
    uniform_rw_motifs = [[chain[-1]
                                 for chain in uniform_motif_with_ic_rw(n,L,des_ic,iterations=None,num_chains=trials)]
                                for des_ic in ic_range]
    plt.scatter(map(motif_ic,concat(maxent_motifs)), map(motif_gini, concat(maxent_motifs)),label="MaxEnt")
    plt.scatter(map(motif_ic,concat(uniform_imh_motifs)), map(motif_gini, concat(uniform_imh_motifs)),
                label="Uniform (IMH)",color='green',marker='^')
    plt.scatter(map(motif_ic,concat(uniform_rw_motifs)), map(motif_gini, concat(uniform_rw_motifs)),label="Uniform (RW)",
                color='red',marker='s')
    plt.legend()
    plt.xlim(0,2*L)
    plt.ylim(0,1)
    plt.xlabel("Information Content (bits)")
    plt.ylabel("Gini Coefficient")
    ps = []
    for xs,ys,zs in zip(maxent_motifs, uniform_imh_motifs, uniform_rw_motifs):
        statistic, p = stats.kruskal(*mmap(motif_gini,[xs,ys,zs]))
        ps.append(p)
    print "groups unequal?",min(ps) < (0.05/len(ps))
    maybesave(filename)
    return maxent_motifs, uniform_imh_motifs, uniform_rw_motifs

def plot2(filename=None,trials=50,motif_statistic=motif_gini):
    """Compare statistical properties of motifs sampled via MaxEnt, Uniform(AR) algorithms"""
    L = 10
    n = 50
    ic_samples = 10
    ic_range = np.linspace(2,2*L-1,ic_samples)
    # maxent_motifs = [[maxent_motif_with_ic(n,L,des_ic) for trial in range(trials)]
    #                  for des_ic in tqdm(ic_range)]
    # uniform_motifs = [[uniform_motif_accept_reject(n,L,des_ic) for trial in range(trials)]
    #                  for des_ic in tqdm(ic_range)]
    maxent_motifs = []
    uniform_motifs = []
    for des_ic in tqdm(ic_range):
        correction_per_col = 3/(2*log(2)*n)
        desired_ic = des_ic + (L * correction_per_col)
        beta = find_beta_for_mean_motif_ic(n,L,desired_ic)
        maxent_motifs.append([maxent_motif_with_ic(n,L,des_ic,beta=beta) for trial in range(trials)])
        uniform_motifs.append([uniform_motif_accept_reject(n,L,des_ic,beta=beta) for trial in range(trials)])
    plt.scatter(map(motif_ic,concat(maxent_motifs)), map(motif_statistic, concat(maxent_motifs)),label="MaxEnt")
    plt.scatter(map(motif_ic,concat(uniform_motifs)), map(motif_statistic, concat(uniform_motifs)),
                label="Uniform (AR)",color='green',marker='^')
    plt.legend()
    # plt.xlim(0,2*L)
    # plt.ylim(0,1)
    plt.xlabel("Information Content (bits)")
    #plt.ylabel("Gini Coefficient")
    gini_ps = []
    for xs,ys in zip(maxent_motifs, uniform_motifs):
        gini_statistic, gini_p = stats.kruskal(*mmap(motif_statistic,[xs,ys]))
        gini_ps.append(gini_p)
    print "statistics unequal?",min(gini_ps) < (0.05/len(gini_ps))
    maybesave(filename)
