"""Is there an inverse relationship between Gini and MI, as hypothesized?"""

from uniform_motif_sampling import uniform_motifs_accept_reject
from maxent_motif_sampling import maxent_motifs_with_ic
from chem_pot_model_on_off import sample_motif as sample_on_off_motif
from chem_pot_model_on_off import spoof_motif as spoof_on_off_motif
from exact_evo_sim_sampling import log_regress_spec2
from utils import motif_ic, motif_gini, total_motif_mi, mmap, fdr, maybesave
from tqdm import *
from matplotlib import pyplot as plt
from scipy.stats import pearsonr
from parse_tfbs_data import tfdf

def maxent_experiment():
    n, L, des_ic, num_motifs = 50,10,10,1000
    #motifs = uniform_motifs_accept_reject(n,L,desired_ic, num_motifs)
    motifs = maxent_motifs_with_ic(n,L,des_ic, num_motifs)
    ics = map(motif_ic, motifs)
    ginis = map(motif_gini, motifs)
    mis = map(total_motif_mi, motifs)

def on_off_experiment1():
    """compare MI vs Gini on synthetic motifs"""
    n, L, des_ic, num_motifs = 50,10,10,1000
    sigma = 1
    copies = 10*n
    def f(Ne):
        return motif_ic(sample_on_off_motif(sigma, Ne, L, copies, n)) - des_ic
    Ne = log_regress_spec2(f,[1,10],tol=10**-5)
    motifs = [sample_on_off_motif(sigma, Ne, L, copies, n) for i in trange(num_motifs)]
    ics = map(motif_ic, motifs)
    ginis = map(motif_gini, motifs)
    mis = map(total_motif_mi, motifs)
    plt.subplot(1,3,1)
    plt.scatter(ics, ginis)
    plt.xlabel("IC (bits)")
    plt.ylabel("Gini")
    print "ic vs gini:",pearsonr(ics,ginis)
    plt.subplot(1,3,2)
    plt.scatter(ics, mis)
    plt.xlabel("IC (bits)")
    plt.ylabel("MI (bits)")
    print "ic vs mi:",pearsonr(ics,mis)
    plt.subplot(1,3,3)
    plt.scatter(ginis, mis)
    plt.xlabel("Gini")
    plt.ylabel("Mi (bits)")
    print "gini vs mi:",pearsonr(ginis,mis)
    plt.tight_layout()
    param_template = ", ".join("{0}=%({0})s".format(v) for v in "n L des_ic sigma copies num_motifs".split())
    param_string = param_template % vars()
    plt.title("Gini vs MI in On-Off Simulations")
    print "Pearson correlation:",pearsonr(ginis,mis)
    print "parameters:", param_string

def on_off_experiment2(num_motifs=100,filename="gini-vs-mi-correlation-in-on-off-spoofs.pdf"):
    """compare MI vs Gini on biological_motifs"""
    bio_motifs = [getattr(tfdf,tf) for tf in tfdf.tfs]
    Ns = map(len, bio_motifs)
    spoofses = [spoof_on_off_motif(motif,num_motifs=num_motifs,trials=1) for motif in bio_motifs]
    spoof_ginises = mmap(motif_gini,tqdm(spoofses))
    spoof_mises = mmap(total_motif_mi,tqdm(spoofses))
    cors, ps = [],[]
    for ginis, mis in zip(ginises, mises):
        cor, p = pearsonr(ginis,mis)
        cors.append(cor)
        ps.append(p)
    q = fdr(ps)
    
    plt.scatter(cors,ps,filename="gini-vs-mi-correlation-in-on-off-spoofs.pdf")
    plt.plot([-1,1],[q,q],linestyle='--',label="FDR-Adjusted Significance Level")
    plt.semilogy()
    plt.legend()
    plt.xlabel("Pearson Correlation Coefficient")
    plt.ylabel("P value")
    plt.xlim([-1,1])
    plt.ylim([10**-4,1+1])
    cor_ps = zip(cors,ps)
    sig_negs = [(c,p) for (c,p) in cor_ps if c < 0 and p < q]
    sig_poses = [(c,p) for (c,p) in cor_ps if c > 0 and p < q]
    insigs = [(c,p) for (c,p) in cor_ps if p > q]
    def weighted_correlation(cor_p_Ns):
        cors,ps,Ns = transpose(cor_p_Ns)
        return sum([cor*N for (cor,N) in zip (cors,Ns)])/sum(Ns)
    plt.title("Gini-MI Correlation Coefficient vs. P-value for On-Off Simulations from Prokaryotic Motifs")
    maybesave(filename)
    
