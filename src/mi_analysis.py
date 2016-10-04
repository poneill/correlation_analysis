from maxent_sampling import spoof_maxent_motifs
from formosa_utils import motif_mi, count, mi_test_cols, transpose, choose2, occupancies, approx_mu
from formosa_utils import sample_matrix
from evo_sampling import sample_motif_cftp, spoof_motif_cftp_occ, spoof_motif_cftp
from chem_pot_model_on_off import spoof_motifs as spoof_oo_motifs
from chem_pot_model_on_off import spoof_motifs_occ as spoof_oo_motifs_occ
import cPickle
import random
from utils import fdr, inverse_cdf_sample, normalize, concat, maybesave, choose, mean, sd
from utils import mh, motif_ic, sample, scatter, score_seq, bs, indices_where, rslice, columnwise_ic
import sys
from tqdm import *
from math import sqrt, exp, log, sin, pi
from matplotlib import pyplot as plt
from collections import defaultdict
import seaborn as sns
from pwm_utils import matrix_from_motif, spoof_psfm
from scipy.stats import pearsonr

def mi_percentile(motif, trials=1000, verbose=False):
    obs_mi = motif_mi(motif)
    iterator = tqdm if verbose else lambda x:x
    spoof_mis = map(motif_mi, iterator(spoof_maxent_motifs(motif,num_motifs=trials)))
    return count(lambda x:x >= obs_mi, spoof_mis)/float(trials)

def motif_mi_col_test(motif, trials=1000):
    cols = transpose(motif)
    return sum(mi_test_cols(colA, colB) for colA, colB in choose2(cols))/float(len(choose2(cols)))

def motif_test_cols(motif):
    cols = transpose(motif)
    return [mi_test_cols(colA, colB, alpha=None) for colA, colB in choose2(cols)]

def motif_mi_dist(motif):
    cols = transpose(motif)
    return [dna_mi(colA, colB) for colA, colB in choose2(cols)]
    
def motif_mi_distances(motif, trials=1000):
    cols = transpose(motif)
    L = len(cols)
    correlated_distances = [j-i for (i,coli), (j,colj) in choose2(list(enumerate(cols)))
                            if mi_test_cols(coli, colj)]
    return (correlated_distances, L)
    
def get_motifs():
    sys.path.append("/home/pat/jaspar")
    from parse_jaspar import jaspar_motifs as euk_motifs
    #sys.path.append("/home/pat/motifs")
    sys.path.append("/home/pat/correlation_analysis/src")
    from parse_tfbs_data import tfdf
    prok_motifs = [getattr(tfdf,tf) for tf in tfdf.tfs]
    return prok_motifs, euk_motifs

def do_mi_tests():
    random.seed("do_mi_tests")
    prok_motifs, euk_motifs = get_motifs()
    prok_tests = [motif_test_cols(motif) for motif in tqdm(prok_motifs)]
    euk_tests = [motif_test_cols(motif) for motif in tqdm(euk_motifs)]
    with open("prok_tests.pkl",'w') as f:
        cPickle.dump(prok_tests,f)
    with open("euk_tests.pkl",'w') as f:
        cPickle.dump(euk_tests,f)

def get_tests():
    with open("prok_tests.pkl") as f:
        prok_tests = cPickle.load(f)
    with open("euk_tests.pkl") as f:
        euk_tests = cPickle.load(f)
    return prok_tests, euk_tests
        
def analyze_mi_tests(prok_tests, euk_tests):
    pass
    prok_q = fdr(concat(prok_tests))
    euk_q = fdr(concat(euk_tests))
    prok_correlated_percentage = count(lambda x:x <= prok_q,(concat(prok_tests)))/float(len(concat(prok_tests)))
    euk_correlated_percentage = count(lambda x:x <= euk_q,(concat(euk_tests)))/float(len(concat(euk_tests)))
    prok_ds = [[j - i for (i, coli), (j,colj) in choose2(list(enumerate(transpose(motif))))]
               for motif in prok_motifs]
    euk_ds = [[j - i for (i, coli), (j,colj) in choose2(list(enumerate(transpose(motif))))]
               for motif in euk_motifs]
    def binom_ci(xs):
        """return width of error bar"""
        bs_means = sorted([mean(bs(xs)) for x in range(1000)])
        mu = mean(xs)
        return (mu - bs_means[25], bs_means[975] - mu)
    prok_cis = [binom_ci([t <= prok_q for t,d in zip(concat(prok_tests), concat(prok_ds)) if d == i])
                for i in trange(1,20)]
    euk_cis = [binom_ci([t <= euk_q for t,d in zip(concat(euk_tests), concat(euk_ds)) if d == i])
                for i in trange(1,20)]
    plt.errorbar(range(1,20),
                 [mean([t <= prok_q for t,d in zip(concat(prok_tests), concat(prok_ds)) if d == i])
                  for i in range(1,20)],yerr=transpose(prok_cis),label="Prokaryotic Motifs",capthick=1)
    plt.errorbar(range(1,20),
                 [mean([t <= euk_q for t,d in zip(concat(euk_tests), concat(euk_ds)) if d == i])
                  for i in range(1,20)],yerr=transpose(euk_cis),label="Eukaryotic Motifs",capthick=1)
    plt.xlabel("Distance (bp)",fontsize="large")
    plt.ylabel("Proportion of Significant Correlations",fontsize="large")
    plt.legend(fontsize='large')

def analyze_mi_tests2(tests, motifs, q=None, label=None):
    q = fdr(concat(tests))
    correlated_percentage = count(lambda x:x <= q,(concat(tests)))/float(len(concat(tests)))
    ds = [[j - i for (i, coli), (j,colj) in choose2(list(enumerate(transpose(motif))))]
               for motif in motifs]
    def binom_ci(xs):
        """return width of error bar"""
        bs_means = sorted([mean(bs(xs)) for x in range(1000)])
        mu = mean(xs)
        return (mu - bs_means[25], bs_means[975] - mu)
    tests_by_dist = [[t <= q for t,d in zip(concat(tests), concat(ds)) if d == i] for i in range(1, 20)]
    mean_vals = map(lambda xs:mean(xs) if xs else 0, tests_by_dist)
    cis = map(lambda xs:binom_ci(xs) if xs else (0,0), tests_by_dist)
    plt.errorbar(range(1,20),
                 mean_vals,yerr=transpose(cis),label=label,capthick=1)
    plt.xlabel("Distance (bp)",fontsize="large")
    plt.ylabel("Proportion of Significant Correlations",fontsize="large")
    plt.legend()

def make_correlation_structure_figure():
    plt.subplot(1,2,1)
    plt.ylim(0,0.2)
    analyze_mi_tests2(prok_tests, prok_motifs, label='Biological')
    analyze_mi_tests2(prok_maxent_tests, prok_motifs, label="MaxEnt")
    analyze_mi_tests2(prok_psfm_tests, prok_motifs, label="PSFM")
    analyze_mi_tests2(prok_apw_tests, prok_motifs, label="APW")
    analyze_mi_tests2(prok_bayes_tests, prok_motifs, label="Evolutionary")
    plt.title("Prokaryotic Motifs",fontsize='large')
    plt.subplot(1,2,2)
    plt.ylim(0,0.2)
    analyze_mi_tests2(euk_tests, euk_motifs, label='Biological')
    analyze_mi_tests2(euk_maxent_tests, euk_motifs, label="MaxEnt")
    analyze_mi_tests2(euk_psfm_tests, euk_motifs, label="PSFM")
    analyze_mi_tests2(euk_apw_tests, euk_motifs, label="APW")
    analyze_mi_tests2(euk_bayes_tests, euk_motifs, label="Evolutionary")
    plt.title("Eukaryotic Motifs", fontsize='large')

def make_correlation_structure_by_cluster_figure():
    from motif_clustering import cluster_motif
    q = fdr(concat(euk_tests))
    euk_clusterses = [map(cluster_motif, tqdm(euk_motifs)) for i in range(3)]
    plt.close() # get rid of output from cluster_motif
    mean_lens = map(lambda xs:round(mean(xs)), transpose([map(len,cs) for cs in euk_clusterses]))
    jss = [indices_where(mean_lens, lambda x:x==i) for i in range(1, 5+1)]
    for i,js in tqdm(enumerate(jss)):
        analyze_mi_tests2(rslice(euk_tests, js), rslice(euk_motifs, js), label=str(i+1), q=q)

def make_correlation_structure_by_length():
    q = fdr(concat(euk_tests))
    plt.close() # get rid of output from cluster_motif
    lens = map(len, euk_motifs)
    jss = [indices_where(lens, lambda x:10**i <= x < 10**(i+1)) for i in range(1, 4+1)]
    for i,js in tqdm(enumerate(jss)):
        analyze_mi_tests2(rslice(euk_tests, js), rslice(euk_motifs, js), label=str("10**%s" % (i+1)), q=q)
    
def sanity_check_analyze_correlated_digrams(motifs):
    digrams = defaultdict(int)
    adj_digrams = defaultdict(int)
    for motif in motifs:
        for ((i,coli),(j,colj)) in choose2(list(enumerate(transpose((motif))))):
            for bi,bj in transpose((coli,colj)):
                digrams[(bi,bj)] += 1
                if j == i + 1:
                    adj_digrams[(bi,bj)] += 1
    return digrams, adj_digrams
                    
def analyze_correlated_digrams(prok_tests, euk_tests, filename=None):
    digrams = [(b1,b2) for b1 in "ACGT" for b2 in "ACGT"]
    prok_q = fdr(concat(prok_tests))
    euk_q = fdr(concat(euk_tests))
    prok_digrams = defaultdict(int)
    prok_corr_digrams = defaultdict(int)
    prok_adj_digrams = defaultdict(int)
    for tests, motif in tqdm(zip(prok_tests, prok_motifs)):
        for test, ((i,coli),(j,colj)) in zip(tests, choose2(list(enumerate(transpose((motif)))))):
            for bi,bj in transpose((coli,colj)):
                prok_digrams[(bi,bj)] += 1
                if j == i + 1:
                    prok_adj_digrams[(bi,bj)] += 1
                if test <= prok_q:
                    prok_corr_digrams[(bi,bj)] += 1
    prok_corr_N = float(sum(prok_corr_digrams.values()))
    prok_adj_N = float(sum(prok_adj_digrams.values()))
    prok_N = float(sum(prok_digrams.values()))
    #prok_ps = normalize(prok_digrams.values())
    #prok_adj_ps = normalize(prok_adj_digrams.values())
    #prok_corr_ps = normalize(prok_corr_digrams.values())
    prok_ps = normalize([prok_digrams[dg] for dg in digrams])
    prok_adj_ps = normalize([prok_adj_digrams[dg] for dg in digrams])
    prok_corr_ps = normalize([prok_corr_digrams[dg] for dg in digrams])
    prok_yerr = [1.96*sqrt(1.0/prok_N*p*(1-p)) for p in prok_ps]
    prok_adj_yerr = [1.96*sqrt(1.0/prok_adj_N*p*(1-p)) for p in prok_adj_ps]
    prok_corr_yerr = [1.96*sqrt(1.0/prok_corr_N*p*(1-p)) for p in prok_corr_ps]

    euk_digrams = defaultdict(int)
    euk_corr_digrams = defaultdict(int)
    euk_adj_digrams = defaultdict(int)
    for tests, motif in tqdm(zip(euk_tests, euk_motifs)):
        for test, ((i,coli),(j,colj)) in zip(tests, choose2(list(enumerate(transpose((motif)))))):
            for bi,bj in transpose((coli,colj)):
                euk_digrams[(bi,bj)] += 1
                if j == i + 1:
                    euk_adj_digrams[(bi,bj)] += 1
                if test <= euk_q:
                    euk_corr_digrams[(bi,bj)] += 1
    euk_corr_N = float(sum(euk_corr_digrams.values()))
    euk_adj_N = float(sum(euk_adj_digrams.values()))
    euk_N = float(sum(euk_digrams.values()))
    # euk_ps = normalize(euk_digrams.values())
    # euk_adj_ps = normalize(euk_adj_digrams.values())
    # euk_corr_ps = normalize(euk_corr_digrams.values())
    euk_ps = normalize([euk_digrams[dg] for dg in digrams])
    euk_adj_ps = normalize([euk_adj_digrams[dg] for dg in digrams])
    euk_corr_ps = normalize([euk_corr_digrams[dg] for dg in digrams])
    euk_yerr = [1.96*sqrt(1.0/euk_N*p*(1-p)) for p in euk_ps]
    euk_adj_yerr = [1.96*sqrt(1.0/euk_adj_N*p*(1-p)) for p in euk_adj_ps]
    euk_corr_yerr = [1.96*sqrt(1.0/euk_corr_N*p*(1-p)) for p in euk_corr_ps]

    palette = sns.cubehelix_palette(4)
    ax = plt.subplot(211)
    # plt.bar(range(16),normalize(prok_digrams.values()))
    # plt.bar(range(16),normalize(prok_corr_digrams.values()),color='g')
    # plt.bar([x-0.2 for x in range(16)], prok_relative_ratios.values(), color='g', label="Correlated Column-pairs",width=0.2)
    # plt.bar([x for x in range(16)],prok_adj_relative_ratios.values(),color='r',alpha=1,yerr=prok_adj_yerr,label="Adjacent Column-pairs",width=0.2)
    # plt.bar([x+0.2 for x in range(16)],[1]*16,color='b',alpha=1,yerr=(prok_yerr),capsize=10,capstyle='butt',label="All Column-pairs",width=0.2)
    plt.bar([x-0.2 for x in range(16)], prok_ps, label="All Column-Pairs",width=0.2,yerr=prok_yerr,color=palette[0])
    plt.bar([x for x in range(16)],prok_adj_ps,label="Adjacent Column-Pairs",
            width=0.2,yerr=prok_adj_yerr,color=palette[1])
    plt.bar([x+0.2 for x in range(16)],prok_corr_ps,alpha=1,
            capstyle='butt',label="Correlated Column-Pairs",width=0.2,yerr=prok_corr_yerr,color=palette[3])
    plt.plot([0,16],[1.0/16, 1.0/16],linestyle='--',color=palette[3],label="Equiprobability",linewidth=1)
    ax.set_xticks([x for x in range(16)])
    ax.set_xticklabels( ["".join(dg) for dg in digrams],fontsize='large')
    plt.xlim(-0.5,15.5)
    plt.ylim(0,0.2)
    #plt.xlabel("Dimer",fontsize='large')
    plt.ylabel("Prokaryotic Frequency",fontsize='large')
    #plt.ylim(0,2)
    plt.legend()
    
    ax2 = plt.subplot(212)
    plt.plot([0,16],[1.0/16, 1.0/16],linestyle='--',color=palette[3],label="Equiprobability",linewidth=1)
    plt.bar([x-0.2 for x in range(16)], euk_ps, label="All Column-Pairs",width=0.2,yerr=euk_yerr,color=palette[0])
    plt.bar([x for x in range(16)],euk_adj_ps,label="Adjacent Column-Pairs",
            width=0.2,yerr=euk_adj_yerr,color=palette[1])
    plt.bar([x+0.2 for x in range(16)],euk_corr_ps,alpha=1,
            capstyle='butt',label="Correlated Column-Pairs",width=0.2,yerr=euk_corr_yerr,color=palette[3])
    ax2.set_xticks([x for x in range(16)])
    ax2.set_xticklabels( ["".join(dg) for dg in digrams],fontsize='large')
    #plt.xlabel("Dimer",fontsize='large')
    plt.xlim(-0.5,15.5)
    plt.ylim(0,0.2)
    plt.ylabel("Eukaryotic Frequency",fontsize='large')
    #plt.ylim(0,2)
    plt.legend()
    maybesave(filename)

def analyze_correlated_digrams_canonical(prok_tests, euk_tests, filename=None):
    digrams = [(b1,b2) for b1 in "ACGT" for b2 in "ACGT"]
    canonical_digrams = sorted(list(set([min(dg,tuple(wc(dg))) for dg in digrams])))
    prok_q = fdr(concat(prok_tests))
    euk_q = fdr(concat(euk_tests))
    prok_digrams = defaultdict(int)
    prok_corr_digrams = defaultdict(int)
    prok_adj_digrams = defaultdict(int)
    for tests, motif in tqdm(zip(prok_tests, prok_motifs)):
        for test, ((i,coli),(j,colj)) in zip(tests, choose2(list(enumerate(transpose((motif)))))):
            for bi,bj in transpose((coli,colj)):
                rev_comp = tuple(wc((bi,bj)))
                if (bi, bj) > rev_comp:
                    bi, bj = rev_comp
                prok_digrams[(bi,bj)] += 1
                if j == i + 1:
                    prok_adj_digrams[(bi,bj)] += 1
                if test <= prok_q:
                    prok_corr_digrams[(bi,bj)] += 1
    prok_corr_N = float(sum(prok_corr_digrams.values()))
    prok_adj_N = float(sum(prok_adj_digrams.values()))
    prok_N = float(sum(prok_digrams.values()))
    #prok_ps = normalize(prok_digrams.values())
    #prok_adj_ps = normalize(prok_adj_digrams.values())
    #prok_corr_ps = normalize(prok_corr_digrams.values())
    prok_ps = normalize([prok_digrams[dg] for dg in canonical_digrams])
    prok_adj_ps = normalize([prok_adj_digrams[dg] for dg in canonical_digrams])
    prok_corr_ps = normalize([prok_corr_digrams[dg] for dg in canonical_digrams])
    prok_yerr = [1.96*sqrt(1.0/prok_N*p*(1-p)) for p in prok_ps]
    prok_adj_yerr = [1.96*sqrt(1.0/prok_adj_N*p*(1-p)) for p in prok_adj_ps]
    prok_corr_yerr = [1.96*sqrt(1.0/prok_corr_N*p*(1-p)) for p in prok_corr_ps]

    euk_digrams = defaultdict(int)
    euk_corr_digrams = defaultdict(int)
    euk_adj_digrams = defaultdict(int)
    for tests, motif in tqdm(zip(euk_tests, euk_motifs)):
        for test, ((i,coli),(j,colj)) in zip(tests, choose2(list(enumerate(transpose((motif)))))):
            for bi,bj in transpose((coli,colj)):
                rev_comp = tuple(wc((bi,bj)))
                if (bi, bj) > rev_comp:
                    bi, bj = rev_comp
                euk_digrams[(bi,bj)] += 1
                if j == i + 1:
                    euk_adj_digrams[(bi,bj)] += 1
                if test <= euk_q:
                    euk_corr_digrams[(bi,bj)] += 1
    euk_corr_N = float(sum(euk_corr_digrams.values()))
    euk_adj_N = float(sum(euk_adj_digrams.values()))
    euk_N = float(sum(euk_digrams.values()))
    # euk_ps = normalize(euk_digrams.values())
    # euk_adj_ps = normalize(euk_adj_digrams.values())
    # euk_corr_ps = normalize(euk_corr_digrams.values())
    euk_ps = normalize([euk_digrams[dg] for dg in canonical_digrams])
    euk_adj_ps = normalize([euk_adj_digrams[dg] for dg in canonical_digrams])
    euk_corr_ps = normalize([euk_corr_digrams[dg] for dg in canonical_digrams])
    euk_yerr = [1.96*sqrt(1.0/euk_N*p*(1-p)) for p in euk_ps]
    euk_adj_yerr = [1.96*sqrt(1.0/euk_adj_N*p*(1-p)) for p in euk_adj_ps]
    euk_corr_yerr = [1.96*sqrt(1.0/euk_corr_N*p*(1-p)) for p in euk_corr_ps]

    palette = sns.cubehelix_palette(4)
    ax = plt.subplot(211)
    # plt.bar(range(16),normalize(prok_digrams.values()))
    # plt.bar(range(16),normalize(prok_corr_digrams.values()),color='g')
    # plt.bar([x-0.2 for x in range(16)], prok_relative_ratios.values(), color='g', label="Correlated Column-pairs",width=0.2)
    # plt.bar([x for x in range(16)],prok_adj_relative_ratios.values(),color='r',alpha=1,yerr=prok_adj_yerr,label="Adjacent Column-pairs",width=0.2)
    # plt.bar([x+0.2 for x in range(16)],[1]*16,color='b',alpha=1,yerr=(prok_yerr),capsize=10,capstyle='butt',label="All Column-pairs",width=0.2)
    plt.bar([x-0.2 for x in range(len(canonical_digrams))], prok_ps, label="All Column-Pairs",width=0.2,yerr=prok_yerr,color=palette[0])
    plt.bar([x for x in range(len(canonical_digrams))],prok_adj_ps,label="Adj. Column-Pairs",
            width=0.2,yerr=prok_adj_yerr,color=palette[1])
    plt.bar([x+0.2 for x in range(len(canonical_digrams))],prok_corr_ps,alpha=1,
            capstyle='butt',label="Corr. Adj. Column-Pairs",width=0.2,yerr=prok_corr_yerr,color=palette[3])
    #plt.plot([0,16],[1.0/16, 1.0/16],linestyle='--',color=palette[3],label="Equiprobability",linewidth=1)
    ax.set_xticks([x for x in range(len(canonical_digrams))])
    ax.set_xticklabels( ["".join(dg) for dg in canonical_digrams],fontsize='large')
    plt.xlim(-0.5,10.5)
    plt.ylim(0,0.3)
    #plt.xlabel("Dimer",fontsize='large')
    plt.ylabel("Prokaryotic Frequency",fontsize='large')
    #plt.ylim(0,2)
    plt.legend(loc='upper right')
    
    ax2 = plt.subplot(212)
    #plt.plot([0,16],[1.0/16, 1.0/16],linestyle='--',color=palette[3],label="Equiprobability",linewidth=1)
    plt.bar([x-0.2 for x in range(len(canonical_digrams))], euk_ps, label="All Column-Pairs",width=0.2,yerr=euk_yerr,color=palette[0])
    plt.bar([x for x in range(len(canonical_digrams))],euk_adj_ps,label="Adj. Column-Pairs",
            width=0.2,yerr=euk_adj_yerr,color=palette[1])
    plt.bar([x+0.2 for x in range(len(canonical_digrams))],euk_corr_ps,alpha=1,
            capstyle='butt',label="Corr. Adj. Column-Pairs",width=0.2,yerr=euk_corr_yerr,color=palette[3])
    ax2.set_xticks([x for x in range(len(canonical_digrams))])
    ax2.set_xticklabels( ["".join(dg) for dg in canonical_digrams],fontsize='large')
    #plt.xlabel("Dimer",fontsize='large')
    plt.xlim(-0.5,10.5)
    plt.ylim(0,0.2)
    plt.ylabel("Eukaryotic Frequency",fontsize='large')
    #plt.ylim(0,2)
    plt.legend(loc='upper right')
    maybesave(filename)

def mean_occupancy(motif):
    return mean(occupancies(motif))
    
def mi_sampling_experiment():
    prok_maxent_motifs = [spoof_maxent_motifs(motif,100) for motif in tqdm(prok_motifs)]
    prok_cftp_motifs = [spoof_motif_cftp_occ(motif,10) for motif in tqdm(prok_motifs)]
    motif_mi_nc = lambda m:motif_mi(m,correct=False)
    scatter(map(motif_mi, prok_motifs), [mean(map(motif_mi,spoofs)) for spoofs in tqdm(prok_maxent_motifs)])
    scatter(map(motif_mi, prok_motifs), [mean(map(motif_mi,spoofs)) for spoofs in tqdm(prok_cftp_motifs)])

def mi_per_col(motif):
    L = len(motif[0])
    return motif_mi(motif)/choose(L,2)

def mi_gradient_experiment():
    """what parameters increase MI?"""
    L = 10
    sigma = 1
    N = 100
    cf = 10 # copy factor
    Ne = 5
    dL = 1
    dsigma = 0.1
    dcf = 5
    dNe = 1
    dN = 10
    def sample_mis(L, sigma, copy_factor, Ne, N, trials=100):
        mis = []
        for _ in trange(trials):
            matrix = sample_matrix(L, sigma)
            copies = copy_factor * N
            mu = approx_mu(matrix, copies)
            motif = sample_motif_cftp(matrix, mu, Ne, N)
            mis.append(motif_mi(motif))
        return mis
    # original_mis = sample_mis(L, sigma, copy_factor, Ne, N, trials=100)
    # sigma_mis = (sample_mis(L, sigma - dsigma, copy_factor, Ne, N, trials=50),
    #              sample_mis(L, sigma + dsigma, copy_factor, Ne, N, trials=50))
    # cf_mis = (sample_mis(L, sigma, copy_factor - dcf, Ne, N, trials=50),
    #              sample_mis(L, sigma, copy_factor + dcf, Ne, N, trials=50))
    # Ne_mis = (sample_mis(L, sigma , copy_factor, Ne - dNe, N, trials=50),
    #              sample_mis(L, sigma, copy_factor, Ne + dNe, N, trials=50))
    # sigma_prime = (mean(sigma_mis[1]) - mean(sigma_mis[0]))/dsigma
    # cf_prime = (mean(cf_mis[1]) - mean(cf_mis[0]))/dcf
    # Ne_prime = (mean(Ne_mis[1]) - mean(Ne_mis[0]))/dNe
    # def motifs_ij(i,j):
    #     def f(dhi,dhj):
    #         return sample_motifs(L + dL*dhi*(i==0))
    #     return sample_mis(L + dL)
    # hessian = [[]]
    f = lambda sigma, cf, Ne:mean(sample_mis(L, sigma, cf, Ne, N, trials=25))
    xs = [sigma, cf, Ne]
    dxs = [dsigma, dcf, dNe]
    hess = hessian(f, xs, dxs)

def hessian(f, xs, dxs):
    """compute hessian H(f)(x), using numerical differences dx"""
    N = len(dxs)
    def dij(i, j):
        if i == j:
            dx = dxs[i]
            xsm = [x - dx*(k==i) for k, (x,h) in enumerate(zip(xs,dxs))]
            xsp = [x + dx*(k==i) for k, (x,h) in enumerate(zip(xs,dxs))]
            return (-f(*xsm) +2*f(*xs) - f(*xsp))/(dx**2)
        else:
            dxi, dxj = dxs[i], dxs[j]
            xspp = [x + dxi*(k==i) + dxj*(k==j) for k, (x,h) in enumerate(zip(xs,dxs))]
            xspm = [x + dxi*(k==i) - dxj*(k==j) for k, (x,h) in enumerate(zip(xs,dxs))]
            xsmp = [x - dxi*(k==i) + dxj*(k==j) for k, (x,h) in enumerate(zip(xs,dxs))]
            xsmm = [x - dxi*(k==i) - dxj*(k==j) for k, (x,h) in enumerate(zip(xs,dxs))]
            return (f(*xspp) - f(*xspm) - f(*xsmp) + f(*xsmm))/(dxi*dxj)
    M = [[dij(i,j) if j >=i else None for j in range(N)] for i in range(N)]
    for i in range(N):
        for j in range(N):
            if M[i][j] is None:
                M[i][j] = M[j][i]
    return M
    

def mi_maximization_experiment(iterations=1000):
    L = 10
    N = 100
    des_ic = 10
    epsilon = 0.1
    beta = 1
    def sample_motif((sigma, cf, Ne)):
        matrix = sample_matrix(L, sigma)
        mu = approx_mu(matrix, cf * N)
        return sample_motif_cftp(matrix, mu, Ne, N)
    def f((sigma, cf, Ne), trials=1):
        motifs = [sample_motif((sigma, cf, Ne)) for trial in range(trials)]
        return exp(mean(beta * motif_mi(motif) - abs(motif_ic(motif) - des_ic) for motif in motifs))
    def max0(x):
        return max(x,0.01)
    def prop((sigma, cf, Ne)):
        new_params = (sigma + random.gauss(0,1), cf + random.gauss(0,1), Ne + random.gauss(0,1))
        return tuple(map(max0, new_params))
    x = [1,10,5]
    #chain = mh(f,prop,x0, iterations=iterations, verbose=True, cache=False)
    chain = []
    motifs = []
    mis = []
    ics = []
    acceptances = 0
    for i in trange(iterations):
        xp = prop(x)
        motif_p = sample_motif(xp)
        motif = sample_motif(x)
        ic = motif_ic(motif)
        mi = motif_mi(motif)
        fx = exp(beta*mi - abs(ic-des_ic))
        ic_p = motif_ic(motif_p)
        mi_p = motif_mi(motif_p)
        fx_p = exp(beta*mi_p - abs(ic_p-des_ic))
        print "fx:", fx, "fx_p:", fx_p
        if random.random() < fx_p/fx:
            x = xp
            motif = motif_p
            acceptances += 1
            ic = ic_p
            mi = mi_p
        chain.append(x)
        motifs.append(motif)
        ics.append(ic)
        mis.append(mi)
        print x, fx, "AR:", acceptances/float(i+1)
        print "IC:", ic, "MI:", mi
    return chain, motifs, ics, mis

def subsample(motif):
    if len(motif) < 200:
        return motif
    else:
        return sample(200, motif, replace=False)

def mi_per_col(motif):
    L = motif[0]
    return motif_mi(motif)/choose(len(L), 2)
    
def grand_spoofing_experiment(prok_motifs, euk_motifs):
    # should we subsample once or each time??
    prok_maxent_spoofs = [spoof_maxent_motifs(motif,10) for motif in tqdm(prok_motifs)]
    euk_maxent_spoofs = [spoof_maxent_motifs(subsample(motif), 10) for motif in tqdm(euk_motifs)]
    prok_cftp_spoofs = [spoof_motif_cftp_occ(motif,10) for motif in tqdm(prok_motifs)]
    euk_cftp_spoofs = [spoof_motif_cftp_occ(subsample(motif),10) for motif in tqdm(euk_motifs)]
    prok_oo_spoofs = [spoof_oo_motifs(motif,10) for motif in tqdm(prok_motifs)]
    prok_oo_occ_spoofs = [spoof_oo_motifs_occ(motif,10) for motif in tqdm(prok_motifs)]
    euk_oo_spoofs = [spoof_oo_motifs(subsample(motif),10) for motif in tqdm(euk_motifs)]
    euk_oo_occ_spoofs = [spoof_oo_motifs_occ(motif,10) for motif in tqdm(euk_motifs)]
    with open("prok_maxent_spoofs",'w') as f:
        cPickle.dump(prok_maxent_spoofs, f)
    with open("euk_maxent_spoofs",'w') as f:
        cPickle.dump(euk_maxent_spoofs, f)
    with open("prok_cftp_spoofs",'w') as f:
        cPickle.dump(prok_cftp_spoofs, f)
    with open("euk_cftp_spoofs",'w') as f:
        cPickle.dump(euk_cftp_spoofs, f)
    with open("prok_oo_spoofs",'w') as f:
        cPickle.dump(prok_oo_spoofs, f)
    with open("euk_oo_spoofs",'w') as f:
        cPickle.dump(euk_oo_spoofs, f)

    with open("prok_maxent_spoofs.pkl") as f:
        prok_maxent_spoofs = cPickle.load(f)
    with open("euk_maxent_spoofs.pkl") as f:
        euk_maxent_spoofs = cPickle.load(f)
    with open("prok_cftp_spoofs") as f:
        prok_cftp_spoofs = cPickle.load(f)
    with open("euk_cftp_spoofs") as f:
        euk_cftp_spoofs = cPickle.load(f)
    with open("prok_oo_spoofs.pkl") as f:
        prok_oo_spoofs = cPickle.load(f)
    with open("euk_oo_spoofs.pkl") as f:
        euk_oo_spoofs = cPickle.load(f)

    prok_mis = map(mi_per_col, prok_motifs)
    prok_maxent_mis = [mean(map(mi_per_col, spoofs)) for spoofs in tqdm(prok_maxent_spoofs)]
    euk_mis = map(mi_per_col, map(subsample,euk_motifs))
    euk_maxent_mis = [mean(map(mi_per_col, spoofs)) for spoofs in tqdm(euk_maxent_spoofs)]
    prok_cftp_mis = [mean(map(mi_per_col, spoofs)) for spoofs in tqdm(prok_cftp_spoofs)]
    euk_cftp_mis = [mean(map(mi_per_col, spoofs)) for spoofs in tqdm(euk_cftp_spoofs)]
    prok_oo_mis = [mean(map(mi_per_col, spoofs)) for spoofs in tqdm(prok_oo_spoofs)]
    euk_oo_mis = [mean(map(mi_per_col, spoofs)) for spoofs in tqdm(euk_oo_spoofs)]
    
    plt.subplot(1,3,1)
    scatter(prok_maxent_mis,
            prok_mis)
    plt.xlabel("Predicted MI",fontsize='large')
    plt.ylabel("Observed MI",fontsize='large')
    plt.title("MaxEnt",fontsize='large')
    scatter(euk_maxent_mis,
            euk_mis,color='g')
    plt.subplot(1,3,2)
    scatter(prok_cftp_mis,
            prok_mis)
    scatter(euk_cftp_mis,
            euk_mis,color='g')
    plt.xlabel("Predicted MI",fontsize='large')
    plt.ylabel("Observed MI",fontsize='large')
    plt.title("Gaussian Linear Ensemble",fontsize='large')
    plt.subplot(1,3,3)
    scatter(prok_oo_mis,
            prok_mis)
    scatter(euk_oo_mis,
            euk_mis,color='g')
    plt.xlabel("Predicted MI",fontsize='large')
    plt.ylabel("Observed MI",fontsize='large')
    plt.title("Match-Mismatch",fontsize='large')
    plt.tight_layout()
    maybesave("mi-spoof-plot.eps")

def spoofing_occupancy_vs_ic_experiment():
    """should we spoof motifs based on IC or occupancy?"""
    with open("prok_maxent_spoofs") as f:
        prok_maxent_spoofs = cPickle.load(f)
    with open("euk_maxent_spoofs") as f:
        euk_maxent_spoofs = cPickle.load(f)
    with open("prok_cftp_spoofs") as f:
        prok_cftp_occ_spoofs = cPickle.load(f)
    with open("euk_cftp_spoofs") as f:
        euk_cftp_occ_spoofs = cPickle.load(f)
    with open("prok_oo_spoofs") as f:
        prok_oo_spoofs = cPickle.load(f)
    with open("euk_oo_spoofs") as f:
        euk_oo_spoofs = cPickle.load(f)
    prok_oo_occ_spoofs = [spoof_oo_motifs_occ(motif,10) for motif in tqdm(prok_motifs)]
    euk_oo_occ_spoofs = [spoof_oo_motifs_occ(motif,10) for motif in tqdm(euk_motifs)]
    prok_cftp_spoofs = [spoof_motif_cftp(motif,10) for motif in tqdm(prok_motifs)]
    euk_cftp_spoofs = [spoof_motif_cftp(subsample(motif),10) for motif in tqdm(euk_motifs)]
    
def restriction_of_range_half_site_experiment(motif):
    """is energy of first half-site negatively correlated with energy of second half-site?"""
    L = len(motif[0])
    l = L/2
    mat = matrix_from_motif(motif)
    eps1 = [score_seq(mat[:l], site[:l]) for site in motif]
    eps2 = [score_seq(mat[l:], site[l:]) for site in motif]
    return pearsonr(eps1,eps2)

def restriction_of_range_loo_experiment(motif):
    """can energy of a given position be predicted from energy of remaining bases?"""
    L = len(motif[0])
    mat = matrix_from_motif(motif)
    eps = [score_seq(mat,site) for site in motif]
    mean_ep = mean(eps)
    results = []
    for j in range(L):
        print j
        loo_mat = mat[:j] + mat[j+1:]
        for site in motif:
            loo_ep = score_seq(loo_mat,site[:j] + site[j+1:])
            pred_ep = mean_ep - loo_ep
            obs_ep = score_seq([mat[j]],[site[j]])
            results.append((pred_ep, obs_ep))
    return results

def restriction_of_range_motif_spoof_experiment(motifs):
    all_eps = []
    all_spoof_eps = []
    for motif in tqdm(motifs):
        mat = matrix_from_motif(motif)
        eps = [score_seq(mat, site) for site in motif]
        spoofs = spoof_psfm(motif, pc=0)
        spoof_eps = [score_seq(mat, site) for site in spoofs]
        all_eps.append(eps)
        all_spoof_eps.append(spoof_eps)
    return all_eps, all_spoof_eps
    
def analyze_correlation_positions(all_tests, alpha="fdr"):
    if alpha == "fdr":
        alpha = fdr(concat(all_tests))
    print "alpha:",alpha
    ds = []
    d_controls = []
    for tests in all_tests:
        K = len(tests)
        L = find(lambda l:round(choose(l,2))==K, range(50))
        if L is None:
            print K
            raise Exception()
        for k, (i,j) in enumerate(choose2(range(L))):
            if j == i + 1 and tests[k] <= alpha:
                d = i/float(L)
                ds.append(d)
                d_controls.append(random.randrange(L-1)/float(L))
                plt.scatter(d, tests[k])
    return ds, d_controls

def sine_wave(motif):
    L = float(len(motif[0]))
    ics = columnwise_ic(motif,correct=False)
    sins = [sin(i/L*2*pi) for i in range(int(L))]

def score_sd(motif):
    matrix = matrix_from_motif(motif)
    eps = [score_seq(matrix, site) for site in motif]
    return sd(eps)

def motif_gc(motif):
    N, L = len(motif), len(motif[0])
    gc = 0
    for site in motif:
        for b in site:
            if b == 'G' or b == 'C':
                gc += 1
    return gc / float(N*L)

def schematic_slide():
    fit = lambda ep:1/(1+exp(ep+10))
    plt.plot(*pl(fit, np.linspace(-20,10, 1000)), label='Fitness', linewidth=5)
    plt.plot(*pl(lambda ep:5*dnorm(ep, 0,5), np.linspace(-20, 10, 1000)), label='Site Density', linewidth=5)
    plt.plot(*pl(lambda ep:fit(ep)*50*dnorm(ep, 0,5), np.linspace(-20, 10, 1000)), label='Fixation Probability', linewidth=5)
    plt.plot([-10, -10], [0,1], linestyle='--')
    plt.xlabel("$\epsilon(s)\ \ \  (K_{B}T)$", fontsize='large')
    plt.legend(fontsize='large')
