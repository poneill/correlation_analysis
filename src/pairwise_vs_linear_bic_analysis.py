from adjacent_pairwise_model import code_from_motif as pairwise_model_from_motif
from adjacent_pairwise_model import prob_site as pw_prob_site
from adjacent_pairwise_model import sample_site as pw_sample_site
from adjacent_pairwise_model import sample_code
from pwm_utils import psfm_from_motif as linear_model_from_motif
from utils import prod, scatter, cv, mean, transpose, sign, random_site
from math import log, pi
import sys
from formosa import maxent_motifs
from matplotlib import pyplot as plt
from utils import sorted_indices, rslice, roc_curve
from tqdm import *
import seaborn as sns

log10 = lambda x:log(x,10)

def linear_prob_site(site, psfm):
    return prod([col["ACGT".index(b)] for col,b in zip(psfm,site)])


def model_comparison(motif,crit="BIC"):
    assert crit in ["AIC", "BIC"]
    L = len(motif[0])
    N = len(motif)
    pw_model = pairwise_model_from_motif(motif)
    li_model = linear_model_from_motif(motif)
    pw_ll = sum(log(pw_prob_site(site,pw_model)) for site in motif)
    li_ll = sum(log(linear_prob_site(site,li_model)) for site in motif)
    pw_k = 9 * (L-1)
    li_k = 3 * L
    if crit == "BIC":
        pw_bic = -2 * pw_ll + pw_k * (log(N) - log(2*pi))
        li_bic = -2 * li_ll + li_k * (log(N) - log(2*pi))
        return pw_bic, li_bic
    elif crit == "AIC":
        pw_aic = -2*(pw_ll) + 2*pw_k
        li_aic = -2*(li_ll) + 2*li_k
        return pw_aic, li_aic

def euk_model_comparison():
    sys.path.append("/home/pat/jaspar")
    from parse_jaspar import jaspar_motifs as euk_motifs
    euk_comps = [model_comparison(motif) for motif in tqdm(euk_motifs)]
    pw_bics, li_bics = transpose(euk_comps)
    Ns = map(len,euk_motifs)
    euk_markers = [{1:'o',2: '|', 3:'^',4:'s',5:'p'}[int(log10(N))] for N in Ns]
    #plt.scatter(li_bics, pw_bics,marker=euk_markers)
    for x, y, m in zip(li_bics, pw_bics, euk_markers):
        plt.scatter(x,y,marker=m)
    plt.plot([1,10**6],[1,10**6],linestyle='--')
    plt.xlabel("Linear BIC")
    plt.ylabel("Pairwise BIC")
    plt.loglog()

def bic_analysis(motifs):
    comps = [model_comparison(motif) for motif in tqdm(motifs)]
    pw_bics, li_bics = transpose(comps)
    Ns = map(len,motifs)
    markers = [{1:'o',2: '|', 3:'^',4:'s',5:'p'}[int(log10(N))] for N in Ns]
    #plt.scatter(li_bics, pw_bics,marker=euk_markers)
    for x, y, m in zip(li_bics, pw_bics, markers):
        plt.scatter(x,y,marker=m)
    minval, maxval = min(pw_bics + li_bics), max(pw_bics + li_bics)
    plt.plot([minval, maxval],[minval, maxval],linestyle='--')
    plt.xlabel("Linear BIC")
    plt.ylabel("Pairwise BIC")
    plt.loglog()
    
def prok_model_comparison():
    sys.path.append("/home/pat/motifs")
    from parse_tfbs_data import tfdf
    prok_motifs = [getattr(tfdf,tf) for tf in tfdf.tfs]
    prok_comps = [model_comparison(motif) for motif in tqdm(prok_motifs)]
    pw_bics, li_bics = transpose(prok_comps)
    scatter(li_bics, pw_bics)
    plt.xlabel("Linear BIC")
    plt.ylabel("Pairwise BIC")
    plt.loglog()

def cv_analysis(motif):
    """given motif, return (linear ll, pairwise ll) for folds of 10-fold cv"""
    lls = []
    for train, test in cv(motif):
        pw_model = pairwise_model_from_motif(train)
        li_model = linear_model_from_motif(train)
        pw_ll = sum(log(pw_prob_site(site,pw_model)) for site in test)
        li_ll = sum(log(linear_prob_site(site,li_model)) for site in test)
        lls.append((li_ll,pw_ll))
    return lls

def cv_experiment(motifs):
    all_lis, all_pws = [], []
    for motif in tqdm(motifs):
        lls = cv_analysis(motif)
        lis, pws = transpose(lls)
        all_lis.append(mean(lis))
        all_pws.append(mean(pws))
    Ns = map(len,motifs)
    markers = [{1:'o',2: '|', 3:'^',4:'s',5:'p'}[int(log10(N))] for N in Ns]
    for x, y, m in zip(all_lis, all_pws, markers):
        plt.scatter(x, y, marker=m)
    minval, maxval = min(all_lis + all_pws),max(all_lis + all_pws)
    plt.plot([minval, maxval],[minval, maxval])
    plt.xlabel("Linear LL")
    plt.ylabel("Pairwise LL")
        

def sanity_check():
    pw_motifs = [(lambda code:[pw_sample_site(code) for i in range(100)])(sample_code(10,1))
                 for _ in range(100)]
    li_motifs = maxent_motifs(100,10,10,100)
    cv_experiment(pw_motifs)
    cv_experiment(li_motifs)

def sample_size_dependence_experiment():
    """is prok=linear, euk=pairwise result merely due to sample size?  plot by N to find out."""
    prok_bics = [model_comparison(motif,crit="BIC") for motif in tqdm(prok_motifs)]
    euk_bics = [model_comparison(motif,crit="BIC") for motif in tqdm(euk_motifs)]
    prok_aics = [model_comparison(motif,crit="AIC") for motif in tqdm(prok_motifs)]
    euk_aics = [model_comparison(motif,crit="AIC") for motif in tqdm(euk_motifs)]
    prok_cvs = [cv_analysis(motif) for motif in tqdm(prok_motifs)]
    euk_cvs = [cv_analysis(motif) for motif in tqdm(euk_motifs)]
    bic_difs = [x - y for (x,y) in prok_bics + euk_bics]
    aic_difs = [x - y for (x,y) in prok_aics + euk_aics]
    cv_difs = [(mean(x-y for (x,y) in lls)) for lls in prok_cvs + euk_cvs]
    prok_Ns = map(len,prok_motifs)
    euk_Ns = map(len,euk_motifs)
    N_colors = [sns.cubehelix_palette(5)[int(round(log10(len(motif))))] for motif in (prok_motifs + euk_motifs)]

    palette2 = sns.cubehelix_palette(5)
    prok_color = palette2[4]
    euk_color = palette2[1]
    colors = [prok_color for _ in prok_motifs] + [euk_color for _ in euk_motifs]

    sns.set_style('darkgrid')
    plt.subplot(3,1,1)
    plt.ylabel("<- Pairwise Better ($\Delta$ BIC) Linear Better ->")
    plt.scatter(prok_Ns,[abslog10(x-y) for (x,y) in (prok_bics)],label="Prokaryotic Motifs",marker='o',
                color=prok_color)
    plt.scatter(euk_Ns,[abslog10(x-y) for (x,y) in (euk_bics)],label="Eukaryotic Motifs",marker='s',color=euk_color)
    # plt.scatter(prok_Ns,[(x-y)/N for (x,y),N in zip(prok_bics,prok_Ns)],label="Prokaryotic Motifs",marker='o',
    #              color=prok_color)
    # plt.scatter(euk_Ns,[(x-y)/N for (x,y),N in zip(euk_bics, euk_Ns)],label="Eukaryotic Motifs",marker='s',
    #             color=euk_color)
    max_N = max(prok_Ns + euk_Ns)
    plt.plot([1,max_N],[0,0],linestyle='--',color='black')
    plt.legend()
    #plt.scatter(prok_Ns+euk_Ns,[abslog(x-y) for (x,y) in (prok_bics + euk_bics)],color=colors)
    plt.semilogx()

    plt.subplot(3,1,2)
    plt.ylabel("<- Pairwise Better ($\Delta$ AIC) Linear Better ->")
    plt.scatter(prok_Ns,[abslog10(x-y) for (x,y) in prok_aics],label="Prokaryotic Motifs",marker='o',
                 color=prok_color)
    plt.scatter(euk_Ns,[abslog10(x-y) for (x,y) in euk_aics],label="Eukaryotic Motifs",marker='s',
                color=euk_color)
    plt.legend()
    plt.plot([1,max(prok_Ns + euk_Ns)],[0,0],linestyle='--',color='black')
    #plt.scatter(prok_Ns+euk_Ns,[abslog(x-y) for (x,y) in (prok_aics + euk_aics)],color=colors)
    plt.semilogx()

    plt.subplot(3,1,3)
    plt.ylabel("<- Pairwise Better ($\Delta$ CV LL) Linear Better ->")
    plt.scatter(prok_Ns,map(abslog10,cv_difs[:len(prok_motifs)]),
                label="Prokaryotic Motifs",marker='o', color=prok_color)
    plt.scatter(euk_Ns,map(abslog10,cv_difs[len(prok_motifs):]),
                label="Eukaryotic Motifs",marker='o', color=euk_color)
    #plt.ylim(-30,30)
    plt.legend()
    plt.plot([1,max(prok_Ns + euk_Ns)],[0,0],linestyle='--',color='black')
    plt.semilogx()
    maybesave("aic_bic_cv_comparison.eps")

    plt.scatter(map(abslog10,cv_difs), map(abslog10,bic_difs),color=colors)
    plt.plot([0,0],[-5,5],linestyle='--',color='black')
    plt.plot([-5,5],[0,0],linestyle='--',color='black')
    plt.plot([-5,5],[-5,5],linestyle='--',color='black')
    plt.xlim(-5,5)
    plt.ylim(-5,5)
    plt.xlabel("<- Pairwise Better ($\Delta$ CV) Linear Better ->")
    plt.ylabel("<- Pairwise Better ($\Delta$ BIC) Linear Better ->")
    
    prok_bic_linear_better = count(lambda x:x>0,bic_difs[:len(prok_motifs)])
    prok_cv_linear_better = count(lambda x:x>0,cv_difs[:len(prok_motifs)])
    prok_bic_cv_linear_better = count(lambda (b,c):b>0 and c > 0,zip(bic_difs,cv_difs)[:len(prok_motifs)])
    euk_bic_linear_better = count(lambda x:x>0,bic_difs[len(prok_motifs):])
    euk_cv_linear_better = count(lambda x:x>0,cv_difs[len(prok_motifs):])
    euk_bic_cv_linear_better = count(lambda (b,c):b>0 and c > 0,zip(bic_difs,cv_difs)[len(prok_motifs):])
        
    
def abslog(x):
    return sign(x) * log(abs(x))

def abslog10(x):
    return sign(x) * log10(abs(x))

def roc_experiment(motif, trials=10**5):
    pw_model = pairwise_model_from_motif(motif)
    li_model = linear_model_from_motif(motif)
    L = len(motif[0])
    negatives = [random_site(L) for i in trange(trials)]
    pw_pos = [pw_prob_site(site,pw_model) for site in motif]
    pw_neg = [pw_prob_site(site,pw_model) for site in tqdm(negatives)]
    li_pos = [linear_prob_site(site,li_model) for site in motif]
    li_neg = [linear_prob_site(site,li_model) for site in tqdm(negatives)]
    _, _, _, pw_auc = roc_curve(pw_pos, pw_neg)
    _, _, _, li_auc = roc_curve(li_pos, li_neg,color='g')
    return li_auc, pw_auc
