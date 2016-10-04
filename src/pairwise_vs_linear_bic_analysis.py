from adjacent_pairwise_model import code_from_motif as pairwise_model_from_motif
from adjacent_pairwise_model import prob_site as pw_prob_site
from adjacent_pairwise_model import prob_sites as pw_prob_sites
from adjacent_pairwise_model import sample_site as pw_sample_site
from adjacent_pairwise_model import sample_code
from pwm_utils import psfm_from_motif as linear_model_from_motif
from pwm_utils import sample_matrix
from utils import prod, scatter, cv, mean, transpose, sign, random_site, mutate_motif
from utils import random_motif, mh, score_seq, motif_ic, subst, inverse_cdf_sample
from math import log, pi, exp
import sys
from formosa import maxent_motifs, spoof_maxent_motifs
from matplotlib import pyplot as plt
from utils import sorted_indices, rslice, roc_curve
from utils import total_motif_mi as motif_mi
from tqdm import *
import seaborn as sns
from evo_sampling import sample_motif_cftp
import random
import matplotlib as mpl
import numpy as np
log10 = lambda x:log(x,10)

def linear_prob_site(site, psfm):
    return prod([col["ACGT".index(b)] for col,b in zip(psfm,site)])


def model_comparison(motif,crit="BIC"):
    assert crit in ["AIC","AIC_dep", "AICc", "BIC"]
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
    elif crit == "AIC_dep": # multivariate adjustment: seems way too strict!
        pw_aic = -2*(pw_ll) + 2*(pw_k * L + 1/2.0*L*(L+1))
        li_aic = -2*(li_ll) + 2*(li_k * L + 1/2.0*L*(L+1))
        return pw_aic, li_aic
    elif crit == "AIC":
        pw_aic = -2*(pw_ll) + 2*pw_k
        li_aic = -2*(li_ll) + 2*li_k
        return pw_aic, li_aic
    elif crit == "AICc":
        pw_aic = -2*(pw_ll) + 2*pw_k * (N/float(N-pw_k-1))
        li_aic = -2*(li_ll) + 2*li_k * (N/float(N-li_k-1))
        return pw_aic, li_aic

def is_suitable_linear(motif):
    N, L = len(motif), len(motif[0])
    return N > 3*L

def is_suitable_pairwise(motif):
    N, L = len(motif), len(motif[0])
    return N > 9 * (L-1)

def suitability_dif(motif):
    """quick and dirty model comparison test, signed to return positive
    value of linear model favored, otherwise pairwise value
    """
    return -2*int(is_suitable_pairwise(motif)) + 1

def suitability_ratio(motif):
    """return log ratio of parameters to data (positive favors linear, negative, pairwise)"""
    N, L = len(motif), len(motif[0])
    return log10(float(9*(L-1))/N)
    
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

def redo_sample_size_dependence(filename=None):
    palette2 = sns.cubehelix_palette(5)
    prok_color = palette2[4]
    euk_color = palette2[1]
    colors = [prok_color for _ in suitable_prok_motifs] + [euk_color for _ in suitable_euk_motifs]
    xmin = 1 * 5
    xmax = max_N * 2
    sns.set_style('white')
    #plt.ylabel("<- Pairwise Better ($\Delta$ BIC) Linear Better ->")
    plt.xlabel("log(N/p) for Pairwise Model",fontsize='large')
    plt.ylabel("logmod($\Delta$ BIC)",fontsize='large')
    euk_ys = [logmod(x-y) for (x,y) in (euk_bics)]
    prok_ys = [logmod(x-y) for (x,y) in (prok_bics)]
    ys = prok_ys + euk_ys
    plt.scatter(euk_xs,euk_ys,label="Eukaryotic Motifs",marker='s',color=euk_color, s=20)
    plt.scatter(prok_xs,prok_ys,label="Prokaryotic Motifs",marker='o',
                color=prok_color, s=20)
    plt.plot([0,0],[min(ys), max(ys)], linestyle='--',color='black')
    plt.plot([min(xs),max(xs)],[0,0], linestyle='--',color='black')
    # plt.scatter(prok_xs,[(x-y)/N for (x,y),N in zip(prok_bics,prok_xs)],label="Prokaryotic Motifs",marker='o',
    #              color=prok_color)
    # plt.scatter(euk_xs,[(x-y)/N for (x,y),N in zip(euk_bics, euk_xs)],label="Eukaryotic Motifs",marker='s',
    #             color=euk_color)
    #plt.plot([10,max_N],[0,0],linestyle='--',color='black')
    #plt.xlim(xmin,xmax)
    leg = plt.legend(frameon=True,
                     #bbox_to_anchor=(1.25,0.75),
                     loc='lower left',
                     fontsize='large',)
    maybesave(filename)
    
def sample_size_dependence_experiment():
    """is prok=linear, euk=pairwise result merely due to sample size?  plot by N to find out."""
    # suitable_prok_motifs = filter(lambda motif:is_suitable_pairwise(motif) and is_suitable_linear(motif), prok_motifs)
    # suitable_euk_motifs = filter(lambda motif:is_suitable_pairwise(motif) and is_suitable_linear(motif), euk_motifs)
    suitable_prok_motifs, suitable_euk_motifs = prok_motifs, euk_motifs
    use_suit_ratios = True
    if use_suit_ratios:
        prok_xs = map(lambda x:-suitability_ratio(x), suitable_prok_motifs)
        euk_xs = map(lambda x:-suitability_ratio(x), suitable_euk_motifs)
        xs = prok_xs + euk_xs
    else:
        prok_xs = map(len, suitable_prok_motifs)
        euk_xs = map(len, suitable_euk_motifs)
        xs = prok_xs + euk_xs
    prok_bics = [model_comparison(motif,crit="BIC") for motif in tqdm(suitable_prok_motifs)]
    euk_bics = [model_comparison(motif,crit="BIC") for motif in tqdm(suitable_euk_motifs)]
    prok_aics = [model_comparison(motif,crit="AIC") for motif in tqdm(suitable_prok_motifs)]
    euk_aics = [model_comparison(motif,crit="AIC") for motif in tqdm(suitable_euk_motifs)]
    prok_aiccs = [model_comparison(motif,crit="AICc") for motif in tqdm(suitable_prok_motifs)]
    euk_aiccs = [model_comparison(motif,crit="AICc") for motif in tqdm(suitable_euk_motifs)]
    prok_cvs = [cv_analysis(motif) for motif in tqdm(suitable_prok_motifs)]
    euk_cvs = [cv_analysis(motif) for motif in tqdm(suitable_euk_motifs)]
    bic_difs = [x - y for (x,y) in prok_bics + euk_bics]
    aic_difs = [x - y for (x,y) in prok_aics + euk_aics]
    aicc_difs = [x - y for (x,y) in prok_aiccs + euk_aiccs]
    cv_difs = [(mean(x-y for (x,y) in lls)) for lls in prok_cvs + euk_cvs]
    # prok_Ns = map(len,suitable_prok_motifs)
    # euk_Ns = map(len,suitable_euk_motifs)
    N_colors = [sns.cubehelix_palette(5)[int(round(log10(len(motif))))] for motif in (suitable_prok_motifs + suitable_euk_motifs)]
    max_N = max(prok_Ns + euk_Ns)
    
    mpl.rcParams['xtick.major.pad'] = 10
    mpl.rcParams['ytick.major.pad'] = 10
    palette2 = sns.cubehelix_palette(5)
    prok_color = palette2[4]
    euk_color = palette2[1]
    colors = [prok_color for _ in suitable_prok_motifs] + [euk_color for _ in suitable_euk_motifs]
    xmin = 1 * 5
    xmax = max_N * 2
    sns.set_style('darkgrid')
    plt.subplot(3,1,1)
    #plt.ylabel("<- Pairwise Better ($\Delta$ BIC) Linear Better ->")
    plt.ylabel("logmod($\Delta$ BIC)")
    euk_ys = [logmod(x-y) for (x,y) in (euk_bics)]
    prok_ys = [logmod(x-y) for (x,y) in (prok_bics)]
    ys = prok_ys + euk_ys
    plt.scatter(euk_xs,euk_ys,label="Eukaryotic Motifs",marker='s',color=euk_color)
    plt.scatter(prok_xs,prok_ys,label="Prokaryotic Motifs",marker='o',
                color=prok_color)
    plt.plot([0,0],[min(ys), max(ys)], linestyle='--',color='black')
    plt.plot([min(xs),max(xs)],[0,0], linestyle='--',color='black')
    # plt.scatter(prok_xs,[(x-y)/N for (x,y),N in zip(prok_bics,prok_xs)],label="Prokaryotic Motifs",marker='o',
    #              color=prok_color)
    # plt.scatter(euk_xs,[(x-y)/N for (x,y),N in zip(euk_bics, euk_xs)],label="Eukaryotic Motifs",marker='s',
    #             color=euk_color)
    #plt.plot([10,max_N],[0,0],linestyle='--',color='black')
    #plt.xlim(xmin,xmax)
    leg = plt.legend(frameon=True,bbox_to_anchor=(1,0.75),loc='center left')
    #plt.scatter(prok_xs+euk_xs,[abslog(x-y) for (x,y) in (prok_bics + euk_bics)],color=colors)
    #plt.semilogx()

    # plt.subplot(3,1,2)
    # #plt.ylabel("<- Pairwise Better ($\Delta$ AIC) Linear Better ->")
    # plt.ylabel("$\Delta$ AIC")
    # plt.scatter(euk_xs,[logmod(x-y) for (x,y) in euk_aics],label="Eukaryotic Motifs",marker='s',
    #             color=euk_color)
    # plt.scatter(prok_xs,[logmod(x-y) for (x,y) in prok_aics],label="Prokaryotic Motifs",marker='o',
    #              color=prok_color)
    # plt.xlim(xmin,xmax)
    # plt.plot([10,max(prok_xs + euk_xs)],[0,0],linestyle='--',color='black')
    # #plt.scatter(prok_xs+euk_xs,[abslog(x-y) for (x,y) in (prok_aics + euk_aics)],color=colors)
    # plt.semilogx()
    
    plt.subplot(3,1,2)
    #plt.ylabel("<- Pairwise Better ($\Delta$ AIC) Linear Better ->")
    plt.ylabel("logmod($\Delta$ AIC)")
    euk_ys = [logmod(x-y) for (x,y) in euk_aics]
    prok_ys =[logmod(x-y) for (x,y) in prok_aics]
    ys = prok_ys + euk_ys
    plt.scatter(euk_xs,euk_ys,label="Eukaryotic Motifs",marker='s',
                color=euk_color)
    plt.scatter(prok_xs,prok_ys,label="Prokaryotic Motifs",marker='o',
                 color=prok_color)
    plt.plot([0,0],[min(ys), max(ys)], linestyle='--',color='black')
    plt.plot([min(xs),max(xs)],[0,0], linestyle='--',color='black')
    #plt.xlim(xmin,xmax)
    #plt.plot([10,max(prok_xs + euk_xs)],[0,0],linestyle='--',color='black')
    #plt.scatter(prok_xs+euk_xs,[abslog(x-y) for (x,y) in (prok_aics + euk_aics)],color=colors)
    #plt.semilogx()

    plt.subplot(3,1,3)
    #plt.ylabel("<- Pairwise Better ($\Delta$ CV LL) Linear Better ->")
    plt.ylabel("logmod($\Delta$ CV)")
    euk_ys = map(logmod,cv_difs[len(suitable_prok_motifs):])
    prok_ys = map(logmod,cv_difs[:len(suitable_prok_motifs)])
    ys = prok_ys + euk_ys
    plt.scatter(euk_xs,euk_ys,
                label="Eukaryotic Motifs",marker='s', color=euk_color)
    plt.scatter(prok_xs,prok_ys,
                label="Prokaryotic Motifs",marker='o', color=prok_color)
    plt.plot([0,0],[min(ys), max(ys)], linestyle='--',color='black')
    plt.plot([min(xs),max(xs)],[0,0], linestyle='--',color='black')
    #plt.ylim(-30,30)
    #plt.xlim(xmin,xmax)
    #plt.plot([10,max(prok_xs + euk_xs)],[0,0],linestyle='--',color='black')
    #plt.semilogx()
    plt.tight_layout()
    #maybesave("aic_bic_cv_comparison.eps")
    plt.xlabel("$\log_{10}(N/p)$ for Pairwise Model")
    #xxl=plt.xlabel("Motif Size")
    # xxl.set_position((xxl.get_position()[0],1)) # This says use the top of the bottom axis as the reference point.
    # xxl.set_verticalalignment('center')
    plt.savefig("aic_bic_cv_comparison.eps",bbox_extra_artists=(leg,),bbox_inches='tight')
    plt.close()

    # plt.scatter(map(abslog10,cv_difs), map(abslog10,bic_difs),color=colors)
    # plt.plot([0,0],[-5,5],linestyle='--',color='black')
    # plt.plot([-5,5],[0,0],linestyle='--',color='black')
    # plt.plot([-5,5],[-5,5],linestyle='--',color='black')
    # plt.xlim(-5,5)
    # plt.ylim(-5,5)
    # plt.xlabel("<- Pairwise Better ($\Delta$ CV) Linear Better ->")
    # plt.ylabel("<- Pairwise Better ($\Delta$ BIC) Linear Better ->")
    
    # prok_bic_linear_better = count(lambda x:x>0,bic_difs[:len(prok_motifs)])
    # prok_cv_linear_better = count(lambda x:x>0,cv_difs[:len(prok_motifs)])
    # prok_bic_cv_linear_better = count(lambda (b,c):b>0 and c > 0,zip(bic_difs,cv_difs)[:len(prok_motifs)])
    # euk_bic_linear_better = count(lambda x:x>0,bic_difs[len(prok_motifs):])
    # euk_cv_linear_better = count(lambda x:x>0,cv_difs[len(prok_motifs):])
    # euk_bic_cv_linear_better = count(lambda (b,c):b>0 and c > 0,zip(bic_difs,cv_difs)[len(prok_motifs):])
        
    
def abslog(x):
    return sign(x) * log(abs(x))

def abslog10(x):
    return sign(x) * log10(abs(x))

def logmod(x):
    return sign(x) * log10(abs(x) + 1)
    
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

def sample_pw_motif_mh(code, N, Ne, mu, iterations=50000):
    nu = Ne - 1
    def log_f(motif):
        eps = map(lambda x:-log(x),pw_prob_sites(motif, code))
        return sum(log(1/(1+exp(ep-mu))**nu) for ep in eps)
    prop = mutate_motif
    L = len(code) + 1
    x0 = random_motif(L, N)
    return mh(log_f,prop,x0,cache=True, use_log=True, iterations=iterations)
    
    
def degradation_experiment():
    """Determine whether linear or pairwise models are more resistant to degradation"""
    L = 10
    N = 50
    Ne = 5
    nu = Ne - 1
    sigma = 1
    mu = -10
    matrix = sample_matrix(L, sigma)
    code = sample_code(L, sigma)
    li_motif = sample_motif_cftp(matrix, mu, Ne, N)
    pw_motif = sample_pw_motif_mh(code, N, Ne, mu, iterations=100000)[-1]
    def li_log_fitness(motif):
        eps = [score_seq(matrix, site) for site in motif]
        return sum(-nu * log((1+exp(ep-mu))) for ep in eps)
    def pw_log_fitness(motif):
        eps = map(lambda x:-log(x),pw_prob_sites(motif, code))
        return sum(log(1/(1+exp(ep-mu))**nu) for ep in eps)
    li_base_fit = li_log_fitness(li_motif)
    li_mut_fits = [li_log_fitness(mutate_motif(li_motif)) for i in range(100)]
    pw_base_fit = pw_log_fitness(pw_motif)
    pw_mut_fits = [pw_log_fitness(mutate_motif(pw_motif)) for i in range(100)]

def estremo(iterations=50000, verbose=False, every=1, sigma=1, mu=-10, Ne=5):
    nu = Ne - 1
    def log_f((code, motif)):
        eps = map(lambda x:-log(x),pw_prob_sites(motif, code))
        return sum(nu * log(1/(1+exp(ep-mu))) for ep in eps)
    def prop((code, motif)):
        code_p = [d.copy() for d in code]
        i = random.randrange(len(code))
        b1, b2 = random.choice("ACGT"), random.choice("ACGT")
        code_p[i][(b1, b2)] += random.gauss(0,sigma)
        motif_p = mutate_motif(motif)
        return (code_p,motif_p)
    x0 = (sample_code(L=10,sigma=1), random_motif(length=10, num_sites=20))
    chain = mh(log_f, prop, x0, use_log=True, iterations=iterations, verbose=verbose,every=every)
    return chain

def estremo_gibbs(iterations=50000, verbose=False, every=1000, sigma=1, mu=-10, Ne=5):
    nu = Ne - 1
    L = 10
    N = 20
    code, motif = (sample_code(L=10,sigma=1), random_motif(length=L, num_sites=N))
    def log_f((code, motif)):
        eps = map(lambda x:-log(x),pw_prob_sites(motif, code))
        return sum(nu * log(1/(1+exp(ep-mu))) for ep in eps)
    chain = [(code, motif[:])]
    print log_f((code, motif))
    for iteration in trange(iterations):
        for i in range(N):
            site = motif[i]
            for j in range(L):
                b = site[j]
                log_ps = []
                bps = [bp for bp in "ACGT" if not bp == b]
                for bp in bps:
                    site_p = subst(site,bp,j)
                    log_ps.append(log_f((code,[site_p])))
                log_ps = [p-min(log_ps) for p in log_ps]
                bp = inverse_cdf_sample(bps,map(exp,log_ps),normalized=False)
                motif[i] = subst(site,bp,j)
        for k in range(L-1):
            for b1 in "ACGT":
                for b2 in "ACGT":
                    dws = [random.gauss(0,0.1) for _ in range(10)]
                    code_ps = [[d.copy() for d in code] for _ in range(10)]
                    for code_p, dw in zip(code_ps, dws):
                        code_p[k][b1,b2] += dw
                    log_ps = [log_f((code_p, motif)) for code_p in code_ps]
                    log_ps = [p-min(log_ps) for p in log_ps]
                    code_p = inverse_cdf_sample(code_ps, map(exp,log_ps),normalized=False)
                    code = code_p
        print log_f((code, motif))
        chain.append((code, motif[:]))
    return chain
                    
                        
                    
    x0 = (sample_code(L=10,sigma=1), random_motif(length=10, num_sites=20))
    chain = mh(log_f, prop, x0, use_log=True, iterations=iterations, verbose=verbose,every=every)
    return chain

def interpret_estremo_chain(chain,mu=-10,Ne=5):
    nu = Ne-1
    def log_f((code, motif)):
        eps = map(lambda x:-log(x),pw_prob_sites(motif, code))
        return sum(nu * log(1/(1+exp(ep-mu))) for ep in eps)
    spoofs = [spoof_maxent_motifs(motif,10) for code,motif in tqdm(chain)]
    plt.plot([motif_ic(motif) for (code, motif) in tqdm(chain)])
    plt.plot([motif_mi(motif) for (code, motif) in tqdm(chain)])
    plt.plot([mean(map(motif_ic,motifs)) for motifs in tqdm(spoofs)])
    plt.plot([mean(map(motif_mi,motifs)) for motifs in tqdm(spoofs)])
    plt.plot([indep_measure(code) for (code, motif) in tqdm(chain)])
    plt.plot(map(log_f,chain))

log2 = lambda x:log(x,2)

def dkl(mat):
    P = -np.exp(mat)/np.sum(-np.exp(mat))
    return sum(P[i,j]*log2(P[i,j]/(np.sum(P[i,:])*np.sum(P[:,j]))) for i in range(4) for j in range(4))
    
def indep_measure(code):
    mats = [np.matrix([[d[b1,b2] for b1 in "ACGT"] for b2 in "ACGT"]) for d in code]
    #return sum(np.linalg.norm((mat + mat.transpose())/2)/np.linalg.norm(mat) for mat in mats)
    return sum(map(dkl,mats))

