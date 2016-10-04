from utils import pairs,argmin,random_site,prod, mean, score_seq, maybesave, variance
from utils import scatter, normalize, inverse_cdf_sample, transpose, mean_ci, bs_ci
import random
from itertools import product
from math import exp,log
import numpy as np
from tqdm import *
from scipy.stats import pearsonr
from collections import Counter
from matplotlib import pyplot as plt

abet = "ACGT"

def kmers(L):
    return ("".join(kmer) for kmer in product(*[abet for i in range(L)]))
    
def sample_code(L,sigma):
    return [{(n1,n2):random.gauss(0,sigma) for n1 in abet for n2 in abet}
            for i in xrange(L-1)]

def best_binder_dep(code):
    L = len(code) + 1
    i = argmin([score(code,site) for site in kmers(L)])
    return list(kmers(L))[i]
    
def score(code, site):
    return sum(c[n1,n2] for c,(n1,n2) in zip(code,pairs(site)))

def occ_ref(sigma,L,G=5*10**6):
    code = sample_code(L,sigma)
    best_site = best_binder(code)
    ep = score(code,best_site)
    Z = sum(exp(-score(code,random_site(L))) for i in range(G))
    return exp(-ep)/(exp(-ep) + Z)

def compute_Zb(code):
    Ws = [np.matrix([[exp(interaction[b1,b2]) for b2 in "ACGT"] for b1 in "ACGT"])
          for interaction in code]
    return np.array([1,1,1,1]).dot(reduce(lambda x,y:x.dot(y),Ws)).dot(np.array([1,1,1,1]))[0,0]

def partial_Zb(code, partial_site):
    partial_site = partial_site + "X"*(len(code)+1-len(partial_site))
    def match(b,s):
        return b == s or s == 'X'
    def element(b1,b2,s1,s2):
        if match(b1,s1) and match(b2,s2):
            return exp(interaction[b1,b2])
        else:
            return 0
    Ws = [np.matrix([[element(b1,b2,s1,s2) for b2 in "ACGT"]
                     for b1 in "ACGT"])
          for interaction,(s1,s2) in zip(code,pairs(partial_site))]
    return np.array([1,1,1,1]).dot(reduce(lambda x,y:x.dot(y),Ws)).dot(np.array([1,1,1,1]))[0,0]
    
def test_compute_Zb(code):
    L = len(code) + 1
    Zb_ref = sum(exp(score(code,site)) for site in kmers(L))
    Zb = compute_Zb(code)
    return Zb,Zb_ref

def best_binder(code):
    """compute consensus sequence via dynamic programming."""
    L = len(code) + 1
    paths = {b:0 for b in "ACGT"} # best sites ending at base b so far
    for i,weight in enumerate(code):
        rel_len = i + 1 # at this point we consider paths of length i + 1
        rel_paths = filter(lambda path:len(path) == rel_len,paths.keys())
        for b2 in "ACGT":
            best_path = max(rel_paths,key=lambda path:paths[path] + weight[path[-1],b2])
            paths[best_path + b2] = paths[best_path] + weight[best_path[-1],b2]
    final_paths = filter(lambda path:len(path) == L,paths.keys())
    return max(final_paths,key=lambda path:paths[path])

def predict_best_binder():
    """explore expected energy of best binder"""
    def pred(L,sigma):
        alpha = 1.7666 # empirical estimation of 16th order statistic
        return -alpha*sigma*(L-1)
    Ls = range(2,10)
    sigmas = range(10)
    trials = 100
    for sigma in sigmas:
        plt.plot(*pl(lambda L:mean((lambda code:score(code,best_binder(code)))(sample_code(L,sigma))
                                   for i in range(trials)),Ls))
        #plt.plot(*pl(lambda L:pred(L,sigma),Ls),linestyle='--')

def test_best_binder(code):
    L = len(code) + 1
    pred_best = best_binder(code)
    obs_best = min(kmers(L),key = lambda site:score(code,site))
    return pred_best == obs_best
    
def max_occ(code,G=5*10**6):
    L = len(code) + 1
    site = best_binder(code)
    ep = score(code,site)
    Zb = G/float(4**L)*compute_Zb(code)
    #return exp(-ep)/(exp(-ep) + Zb)
    return 1/(1 + Zb*exp(ep))

def occ_mat(trials=100):
    Ls = range(2,31)
    sigmas = np.linspace(0,20,100)
    mat = [[mean(max_occ(sample_code(L,sigma)) for i in xrange(trials)) for L in Ls] for sigma in tqdm(sigmas)][::-1]
    return mat

def plot_occ_mat(mat,Ls=range(2,31),sigmas=np.linspace(0,20,100),filename=None):
    plt.imshow(mat,interpolation='nearest',extent=[2,31,0,20],aspect='auto')
    plt.xticks(Ls)
    #plt.yticks(["%1.2f" % x for x in sigmas[::-1]])
    plt.xlabel("L")
    plt.ylabel("$\sigma$")
    plt.title("Occupancy for Optimal Binding Site in Adjacent Pairwise Model")
    maybesave(filename)
    
def linearize(code):
    """return linear approximation of code"""
    L = len(code) + 1
    pssm = [[None]*4 for i in range(L)]
    pssm[0] = [mean(code[0][b1,b2] for b2 in "ACGT") for b1 in "ACGT"]
    pssm[-1] = [mean(code[-1][b1,b2] for b1 in "ACGT") for b2 in "ACGT"]
    for i, (w1,w2) in enumerate(pairs(code)):
        pssm[i+1] = [mean(w1[b1,b2] for b1 in "ACGT") + mean(w2[b2,b3]  for b3 in "ACGT") for b2 in "ACGT"]
    return pssm

def linearize_ref(code):
    L = len(code) + 1
    pssm = [[0]*4 for i in range(L)]
    for site in kmers(L):
        ep = score(code,site)
        for i,b in enumerate(site):
            pssm[i]["ACGT".index(b)] += ep
    #pssm = [[x/(4.0**(L-1)) for x in row] for row in pssm]
    pssm = [[x/(4.0**(L-1)) for x in row] for row in pssm]
    return pssm
    
def test_linearize(L,sigma):
    code = sample_code(L,sigma)
    lin = linearize_ref(code)
    pws = [score(code,site) for site in tqdm(kmers(L),total=4**L)]
    lins = [score_seq(lin,site) for site in tqdm(kmers(L),total=4**L)]
    plt.scatter(pws,lins,marker='.')
    plt.plot([-10,10],[-10,10])
    print pearsonr(pws,lins)

def test_linearize_explict(L,sigma):
    code = sample_code(L,sigma)
    lin = linearize(code)
    # i = random.randrange(L)
    # j = random.randrange(4)
    pred = []
    obs = []
    for i in range(L):
        for j in range(4):
            w = lin[i][j]
            w_obs = mean(score(code,site) for site in kmers(L) if site[i]=="ACGT"[j])
            pred.append(w)
            obs.append(w_obs)
    plt.scatter(pred,obs)
    plt.plot([-2,2],[-2,2])
    print pearsonr(pred,obs)

def sample_site(code):
    L = len(code) + 1
    partial_site = ""
    while len(partial_site) < L:
        #print "partial_site:",partial_site
        qs = normalize([partial_Zb(code,partial_site+b) for b in "ACGT"])
        #print "qs:",qs
        b = inverse_cdf_sample("ACGT",qs)
        #print "b:",b
        partial_site = partial_site + b
        #print "partial_site (end):",partial_site
    return partial_site

def prob_site(site,code):
    return exp(score(code, site))/compute_Zb(code)

def prob_sites(sites,code):
    Zb = compute_Zb(code)
    return [exp(score(code, site))/Zb for site in sites]

def test_sample_site(trials=10000):
    L, sigma = 3, 1
    code = sample_code(L,sigma)
    phats = [exp(score(code,site)) for site in kmers(L)]
    Z = sum(phats)
    ps = [phat/Z for phat in phats]
    sites = [sample_site(code) for _ in trange(trials)]
    counts = Counter(sites)
    p_obs = [counts[kmer]/float(trials) for kmer in kmers(L)]
    scatter(ps,p_obs)

def code_from_motif(motif, pc=1):
    cols = transpose(motif)
    N = float(len(motif))
    freqs = [{(b1,b2):((zip(col1,col2).count((b1,b2))+pc)/(N+16*pc))
           for b1 in "ACGT" for b2 in "ACGT"} for col1,col2 in pairs(cols)]
    ws = [{k:log(p) for k,p in freqs[0].items()}]
    for freq in freqs[1:]:
        ws.append({(b1,b2):log(freq[b1,b2]/sum(freq[b1,b2p] for b2p in "ACGT"))
                   for b1 in "ACGT" for b2 in "ACGT"})
    return ws

def test_code_from_motif():
    L = 3
    code = sample_code(L,sigma=1)
    motif = [sample_site(code) for i in trange(10000)]
    rec_code = code_from_motif(motif)
    rec_motif = [sample_site(rec_code) for i in trange(10000)]
    ps = [motif.count(kmer) for kmer in kmers(L)]
    rec_ps = [rec_motif.count(kmer) for kmer in kmers(L)]
    scatter(ps,rec_ps)
    
def ps_from_code(code):
    L = len(code) + 1
    return normalize([exp(score(code,kmer)) for kmer in kmers(L)])

def pairwise_model_from_motif(motif, pc=1):
    N = float(len(motif))
    cols = transpose(motif)
    col_pairs = map(transpose,pairs(cols))
    return [{(b1,b2):-log((col_pair.count((b1,b2)) + pc)/(N + 16*pc))
             for b1 in "ACGT" for b2 in "ACGT"} for col_pair in col_pairs]

def pairwise_likelihood(site,model):
    ws = [d[(si,sj)] for d,(si,sj) in zip(model,pairs(site))]
    return exp(sum(ws))

def code_mi(code):
    """estimate mutual information in code"""
    def weight_mi(weight):
        pass

def mgf1(code):
    """return mean of code Ws by computing moment generating function"""
    L = len(code) + 1
    Ws = [np.matrix([[w[b1,b2] for b2 in "ACGT"] for b1 in "ACGT"]) for w in code]
    one = np.ones((4,4))
    mp = np.linalg.matrix_power
    return 1.0/(4**L) * np.sum(np.sum(mp(one, i).dot(W).dot(mp(one, L-2-i)) for i, W in enumerate(Ws)))

def test_mgf1(trials=100):
    L = 10
    code = sample_code(L,1)
    a,b = mean_ci([(score(code, random_site(L))) for i in range(trials)])
    ans = mgf1(code)
    print ans, (a,b)
    return a < ans < b
    
def mgf2(code):
    """return variance of code Ws by computing moment generating function"""
    L = len(code) + 1
    Ws = [np.matrix([[w[b1,b2] for b2 in "ACGT"] for b1 in "ACGT"]) for w in code]
    one = np.ones((4,4))
    mp = np.linalg.matrix_power
    terms = [reduce(np.dot, [np.power(W,(i==k) + (j==k)) for k, W in enumerate(Ws)])
             for i in (range(L-1)) for j in range(L-1)]
    return 1.0/(4**L) * np.sum(np.sum(terms))

def test_mgf2(trials=100):
    L = 10
    code = sample_code(L,1)
    a,b = bs_ci(variance, ([(score(code, random_site(L))) for i in range(trials)]))
    ans = mgf2(code)
    print ans, (a,b)
    return a < ans < b
