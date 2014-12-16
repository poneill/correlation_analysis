"""
Construct maxent distributions for binding site collections subject to
mono- and dinucleotide frequencies and determine which better supports
data via likelihood ratio tests.
"""
from motifs import Escherichia_coli
from parse_tfbs_data import tfbss
from itertools import product
from utils import verbose_gen,mean,motif_ic,show,concat,transpose
from math import exp,log,pi
import numpy as np
import random
from matplotlib import pyplot as plt
import sys
sys.path.append("/home/pat")
from bvh_model import psfm
from tqdm import tqdm

bases = "ACGT"

def idx_from_base(b):
    return bases.find(b)

def base_from_idx(i):
    return bases[i]
    
def kmers(L):
    return verbose_gen(product(*[bases for i in range(L)]),modulus=10000)

def make_mono_fs(L):
    return [lambda site,i=i,b=b:site[i]==b for i in range(L) for b in bases]
    
def make_di_fs(L):
    # use default value hack to get around lack of proper lexical scoping.
        # see: http://stackoverflow.com/questions/452610
    di_fs = [lambda site,i=i,b1=b1,b2=b2:(site[i]==b1 and site[i+1]==b2)
             for i in range(L-1)
             for b1 in bases
             for b2 in bases]
    return di_fs

def analytic_mono_lambs(sites,pseudocount=True):
    L = len(sites[0])
    freqs = psfm(sites,pseudocount)
    return [log(freqs[i][b]/0.25) for i in range(L) for b in range(4)]

def parametrize(sites,eta=1,tol=10**-3,lambs=None,mono=True,transfer_mats=True,debug=False,pseudocount="laplace"):
    L = len(sites[0])
    mono_fs = make_mono_fs(L)
    di_fs = make_di_fs(L)
    alphabet_size = (4 if mono else 16)
    if mono:
        fs = mono_fs
    else:
        fs = di_fs
    # count one more for each function if pseudocounting
    if pseudocount == "laplace":
        ps = 1
    elif pseudocount == "sg":
        ps = 1.0/alphabet_size
    else:
        ps = 10**-50
    ps_num = 1*ps
    ps_denom = alphabet_size * ps
    ys = [(sum([f(site) for site in sites]) + ps_num)/float(len(sites)+ps_denom) for f in fs]
    #print ys
    if lambs is None:
        lambs = [1 for y in ys]
    err = 1
    while err > tol:
        print err
        def w(x):
            return exp(-energy(x,fs,lambs))
        #print "computing partition"
        if transfer_mats:
            if mono:
                Z = mono_partition(lambs)
                yhats = [mono_fhat(lambs,i,b,Z=Z) for i in range(L) for b in bases]
            else: # if di
                Z = di_partition(lambs)
                yhats = [di_fhat(lambs,i,b1,b2,Z=Z) for i in range(L-1) for b1 in bases for b2 in bases]
                if False:
                    print err
        else: # if not transfer_mats
            Z = compute_partition(L,fs,lambs)
            yhats = [sum(fi(x)*w(x)/Z for x in kmers(L))
                     for fi in fs]
        lambs_new = [lamb + (yhat - y)*eta for lamb,y,yhat in zip(lambs,ys,yhats)]
        if False:
            for y,yhat,lamb,lamb_new in zip(ys,yhats,lambs,lambs_new):
                print y,"vs.",yhat,":",lamb,"->",lamb_new
        err = sum((y-yhat)**2 for y,yhat in zip(ys,yhats))
        #print "err:",err
        lambs = lambs_new
    #print lambs
    return lambs

def parametrize_approx(site,eta=1,tol=10**-2,mono=True,iterations=100000):
    L = len(sites[0])
    mono_fs = [lambda site,i=i,b=b:site[i]==b for i in range(L) for b in bases]
    di_fs = [lambda site,i=i,b1=b1,b2=b2:(site[i]==b1 and site[i+1]==b2)
             for i in range(L-1)
             for b1 in bases
             for b2 in bases]
    if mono:
        fs = mono_fs
    else:
        fs = di_fs
    ys = [mean(f(site) for site in sites) for f in fs]
    lambs = [1 for y in ys]
    err = 1
    while err > tol:
        site_chain = sample_dist(fs,lambs,iterations=iterations)
        yhats = [mean(fi(site) for site in site_chain)
                 for fi in fs]
        lambs_new = [lamb + (yhat - y)*eta for lamb,y,yhat in zip(lambs,ys,yhats)]
        for y,yhat,lamb,lamb_new in zip(ys,yhats,lambs,lambs_new):
            print y,"vs.",yhat,":",lamb,"->",lamb_new
        err = sum((y-yhat)**2 for y,yhat in zip(ys,yhats))
        print "err:",err
        lambs = lambs_new
    return lambs

def energy(x,fs,lambs):
    return sum(lamb*f(x) for lamb,f in zip(lambs,fs))

def compute_partition(L,fs,lambs):
    return sum(exp(-energy(xp,fs,lambs)) for xp in kmers(L))

def mono_partition(lambs):
    """compute partition function for mononucleotide model"""
    L = len(lambs)/4
    Ws = [np.matrix([[exp(-lambs[4*i + j]) for j in range(4)]] * 4) for i in range(L)]
    v0 = np.array([1/4.0]*4)
    vf = np.array([1]*4)
    Z = v0.dot(reduce(lambda w1,w2:w1.dot(w2),Ws)).dot(vf)[(0,0)]
    return Z

def di_partition(lambs):
    assert len(lambs) % 16 == 0
    L = len(lambs)/16 + 1
    Ws = [np.matrix([[exp(-lambs[16*i + 4*j2 + j1]) for j1 in range(4)] for j2 in range(4)]) for i in range(L-1)]
    v0 = np.array([1.0]*4)
    vf = np.array([1]*4)
    Z = v0.dot(reduce(lambda w1,w2:w1.dot(w2),Ws)).dot(vf)[(0,0)]
    return Z

def test_di_partition(L,lambs):
    fs = make_di_fs(L)
    #lambs = [lamb_f(i,b1,b2) for i in range(L-1) for b1 in bases for b2 in bases]
    Z_ref = compute_partition(L,fs,lambs)
    Z_spec = di_partition(lambs)
    return Z_ref,Z_spec,Z_ref/Z_spec
    
def mono_fhat(lambs,i,b,Z=None):
    """Compute frequency of base b at position i, given lambs."""
    if Z is None:
        Z = mono_partition(lambs)
    L = L_from_mono_lambs(lambs)
    Ws = [np.matrix([[exp(-lambs[4*ip + j]) for j in range(4)]] * 4) for ip in range(L)]
    jp = idx_from_base(b)
    Ws[i] = np.matrix([[exp(-lambs[4*i + j]) if j == jp else 0 for j in range(4)]] * 4)
    v0 = np.array([1/4.0]*4)
    vf = np.array([1]*4)
    Z_numer = v0.dot(reduce(lambda w1,w2:w1.dot(w2),Ws)).dot(vf)[(0,0)]
    return Z_numer/Z

def make_di_pdf_ref(fs,lambs):
    L = L_from_di_lambs(lambs)
    Z = compute_partition(L,fs,lambs)
    p = lambda x:exp(-energy(x,fs,lambs))/Z
    return p

def make_di_pdf(fs,lambs):
    L = L_from_di_lambs(lambs)
    Z = di_partition(lambs)
    p = lambda x:exp(-energy(x,fs,lambs))/Z
    return p

def make_mono_pdf_ref(fs,lambs):
    L = L_from_mono_lambs(lambs)
    Z = compute_partition(L,fs,lambs)
    p = lambda x:exp(-energy(x,fs,lambs))/Z
    return p

def make_mono_pdf(fs,lambs):
    L = L_from_mono_lambs(lambs)
    Z = mono_partition(lambs)
    p = lambda x:exp(-energy(x,fs,lambs))/Z
    return p

def di_fhat_ref(lambs,i,b1,b2,Z=None):
    L = L_from_di_lambs(lambs)
    fs = make_di_fs(L)
    if Z is None:
        Z = compute_partition(L,fs,lambs)
    f = lambda site:site[i] == b1 and site[i+1] == b2
    return sum(f(x)*exp(-energy(x,fs,lambs)) for x in kmers(L))/Z

def di_fhat_ref2(lambs,i,b1,b2):
    L = L_from_di_lambs(lambs)
    fs = make_di_fs(L)
    p = make_di_pdf_ref(fs,lambs)
    f = lambda site:site[i] == b1 and site[i+1] == b2
    return sum(f(x)*p(x) for x in kmers(L))
    
def di_fhat(lambs,i,b1,b2,Z=None):
    """Compute frequency of dinucleotide b1,b2 at position (i,i+1)"""
    L = L_from_di_lambs(lambs)
    if Z is None:
        Z = di_partition(lambs)
    #Ws = [np.matrix([[exp(-lambs[16*ip + 4*j1 + j2]) for j1 in range(4)] for j2 in range(4)]) for ip in range(L-1)]
    Ws = [np.matrix([[exp(-lambs[16*ip + 4*j2 + j1]) for j1 in range(4)] for j2 in range(4)]) for ip in range(L-1)]
    jp1 = idx_from_base(b1)
    jp2 = idx_from_base(b2)
    Ws[i] = np.matrix([[exp(-lambs[16*i + 4*j2 + j1]) if j1 == jp2 and j2 == jp1 else 0
                        for j1 in range(4)] for j2 in range(4)])
    v0 = np.array([1]*4)
    vf = np.array([1]*4)
    Z_numer = v0.dot(reduce(lambda w1,w2:w1.dot(w2),Ws)).dot(vf)[(0,0)]
    return Z_numer/Z
    
def sample_dist(fs,lambs,iterations=100000):
    L = L_from_di_lambs(lambs)
    f = lambda x:exp(-energy(x,fs,lambs))
    def prop(x):
        i = random.randrange(L)
        b = x[i]
        bp = random.choice([c for c in bases if not c == b])
        return subst(x,bp,i)
    def prop2(x):
        return random_site(L)
    site_chain = mh(f,prop2,x0=random_site(L),iterations=iterations)
    return site_chain
        
def prob(x,fs,lambs,Z=None):
    if Z is None:
        L = len(x)
        Z = sum(exp(-energy(xp,fs,lambs)) for xp in kmers(L))
    return exp(-energy(x,fs,lambs))/Z

def subst(xs,ys,i):
    ans = xs[:i] + ys + xs[i+len(ys):]
    assert(len(x) == len(ans))
    return ans

def test_yhats_ref(lambs):
    L = L_from_di_lambs(lambs)
    Z = di_partition(lambs)
    fs = make_di_fs(L)
    yhats = [di_fhat(lambs,i,b1,b2,Z=Z) for i in range(L-1) for b1 in bases for b2 in bases]
    Z_ref = compute_partition(L,fs,lambs)
    yhats_ref = [di_fhat_ref(lambs,i,b1,b2) for i in range(L-1) for b1 in bases for b2 in bases]
    plt.plot(yhats_ref)
    plt.plot(yhats)
    print sum(yhats_ref),sum(yhats)
    for i in range(0,len(yhats),16):
        print sum(yhats_ref[i:i+16]),sum(yhats[i:i+16])
    
def mono_likelihood(sites,tol=10**-3,pseudocount="laplace"):
    L = len(sites[0])
    lambs = parametrize(sites,eta=1,tol=tol,mono=True,debug=False,pseudocount=pseudocount)
    fs = make_mono_fs(L)
    p = make_mono_pdf(fs,lambs)
    return sum(log(p(site)) for site in sites)

def di_likelihood(sites,tol=10**-3,pseudocount="laplace"):
    L = len(sites[0])
    lambs = parametrize(sites,eta=1,tol=tol,mono=False,pseudocount=pseudocount)
    fs = make_di_fs(L)
    p = make_di_pdf(fs,lambs)
    return sum(log(p(site)) for site in sites)

def L_from_mono_lambs(lambs):
    assert len(lambs) % 4 == 0
    return len(lambs) / 4

    
def L_from_di_lambs(lambs):
    assert len(lambs) % 16 == 0
    return len(lambs) / 16 + 1

def main_likelihood_experiment():
    for tf in Escherichia_coli.tfs:
        print tf
        sites = getattr(Escherichia_coli,tf)
        tols = [10**-i for i in range(7)]
        print "mono"
        mono_lls = [show(mono_likelihood(sites,tol=show(tol))) for tol in tols]
        print "di"
        di_lls = [show(di_likelihood(sites,tol=show(tol))) for tol in tols]
        plt.close()
        plt.plot(tols,mono_lls,label="Mono")
        plt.plot(tols,di_lls,label="Di")
        plt.xlabel("Tolerance")
        plt.ylabel("Log Likelihood")
        plt.semilogx()
        plt.legend()
        #plt.title("Mono- vs. Di-nucleotide Log-Likelihood in %s sites" % tf)
        fmt_string = "%s, site length:%s,num sites:%s,motif ic:%1.2f" % (tf,len(sites[0]),len(sites),motif_ic(sites))
        plt.title(fmt_string)
        plt.savefig("%s_mono_vs_di_ll_w_pseudocount.png" % tf,dpi=300)
        plt.close()

def likelihood_dict(pseudocount="laplace"):
    d = {}
    for tf in tqdm(Escherichia_coli.tfs):
        print tf
        sites = getattr(Escherichia_coli,tf)
        d[tf] = (mono_likelihood(sites,tol=10**-3,pseudocount=pseudocount),
                 di_likelihood(sites,tol=10**-3,pseudocount=pseudocount))
    return d

def full_likelihood_dict(pseudocount="laplace"):
    d = {}
    tfbs_dict = make_tfbs_dict()
    for prt_id in tqdm(tfbs_dict):
        sites = tfbs_dict[prt_id]
        d[prt_id] = (mono_likelihood(sites,tol=10**-3,pseudocount=pseudocount),
                     di_likelihood(sites,tol=10**-3,pseudocount=pseudocount))
    return d
    
def plot_mono_vs_di_likelihood(ll_dict = None):
    if ll_dict is None:
        ll_dict = likelihood_dict()
    normed_dict = {tf:tuple(map(lambda x:x/float(len(getattr(Escherichia_coli,tf))*len(getattr(Escherichia_coli,tf)[0])),(mono,di))) for (tf,(mono,di)) in ll_dict.items()}
    plt.scatter(*transpose(ll_dict.values()))
    for (tf,(mono,di)) in ll_dict.items():
        sites = getattr(Escherichia_coli,tf)
        text = "%s\n#:%s\nw:%s\nIC:%1.2f" % (tf,len(sites),len(sites[0]),motif_ic(sites))
        plt.annotate(text,(mono,di))
    min_val = min(concat(ll_dict.values()))
    max_val = max(concat(ll_dict.values()))
    plt.xlabel("Mono LL")
    plt.ylabel("Di LL")
    plt.plot([min_val,max_val],[min_val,max_val],linestyle="--")

def aic(l,k,n):
    return 2*k - 2*l

def bic(ll,k,n):
    return -2*ll + k*(log(n) - log(2*pi))
    
def info_crit_analysis(ll_dict,crit=bic):
    """BIC: lower is better"""
    print "tf, mono_ll,di_ll,mono_crit,di_crit,di_crit - mono_crit"
    mono_crits = []
    di_crits = []
    for tf in Escherichia_coli.tfs:
        sites = getattr(Escherichia_coli,tf)
        n = len(sites)
        w = len(sites[0])
        mono_k = (4-1)*w
        di_k = ((16-3)*(w-1))
        #di_k = (4-1)*w
        mono_ll,di_ll = ll_dict[tf]
        mono_crit = crit(mono_ll,mono_k,n)
        di_crit = crit(di_ll,di_k,n)
        mono_crits.append(mono_crit)
        di_crits.append(di_crit)
        print tf,mono_ll,di_ll,mono_crit,di_crit,di_crit - mono_crit
    plt.scatter(mono_crits,di_crits)
    min_val = min(mono_crits + di_crits)
    max_val = max(mono_crits + di_crits)
    print min_val,max_val
    plt.plot([min_val,max_val],[min_val,max_val],linestyle="--")
    plt.xlabel("Mono BIC")
    plt.ylabel("DI BIC")
def log2(x):
    return log(x,2)
    
def kl(ps,qs):
    return sum(p*log2(p/q) for p,q in zip(ps,qs))
    
def compare_kls():
    """Plot kl_mono/base against number of sites.  No relationship."""
    kl_dict = {}
    for tf in Escherichia_coli.tfs:
        sites = getattr(Escherichia_coli,tf)
        L = len(sites[0])
        lambs = parametrize(sites)
        fs = make_mono_fs(L)
        q = make_mono_pdf(fs,lambs)
        ps = [sites.count(site)/float(len(sites)) for site in sites]
        qs = [q(site) for site in sites]
        plt.close()
        plt.scatter(ps,qs)
        plt.loglog()
        plt.plot([min(ps+qs),min(ps+qs)],[0,0])
        kl_dict[tf] = kl(ps,qs)
    return kl_dict

