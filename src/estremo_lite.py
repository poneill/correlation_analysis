from moran_process import moran_process
from utils import random_motif,mutate_motif_p,choose,mean,mutate_motif,prod,fac,h,permute,prod,mutate_motif_p
from utils import inverse_cdf_sample,mh
from math import exp,log
import itertools
from collections import Counter
import numpy as np
import random
from tqdm import *

sigma = 2#kbt per match

def on_off_simulation(N=10000,turns=100000):
    """simulate evolution of motifs with simple on-off model against randomized backgrounds"""
    n = 16
    L = 16
    G = 1000
    def init_species():
        return random_motif(n,L)
    def mutate(motif):
        p = 0.5
        return mutate_motif_p(motif,p/(n*L))
    def fitness(motif):
        eps = [sigma*sum(b!="A" for b in site) for site in motif]
        Zb = G * sum(exp(-sigma*i)*choose(L,i)*(1/4.0)**i*(3/4.0)**(L-i) for i in range(L+1))
        fg = sum(exp(-ep) for ep in eps)
        return fg/(fg + Zb)
    def ringer():
        return ["A"*L]*n
    def diagnostics(pop):
        return mean(sum(sum(b=="A" for b in site) for site in motif) for (motif,f) in pop)/(n*L)
    max_fit = fitness(ringer())
    pop = moran_process(fitness,mutate=mutate,init_species=init_species,turns=turns,N=N,diagnostic_modulus=1000)
    return pop
         

def enumerate_eps(n,L):
        return itertools.combinations_with_replacement(range(L+1),n)

def num_eps(n,L):
    return choose(n+L,n)

def test_num_eps():
    for _ in range(10):
        n = random.randrange(5)
        L = random.randrange(5)
        print len(list(enumerate_eps(n,L))) == num_eps(n,L)
    
def enumerate_eps_by_mismatch(n,L):
    return (part for i in range(n*(L+1)) for part in partition_func(i,k=n,l=0) if all([x <= L for x in part]))

def enumerate_eps_by_mismatch_ref(n,L):
    return sorted(enumerate_eps(n,L),key=sum)

def test_enumerate_eps_by_mismatch(n,L):
    eps_ref = enumerate_eps_by_mismatch_ref(n,L)
    eps = enumerate_eps_by_mismatch(n,L)
    return list(eps) == list(eps_ref)
    
def partition(number):
    answer = set()
    answer.add((number, ))
    for x in range(1, number):
        for y in partition(number - x):
            answer.add(tuple(sorted((x, ) + y)))
    return answer

def partition_func(n,k,l=1):
    '''n is the integer to partition, k is the length of partitions, l is the min partition element size'''
    # from SO
    if k < 1:
        raise StopIteration
    if k == 1:
        if n >= l:
            yield (n,)
        raise StopIteration
    for i in range(l,n+1):
        for result in partitionfunc(n-i,k-1,i):
            yield (i,)+result

def num_combs_with_replacement(n,L):
    return choose(L+n,L)

def mean_ic_from_eps(cs,n,L):
    """compute approximate information content of motif with c mismatches in each site"""
        #"""Should depend only on permutations, not on substitutions"""
        # cs counts number of MISMATCHES in each site
    p_mismatch = 1.0/(n*L)*sum(cs)
    p_match = 1 - p_mismatch
    col_ent = h([p_match,p_mismatch/3.0,p_mismatch/3.0,p_mismatch/3.0])
    return L*(2 - col_ent)

def compute_Zb(n,L,sigma,G):
    return G * sum(exp(-sigma*i)*choose(L,i)*(3/4.0)**i*(1/4.0)**(L-i) for i in range(L+1))

def compute_log_fs(n,L,sigma,G):
    Zb = compute_Zb(n,L,sigma,G)
    def log_f(eps):
        fgs = [exp(-sigma*ep) for ep in eps]
        Zf = sum(fgs)
        # log(prod(fg/(Zf+Zb) for fg in fgs)
        return sum(log(fg) - log(Zf + Zb) for fg in fgs)
    log_fs = np.array([log_f(eps) for eps in tqdm(enumerate_eps(n,L),total=num_combs_with_replacement(n,L))])
    return log_fs

def compute_gs(n,L):
    def w(eps):
        """number of ways to """
        return (fac(L)**len(eps))/prod(fac(ep)*fac(L-ep) for ep in eps)*3**sum(eps)
    def multiplicities(eps):
        metacounts = Counter(eps)
        substitutions = fac(n)/prod(fac(multiplicity) for multiplicity in metacounts.values())
        return substitutions
    def num_genotypes(eps):
        return w(eps)*multiplicities(eps)
    gs = np.array([num_genotypes(eps) for eps in tqdm(enumerate_eps(n,L),total=num_combs_with_replacement(n,L))])
    return gs

def mean_ic(n=16,L=16,G=1000,N=100,sigma=1,log_fs=None,gs=None):
    ps = sella_hirsch_predictions(n,L,G,N,sigma,log_fs,gs)
    ics = np.array([mean_ic_from_eps(eps,n,L) for eps in enumerate_eps(n,L)])
    return ps.dot(ics)

def mean_fitness(n=16,L=16,G=1000,N=100,sigma=1,log_fs=None,gs=None):
    log_fs = compute_log_fs(n,L,sigma,G)
    ps = sella_hirsch_predictions(n,L,G,N,sigma,log_fs,gs)
    return ps.dot(np.exp(log_fs))

def mean_occ(n=16,L=16,G=1000,N=100,sigma=1,log_fs=None,gs=None):
    if log_fs is None:
        log_fs = compute_log_fs(n,L,sigma,G)
    log_occs = (1.0/n)*log_fs
    ps = sella_hirsch_predictions(n,L,G,N,sigma,log_fs=log_fs,gs=gs)
    return ps.dot(np.exp(log_occs))

def mean_mismatches(n=16,L=16,G=1000,N=100,sigma=1,log_fs=None,gs=None):
    ps = sella_hirsch_predictions(n,L,G,N,sigma,log_fs=log_fs,gs=gs)
    mismatches = np.array([sum(eps) for eps in enumerate_eps(n,L)])
    return ps.dot(mismatches)
    
def sella_hirsch_predictions(n=16,L=16,G=1000,N=100,sigma=1,log_fs=None,gs=None):
    nu = N - 1
    if log_fs is None:
        # depends on G, but not on N
        log_fs = compute_log_fs(n,L,sigma,G)
    #ics = np.array([mean_ic_from_eps(eps,n,L) for eps in enumerate_eps()])
    #mismatches = np.array([sum(eps) for eps in enumerate_eps()])
    if gs is None:
        gs = compute_gs(n,L)
    log_gs = np.log(gs)
    #bzs = gs * (np.exp(nu*log_fs))
    log_bzs = log_gs + nu*log_fs
    bzs = np.exp(log_bzs)
    #print sum(np.abs(np.exp(log_bzs)-bzs))
    #boltzmann_factors = np.array([num_genotypes(eps)*exp(nu*f(eps)) for eps in enumerate_eps()])
    Z = np.sum(bzs)
    ps = bzs/Z
    #print "mean fitness:",fs.dot(ps)
    #print "mean ic:",ics.dot(ps)
    return ps
    #return fs.dot(ps),mismatches.dot(ps),ics.dot(ps)

def expect_stat(n=16,L=16,G=5*10**6,N=10,sigma=1,T=None,theta=0.999):
    if T is None:
        T = lambda eps:mean_ic_from_eps(eps,n,L)
    nu = N - 1
    Zb = compute_Zb(n,L,sigma,G)
    def log_f(eps):
        fgs = [exp(-sigma*ep) for ep in eps]
        Zf = sum(fgs)
        # log(prod(fg/(Zf+Zb) for fg in fgs)
        return sum(log(fg) - log(Zf + Zb) for fg in fgs)
    def w(eps):
        """number of ways to """
        return (fac(L)**len(eps))/prod(fac(ep)*fac(L-ep) for ep in eps)*3**sum(eps)
    def multiplicities(eps):
        metacounts = Counter(eps)
        substitutions = fac(n)/prod(fac(multiplicity) for multiplicity in metacounts.values())
        return substitutions
    def num_genotypes(eps):
        return w(eps)*multiplicities(eps)
    def log_bz(eps):
        return log(num_genotypes(eps)) + nu*log_f(eps)
    Z = 0
    acc = 0
    len_eps = num_eps(n,L)
    xs = []
    for i,eps in enumerate(enumerate_eps_by_mismatch(n,L)):
        cur_bz = exp(log_bz(eps))
        xs.append(cur_bz)
        Z += cur_bz
        print i,eps,cur_bz,Z
        cur_T = T(eps)
        acc += cur_T*cur_bz
        remaining_terms = len_eps - i
        max_prop_remaining = cur_bz*remaining_terms/Z
        #print i,cur_w,max_prop_remaining
        if max_prop_remaining < 1-theta:
            print "stopped at:",i,cur_bz,remaining_terms,Z,cur_bz*remaining_terms/Z
            print "savings on terms: %1.2f %%" % (100* (1 - float(i)/len_eps))
            return acc/Z,xs
    return acc/Z,xs
    

def stat_matrix(n=10,G=5*10**6,N=5,sigma=1,log_fs=None,gs=None,T=mean_occ):
    Ls = range(1,12)
    sigmas = np.linspace(0,10,10)
    print "computing gs"
    gss = [compute_gs(n,L) for L in Ls]
    print "computing matrix"
    return [[T(n,show(L),G,N,show(sigma),log_fs=None,gs=gs) for L,gs in zip(Ls,gss)] for sigma in sigmas]
    
def sample_motif_from_mismatches(cs,L):
    return ["".join(permute(['A'] * (L - c) + [random.choice("CGT") for i in range(c)])) for c in cs]

def estremo_lite_vs_maxent_motifs():
    n = 20
    L = 10
    Ns = np.linspace(10,10000,100)
    pss = [sella_hirsch_predictions(n=n,L=L,G=1000,N=N) for N in tqdm(Ns)]
    ics = np.array([mean_ic_from_eps(eps,n,L) for eps in enumerate_eps(N,L)])
    expected_ics = [ics.dot(ps) for ps in pss]

def sella_hirsch_mh_sampling(n=16,L=16,G=1000,N=100,sigma=1,iterations=50000):
    Zb = compute_Zb(n,L,sigma,G)
    nu = N-1
    def fitness(motif):
        eps = [sigma*sum(b!="A" for b in site) for site in motif]
        fg = sum(exp(-sigma*ep) for ep in eps)
        return fg/(fg + Zb)
    def log_p(motif):
        return (nu * log(fitness(motif)))
    def proposal(motif):
        p = 4.0/(n*L)
        return mutate_motif_p(motif,p)
    x0 = random_motif(n,L)
    chain = mh(log_p,proposal,x0,use_log=True,iterations=iterations)
    return chain
