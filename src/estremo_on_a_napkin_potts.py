import numpy as np
import random
from simplex_sampling_experiment import make_kmers,h_np
from estremo_on_a_napkin import sample_eps,boltzmann
from estremo_on_a_napkin_discrete import mutate_sites,sample_sites
from utils import random_site,inverse_cdf_sample,mean,motif_ic,h,transpose,rslice,pairs,log2,total_motif_mi,variance
from utils import choose,choose2,mean,sd,subst,fac,log_fac
from tqdm import tqdm,trange
from math import exp,log,sqrt,ceil
from matplotlib import pyplot as plt
#import seaborn as sbn
import scipy
from collections import defaultdict
from sde import ou_relaxation_time_from_sample,is_stationary

n = 16
L = 5
beta = 1
K = 4**L
G = 5*10**6
AAS = 5
num_recognizers = AAS**L
num_motifs = 4**(n*L)
permutations = fac(L)*fac(n)
num_genotypes = num_recognizers * num_motifs / float(permutations)
log_num_genotypes = L*log(AAS) + (n*L)*log(4) - log_fac(L)
# site_mut_prob = 10**-0.5
# rec_mut_prob = 10**-0.5

def sample_code(sigma=1):
    return {(i,b):random.gauss(0,sigma) for i in range(AAS) for b in "ACGT"}

def code_statistics(code):
    mus = [mean([code[(aa,b)] for aa in range(AAS)]) for b in "ACGT"]
    sigmas = [sd([code[(aa,b)] for aa in range(AAS)]) for b in "ACGT"]
    return sum(mus),sum(sigmas)

if not 'code' in globals():
    code = sample_code()
    bg_mu,bg_sigma = code_statistics(code)

def score_site(code,bd,site):
    return sum(code[aa,s] for (aa,s) in zip(bd,site))

def occs(code,bd,sites):
    site_energies = [score_site(code,bd,site) for site in sites]
    #background = [score_site(code,bd,random_site(L)) for i in range(G)]
    mu = sum([mean([code[aa,b] for b in "ACGT"]) for aa in bd])
    sigma = sqrt(sum([variance([code[aa,b] for b in "ACGT"]) for aa in bd]))
    fg = sum(exp(-ep) for ep in site_energies)
    #test_bg = np.sum(np.exp(-background))
    bg = ln_mean(-mu,sigma)*G
    #print "error: %1.2f" % ((bg - test_bg)/test_bg * 100)
    return fg/(fg+bg)

def transition_prob(fi,fj,N):
    """compute moran process fixation probability pi(i->j) in rare
    mutation limit according to Sella and Hirsh 2005"""
    if fi == fj:
        return 1.0/N
    else:
        return (1-fi/float(fj))/(1-(fi/float(fj))**N)
    
def ln_mean(mu,sigma):
    return exp(mu+(sigma**2)/2.0)

def ln_sd(mu,sigma):
    return sqrt(exp(sigma**2)-1)*exp(mu+(sigma**2)/2.0)
    
def fitness((bd,sites)):
    return occs(code,bd,sites)

def mutate_bd(bd,bd_mu):
    new_bd = bd[:]
    for i in range(len(bd)):
        if random.random() < bd_mu:
            new_bd[i] = random.randrange(AAS)
    return new_bd

def mutate_sites_spec(sites,site_mu):
    muts = np.random.binomial(n*L,site_mu)
    if muts == 0:
        return sites
    # else...
    new_sites = sites[:]
    for mut in range(muts):
        i = random.randrange(n)
        j = random.randrange(L)
        site = new_sites[i]
        b = site[j]
        new_b = random.choice([c for c in "ACGT" if not c == b])
        new_sites[i] = subst(site,new_b,j)
    return new_sites
        
def mutate((bd,sites),site_mu,bd_mu):
    new_bd = mutate_bd(bd,bd_mu)
    new_sites = mutate_sites_spec(sites,site_mu)
    return (new_bd,new_sites)

def sample_species():
    bd = [random.randrange(AAS) for i in range(L)]
    sites = sample_sites(n,L)
    return (bd,sites)

def extract_bd((bd,sites)):
    return bd

def extract_sites((bd,sites)):
    return sites
    
def make_ringer():
    (aa,b),min_score = min(code.items(),key=lambda (pair,score):score)
    bd = [aa]*L
    sites = [b*L for i in range(n)]
    return bd,sites

def make_ringer2():
    """minimize eps(site) - mu + (sigma^2)/2"""
    def aa_mu(aa):
        return mean([code[aa,b] for b in "ACGT"])
    def aa_sigma(aa):
        return sqrt(variance([code[aa,b] for b in "ACGT"]))
    (aa,b),min_score = min(code.items(),key=lambda ((aa,b),score):score - aa_mu(aa) + (aa_sigma(aa)**2)/2.0)
    bd = [aa]*L
    sites = [b*L for i in range(n)]
    return bd,sites

def optimal_sites_for_bd(bd):
    best_site = "".join([min("ACGT",key=lambda b:code[aa,b]) for aa in bd])
    return [best_site]*n
    
def moran_process(mean_rec_muts,mean_site_muts,N=1000,turns=10000,
                  init=make_ringer2,mutate=mutate,fitness=fitness,pop=None):
    site_mu = mean_site_muts/float(n*L)
    bd_mu = mean_rec_muts/float(L)
    if pop is None:
        pop = [(lambda spec:(spec,fitness(spec)))(init())
               for _ in trange(N)]
    hist = []
    for turn in xrange(turns):
        fits = [f for (s,f) in pop]
        birth_idx = inverse_cdf_sample(range(N),fits,normalized=False)
        death_idx = random.randrange(N)
        #print birth_idx,death_idx
        mother,f = pop[birth_idx]
        daughter = mutate(mother,site_mu,bd_mu)
        #print "mutated"
        pop[death_idx] = (daughter,fitness(daughter))
        mean_fits = mean(fits)
        hist.append((f,mean_fits))
        if turn % 1000 == 0:
            mean_dna_ic = mean([motif_ic(sites,correct=False) for ((bd,sites),_) in pop])
            print turn,"sel_fit:",f,"mean_fit:",mean_fits,"mean_dna_ic:",mean_dna_ic
    return pop,hist

def main(turns=200000):
    mean_rec_mus = [10**i for i in np.linspace(-3,0,5)]
    mean_site_mus = [10**i for i in np.linspace(-3,0,5)]
    d = {}
    for mean_rec_mu in mean_rec_mus:
        for mean_site_mu in mean_site_mus:
            print "starting on:",mean_rec_mu,mean_site_mu
            pop,hist = moran_process(mean_rec_mu,mean_site_mu,
                                     init=make_ringer2,
                                     N=1000,turns=turns,
                                     mutate=mutate,fitness=fitness)
            d[mean_rec_mu,mean_site_mu] = (pop,hist)
    return d

def interpret_main_experiment(results_dict,named_fs=None):
    if named_fs is None:
        named_fs = [(fitness,"fitness"),(lambda org:motif_ic(extract_sites(org)),"Motif IC"),
                    (lambda org:total_motif_mi(extract_sites(org)),"Motif MI")]
    rec_muts,site_muts = map(lambda x:sorted(set(x)),transpose(results_dict.keys()))
    fs,names = transpose(named_fs)
    subplot_dimension = ceil(sqrt(len(fs)))
    for idx,f in enumerate(fs):
        mat = np.zeros((len(rec_muts),len(site_muts)))
        for i,rec_mut in enumerate(sorted(rec_muts)):
            for j,site_mut in enumerate(sorted(site_muts)):
                pop,hist = results_dict[(rec_mut,site_mut)]
                mat[i,j] = mean([f(x) for x,fit in pop])
                print i,j,mat[i,j]
        plt.subplot(subplot_dimension,subplot_dimension,idx)
        plt.imshow(mat,interpolation='none')
        plt.xticks(range(len(site_muts)),map(str,site_muts))
        plt.yticks(range(len(rec_muts)),map(str,rec_muts))
        #plt.yticks(rec_muts)
        plt.xlabel("site mutation rate")
        plt.ylabel("rec mutation rate")
        plt.colorbar()
        title = names[idx]
        plt.title(title)
    plt.show()

def plot_main_experiment_trajectories(results):
    for k,(pop,hist) in results.items():
        print k
        traj = transpose(hist)[1]
        ou_param_recovery(traj)
        plt.plot(traj)
    plt.show()

def rec_site_mi((bd,sites),correct=False):
    """compute mutual information between binding domain, sites"""
    xs = [b for b in bd for s in sites]
    ys = [s for col in transpose(sites) for s in col]
    return mi(xs,ys,correct=correct)
