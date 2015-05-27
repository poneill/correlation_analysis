import numpy as np
import random
from simplex_sampling_experiment import make_kmers,h_np
from estremo_on_a_napkin import sample_eps,boltzmann
from estremo_on_a_napkin_discrete import sample_sites#mutate_sites,
from estremo_on_a_napkin_potts import mutate,ln_mean
from utils import random_site,inverse_cdf_sample,mean,motif_ic,h,transpose,rslice,pairs,log2,total_motif_mi,variance
from utils import choose,choose2,mean,sd,subst,fac,log_fac,pl,concat
from tqdm import tqdm,trange
from math import exp,log,sqrt,ceil,pi
from matplotlib import pyplot as plt
import seaborn as sbn
import scipy
from scipy.stats import norm
from collections import defaultdict
from sde import ou_relaxation_time_from_sample,is_stationary

n = 16
L = 6
beta = 1
K = 4**L
G = 5*10**6
AAS = 5

nuc_pairs = [(b1,b2) for b1 in "ACGT" for b2 in "ACGT"]

def sample_code(sigma=1):
    return {(aa,b1,b2):random.gauss(0,sigma) for aa in range(AAS) for (b1,b2) in nuc_pairs}

def score_site(code,bd,site):
    return sum(code[aa,n1,n2] for aa,(n1,n2) in zip(bd,pairs(site)))

def approximate_max_fitness(N=100000):
    """Approximate max fitness from sample, assuming fitnesses log-normally distributed"""
    log_num_bds = (L-1) * log(AAS)
    log_num_motifs = L*n * log(4)
    log_num_genotypes = log_num_bds + log_num_motifs
    random_fits = ([fitness(sample_species()) for i in trange(N)])
    log_fits = map(log,random_fits)
    m,s = mean(log_fits),sd(log_fits)
    standard_max = sqrt(2*log_num_genotypes)
    shifted_max = standard_max * s + m
    # not terribly helpful because it exceeds 1
    
def occs(code,bd,sites):
    site_energies = [score_site(code,bd,site) for site in sites]
    #print "test background"
    #background = np.matrix([score_site(code,bd,random_site(L)) for i in trange(G)])
    #print "finish test background"
    mu = sum([mean([code[aa,b1,b2] for (b1,b2) in nuc_pairs]) for aa in bd])
    sigma = sqrt(sum([variance([code[aa,b1,b2] for (b1,b2) in nuc_pairs]) for aa in bd]))
    fg = sum(exp(-ep) for ep in site_energies)
    #test_bg = np.sum(np.exp(-background))
    bg = ln_mean(-mu,sigma)*G
    #print "error: %1.2f" % ((bg - test_bg)/test_bg * 100)
    return fg/(fg+bg)

def fitness(code,(bd,sites)):
    return occs(code,bd,sites)

def make_ringer(code):
    """minimize eps(site) - mu + (sigma^2)/2"""
    def aa_mu(aa):
        return mean([code[aa,b1,b2] for b1,b2 in nuc_pairs])
    def aa_sigma(aa):
        return sqrt(variance([code[aa,b1,b2] for b1,b2 in nuc_pairs]))
    (aa,b1,b2),min_score = min(code.items(),key=lambda ((aa,b1,b2),score):score - aa_mu(aa) + (aa_sigma(aa)**2)/2.0)
    bd = [aa]*(L-1)
    sites = ["".join(concat([(b1,b2) for j in range(L/2)])) for i in range(n)]
    return bd,sites

def make_ringer_spec(code):
    def aa_mu(aa):
        return mean([code[aa,b1,b2] for b1,b2 in nuc_pairs])
    def aa_sigma(aa):
        return sqrt(variance([code[aa,b1,b2] for b1,b2 in nuc_pairs]))
    (aa,b1,b2),min_score = min(code.items(),key=lambda ((aa,b1,b2),score):score - aa_mu(aa) + (aa_sigma(aa)**2)/2.0)
    
def sample_species():
    bd = [random.randrange(AAS) for i in range(L-1)]
    sites = sample_sites(n,L)
    return (bd,sites)

def moran_process(code,mutation_rate,N=1000,turns=10000,
                  init=sample_species,mutate=mutate,fitness=fitness,pop=None):
    mean_rec_muts,mean_site_muts = mutation_rate/3.0,mutation_rate
    site_mu = mean_site_muts/float(n*L)
    bd_mu = mean_rec_muts/float(L)
    if pop is None:
        pop = [(lambda spec:(spec,fitness(code,spec)))(init())
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
        pop[death_idx] = (daughter,fitness(code,daughter))
        mean_fits = mean(fits)
        hist.append((f,mean_fits))
        if turn % 1000 == 0:
            mean_dna_ic = mean([motif_ic(sites,correct=False) for ((bd,sites),_) in pop])
            print turn,"sel_fit:",f,"mean_fit:",mean_fits,"mean_dna_ic:",mean_dna_ic
    return pop,hist

def main(turns=1000000):
    mutation_rates = [10**i for i in np.linspace(-3,0,5)]
    d = {}
    sigmas = [10**i for i in np.linspace(-3,1,5)]
    for mutation_rate in mutation_rates:
        for sigma in sigmas:
            print "starting on:","mutation rate:",mutation_rate,"sigma:",sigma
            code = sample_code(sigma)
            pop,hist = moran_process(code,mutation_rate,
                                     init=lambda :make_ringer(code),
                                     N=1000,turns=turns,
                                     mutate=mutate,fitness=fitness)
            d[mutation_rate,sigma] = (pop,hist,code)
    return d


def pop_fitness(pop,code):
    return mean(f(spec,code) for spec in pop)

def pop_ic(pop,code):
    return mean(motif_ic(extract_sites(spec),correct=False) for spec in pop)

def pop_mi(pop,code):
    return mean(total_motif_mi(extract_sites(spec),correct=False) for spec in pop)

def pop_total(pop,code):
    return pop_ic(pop,code) + pop_mi(pop,code)

named_fs = [(pop_fitness,"fitness"),(pop_ic,"Motif IC"),
                    (pop_mi,"Motif MI"),(pop_total,"Pop Total")]

def interpret_main_experiment(results_dict):
    mutation_rates,sigmas = map(lambda x:sorted(set(x)),transpose(results_dict.keys()))
    fs,names = transpose(named_fs)
    subplot_dimension = ceil(sqrt(len(fs)))
    for idx,f in enumerate(fs):
        mat = np.zeros((len(mutation_rates),len(sigmas)))
        for i,mutation_rate in enumerate(sorted(mutation_rates)):
            for j,sigma in enumerate(sorted(sigmas)):
                pop,hist,code = results_dict[(rec_mut,site_mut)]
                mat[i,j] = f(pop,code)
                print i,j,mat[i,j]
        plt.subplot(subplot_dimension,subplot_dimension,idx)
        plt.imshow(mat,interpolation='none')
        plt.xticks(range(len(site_muts)),map(str,site_muts))
        plt.yticks(range(len(rec_muts)),map(str,rec_muts))
        #plt.yticks(rec_muts)
        plt.xlabel("sigma")
        plt.ylabel("mutation rate")
        plt.colorbar()
        title = names[idx]
        plt.title(title)
    plt.show()
