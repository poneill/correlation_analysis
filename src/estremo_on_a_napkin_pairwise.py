import numpy as np
import random
from simplex_sampling_experiment import make_kmers,h_np
from estremo_on_a_napkin import sample_eps,boltzmann
from estremo_on_a_napkin_discrete import sample_sites,mutate_site
from estremo_on_a_napkin_potts import mutate,ln_mean,extract_sites,mutate_bd
from utils import random_site,inverse_cdf_sample,mean,motif_ic,h,transpose,rslice,pairs,log2,total_motif_mi,variance
from utils import choose,choose2,mean,sd,subst,fac,log_fac,pl,concat,mh,anneal
from tqdm import tqdm,trange
from math import exp,log,sqrt,ceil,pi
from matplotlib import pyplot as plt
#import seaborn as sbn
import scipy
from scipy.stats import norm,spearmanr,pearsonr
from collections import defaultdict
from sde import ou_relaxation_time_from_sample,is_stationary
import itertools

n = 16
L = 10
beta = 1
K = 4**L
G = 5*10**6
num_aas = 20
aas = range(num_aas)
nucs = "ACGT"
nuc_pairs = [(b1,b2) for b1 in nucs for b2 in nucs]
nuc_trips = [(b1,b2,b3) for b1 in nucs for b2 in nucs for b3 in nucs]
log10 = lambda x:log(x,10)

def sample_code(sigma=1):
    return {(aa,b1,b2):random.gauss(0,sigma) for aa in aas for (b1,b2) in nuc_pairs}

def score_site(code,bd,site):
    return sum(code[aa,n1,n2] for aa,(n1,n2) in zip(bd,pairs(site)))

def approximate_max_fitness(N=100000):
    """Approximate max fitness from sample, assuming fitnesses log-normally distributed"""
    log_num_bds = (L-1) * log(num_AAS)
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
    sigma = sqrt(sum([variance([code[aa,b1,b2] for (b1,b2) in nuc_pairs]) for aa in bd])) # XXX revisit w/ bd_variance
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

def mh_ringer(code):
    f = lambda(x):fitness(code,x)
    prop = lambda x:mutate(x,0.001,0.001)
    x0 = sample_species()
    chain = mh(f,prop,x0,use_log=True)

def anneal_ringer(code):
    f = lambda(x):-log(fitness(code,x))
    mu = 0.1
    prop = lambda (bd,sites):(mutate_bd(bd,mu),[mutate_site(sites[0],mu)]*n)
    x0 = sample_species()
    ring = anneal(f,prop,x0,return_trajectory =False,k=0.001)
    return ring
    
def make_ringer_viterbi(code,L=L):
    """Make ringer using viterbi algorithm"""
    etas = []
    etas.append({x3:min(code[aa,x1,x3] for x1 in nucs for aa in aas) for x3 in nucs})
    for i in range(1,L):
        d = {xnp1:min(code[aa,xn,xnp1] + etas[i-1][xn] for xn in nucs for aa in aas) for xnp1 in nucs}
        etas.append(d)
    binding_site = "".join([min(nucs,key=lambda n:eta[n]) for eta in etas])
    sites = [binding_site for i in range(n)]
    bd = [min(aas,key=lambda aa:code[aa,n1,n2]) for n1,n2 in pairs(binding_site)]
    return bd,sites

def maximum_binder_ref(code,bd):
    """given a code and binding domain, find minimum energy binding site"""
    L = len(bd) + 1
    return min(make_kmers(L),key=lambda s:score_site(code,bd,s))

def maximum_binder(code,bd):
    L = len(bd) + 1
    etas = []
    etas.append({x1:min(code[bd[0],x0,x1] for x0 in nucs) for x1 in nucs})
    for i in range(1,L-1):
        d = {xnp1:min(code[bd[i],xn,xnp1] + etas[i-1][xn] for xn in nucs) for xnp1 in nucs}
        etas.append(d)
    bs = [None for i in range(L)]
    #bs = [min(nucs,key=lambda x:etas[-1])]
    for i in range(L-1,0-1,-1):
        if i == L-1:
            key = lambda nuc:etas[i-1][nuc]
        elif i == 0:
            key = lambda nuc:code[bd[i],nuc,bs[i+1]]
        else:
            key = lambda nuc:etas[i-1][nuc] + code[bd[i],nuc,bs[i+1]]
        bs[i] = min(nucs,key=key)
    #binding_site = "".join([min(nucs,key=lambda nuc:eta[nuc]) for eta in etas])
    return "".join(bs)

def test_maximum_binder(N=100,L=5):
    for i in range(N):
        code = sample_code()
        bd = [random.choice(aas) for i in range(L-1)]
        if not maximum_binder_ref(code,bd) == maximum_binder(code,bd):
            return code,bd
    print "Passed"
    return True,True
    
def make_ringer_viterbi2(code,L=L):
    """Make ringer using viterbi algorithm"""
    def aa_mu(aa):
        return mean([code[aa,b1,b2] for b1,b2 in nuc_pairs])
    def aa_sigma(aa):
        return sqrt(variance([code[aa,b1,b2] for b1,b2 in nuc_pairs]))
    etas = []
    #etas.append({x3:min(code[aa,x1,x3] for x1 in nucs for aa in aas) for x3 in nucs})
    etas.append({x3:min(code[aa,x1,x3] - aa_mu(aa) + (aa_sigma(aa)**2)/2.0
                        for x1 in nucs for aa in aas) for x3 in nucs})
    for i in range(1,L):
        d = {xnp1:min(code[aa,xn,xnp1] + etas[i-1][xn] for xn in nucs for aa in aas) for xnp1 in nucs}
        etas.append(d)
    binding_site = "".join([min(nucs,key=lambda n:eta[n]) for eta in etas])
    sites = [binding_site for i in range(n)]
    bd = [min(aas,key=lambda aa:code[aa,n1,n2] - aa_mu(aa) + (aa_sigma(aa)**2)/2.0)
          for n1,n2 in pairs(binding_site)]
    return bd,sites

def make_ringer_ref(code,L):
    f = lambda bd:(-score_site(code,bd,maximum_binder(code,bd))
                   + bd_mean(code,bd)
                   - bd_variance_ref(code,bd)/2)
    bd = list(max(tqdm(itertools.product(*[aas for i in range(L)]),total=len(aas)**L),key=f))
    site = maximum_binder(code,bd)
    sites = [site for i in range(n)]
    return bd,sites

def analyze_ringer_degeneracy(trials=100):
    """does ringer always consist of degenerate site, binding domain?"""
    for i in range(trials):
        print i
        code = sample_code()
        bd,sites = make_ringer_ref(code,L)
        if not len(set(bd)) == 1 and len(set(sites[0])) == 1:
            return code, bd,sites
    return True,True,True
    
def predicted_vs_actual_Zb(code,bd):
    L = len(bd) + 1
    kmer_scores = [score_site(code,bd,kmer) for kmer in make_kmers(L)]
    pred_mu = sum([mean([code[aa,n1,n2] for (n1,n2) in nuc_pairs]) for aa in bd])
    pred_sigma_sq = sum([variance([code[aa,n1,n2] for (n1,n2) in nuc_pairs]) for aa in bd])
    pred_mean = exp(pred_mu + (pred_sigma_sq**2)/2.0)
    obs_mu = mean(kmer_scores)
    obs_sigma_sq = variance(kmer_scores)
    print "mu:",pred_mu,obs_mu,(obs_mu-pred_mu)/obs_mu # should be very low
    print "sigma_sq:",pred_sigma_sq,obs_sigma_sq,(obs_sigma_sq-pred_sigma_sq)/obs_sigma_sq # should be very low
    Zb_obs = sum(exp(-kmer_score) for kmer_score in kmer_scores)
    Zb_pred = (4**L)*exp(-pred_mu + pred_sigma_sq/2.0)
    print Zb_pred,Zb_obs
    print (Zb_obs - Zb_pred)/Zb_obs

def bd_mean_ref(code,bd):
    kmer_scores = [score_site(code,bd,kmer) for kmer in make_kmers(L)]
    return mean(kmer_scores)

def bd_eps_sq_ref(code,bd):
    kmer_scores_sq = [score_site(code,bd,kmer)**2 for kmer in make_kmers(L)]
    return mean(kmer_scores_sq)

def bd_mean(code,bd):
    return sum([mean([code[aa,n1,n2] for (n1,n2) in nuc_pairs]) for aa in bd])

def test_bd_mean(trials=100,bd_length=L-1):
    for i in trange(trials):
        code = sample_code()
        bd = sample_bd(bd_length)
        mu_ref = bd_mean_ref(code,bd)
        mu = bd_mean(code,bd)
        print mu,mu_ref,(mu-mu_ref)/mu
        
def bd_variance_ref(code,bd):
    kmer_scores = [score_site(code,bd,kmer) for kmer in make_kmers(L)]
    return variance(kmer_scores)

def bd_variance_spec(code,bd):
    return sum(total_variance(code,bd,i) for i in range(len(bd) + 1))

def bd_variance(code,bd):
    mean_fs = [mean(code[aa,x,y] for (x,y) in nuc_pairs) for aa in bd]
    mean_sq_fs = [mean(code[aa,x,y]**2 for (x,y) in nuc_pairs) for aa in bd]
    mean_di_fs = [2*mean(code[aa1,x,y]*code[aa2,y,z] for (x,y,z) in nuc_trips) for (aa1,aa2) in pairs(bd)]
    higher_terms = [f1*f2 for i,f1 in enumerate(mean_fs) for j,f2 in enumerate(mean_fs) if abs(i-j) > 1]
    # print len(mean_sq_fs + mean_di_fs + higher_terms),len(bd)**2
    # print len(mean_sq_fs), len(mean_di_fs), len(higher_terms)
    # print [(i,j) for i,aa1 in enumerate(bd)
    #               for j,aa2 in enumerate(bd) if abs(j-i) == 1]
    # print [(i,j) for i,f1 in enumerate(mean_fs) for j,f2 in enumerate(mean_fs) if abs(i-j) > 1]
    print len(mean_sq_fs),2*len(mean_di_fs),len(higher_terms)
    eps_sq = sum(mean_sq_fs) + sum(mean_di_fs) + sum(higher_terms)
    print "eps_sq:",bd_eps_sq_ref(code,bd),eps_sq
    bd_mean_sq = bd_mean(code,bd)**2
    print "bd_mean_sq:",bd_mean_sq
    return  eps_sq - bd_mean_sq
    
def total_variance(code,bd,i):
    """find the total variance of the ith position, conditional on the
    i-1th position.
    """
    if i == 0:
        aa = bd[0]
        return variance([code[aa,x,y] for x in nucs for y in nucs])
    else:
        aa = bd[i-1]
        first_term = mean(variance([code[aa,x,y] for y in nucs]) for x in nucs)
        second_term = variance([mean(code[aa,x,y] for y in nucs) for x in nucs])
        print first_term,second_term
        return first_term + second_term
    
def test_bd_variance(trials=100,bd_length=L-1,sigma=1):
    preds = []
    obs = []
    for i in trange(trials):
        code = sample_code(sigma)
        bd = sample_bd(bd_length)
        sigma_sq_ref = bd_variance_ref(code,bd)
        sigma_sq = bd_variance(code,bd)
        #print mu,mu_ref,(mu-mu_ref)/mu
        preds.append(sigma_sq)
        obs.append(sigma_sq_ref)
    plt.scatter(preds,obs)
    max_ = max(pred+obs)
    plt.plot([0,max_],[0,max_])
    plt.show()
    #return preds,obs
        
def bd_variance_spec(code,bd):
    return sum([mean([code[aa,n1,n2]**2 for (n1,n2) in nuc_pairs]) for aa in bd]) - bd_mean(code,bd)**2

def bd_variance_spec2(code,bd):
    pass
    
def make_ringer_spec(code):
    def aa_mu(aa):
        return mean([code[aa,b1,b2] for b1,b2 in nuc_pairs])
    def aa_sigma(aa):
        return sqrt(variance([code[aa,b1,b2] for b1,b2 in nuc_pairs]))
    (aa,b1,b2),min_score = min(code.items(),key=lambda ((aa,b1,b2),score):score - aa_mu(aa) + (aa_sigma(aa)**2)/2.0)

def sample_bd(bd_length):
    bd = [random.choice(aas) for i in range(bd_length)]
    return bd
    
def sample_species():
    bd = sample_bd(L-1)
    sites = sample_sites(n,L)
    return (bd,sites)

def sample_species2():
    bd = [random.choice(aas) for i in range(L)]
    site = random_site(L)
    sites = [site for i in range(n)]
    return (bd,sites)

def sample_fitnesses(N,code,init=sample_species2):
    return [fitness(code,init()) for i in trange(N)]
    
def moran_process(code,mutation_rate,N=1000,turns=10000,
                  init=sample_species,mutate=mutate,fitness=fitness,pop=None):
    mean_rec_muts,mean_site_muts = mutation_rate/3.0,mutation_rate
    site_mu = mean_site_muts/float(n*L)
    bd_mu = mean_rec_muts/float(L)
    if pop is None:
        pop = [(lambda spec:(spec,fitness(code,spec)))(init())
               for _ in trange(N)]
    else:
        pop = pop[:]
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
            print "starting on:","mutation rate:","%1.3e" % mutation_rate,"sigma:","%1.3e" % sigma
            code = sample_code(sigma)
            ringer = make_ringer_ref(code,L)
            pop,hist = moran_process(code,mutation_rate,
                                     init=lambda :ringer,
                                     N=1000,turns=turns,
                                     mutate=mutate,fitness=fitness)
            d[mutation_rate,sigma] = (pop,hist,code)
    return d


def pop_fitness(pop,hist,code):
    return mean(fitness(code,spec) for (spec,fit) in pop)

def pop_log_relative_fitness(pop,hist,code):
    # fs = [fitness(code,spec) for (spec,fit) in pop]
    # return mean(fs)/max(fs)
    return log10(hist[-1][1]/hist[0][1])

def pop_ic(pop,hist,code):
    return mean(motif_ic(extract_sites(spec),correct=False) for (spec,fit) in pop)

def pop_mi(pop,hist,code):
    return mean(total_motif_mi(extract_sites(spec)) for (spec,fit) in pop)

def pop_total(pop,hist,code):
    return pop_ic(pop,hist,code) + pop_mi(pop,hist,code)

named_fs = [(pop_log_relative_fitness,"log relative fitness"),(pop_ic,"Motif IC"),
                    (pop_mi,"Motif MI"),(pop_total,"IC + MI")]

def interpret_trajectories(results_dict):
    for k,(pop,hist,code) in results_dict.items():
        print k
        sel_fits,mean_fits = transpose(hist[:100000])
        plt.plot(np.array(mean_fits)/mean_fits[0],label=k)
    plt.legend()
    plt.semilogy()
    plt.show()

def test_trajectories_for_stationarity(results_dict):
    for k,(pop,hist,code) in results_dict.items():
        mean_fits = [row[1] for row in hist]
        relevant_fits = mean_fits[int(len(mean_fits)*9.0/10):]
        stationary = is_stationary(relevant_fits)
        print k,stationary
        if not stationary:
            plt.plot(mean_fits,label=k)
    plt.legend()
    plt.semilogy()
    plt.show()

        
def interpret_main_experiment(results_dict):
    mutation_rates,sigmas = map(lambda x:sorted(set(x)),transpose(results_dict.keys()))
    fs,names = transpose(named_fs)
    subplot_dimension = ceil(sqrt(len(fs)))
    for idx,f in enumerate(fs):
        mat = np.zeros((len(mutation_rates),len(sigmas)))
        for i,mutation_rate in enumerate(sorted(mutation_rates)):
            for j,sigma in enumerate(sorted(sigmas)):
                pop,hist,code = results_dict[(mutation_rate,sigma)]
                mean_fits = [row[1] for row in hist]
                stationary = is_stationary(mean_fits[len(mean_fits)/2:])
                if stationary:
                    mat[i,j] = f(pop,hist,code)
                else:
                    mat[i,j] = None
                print i,j,mat[i,j]
        plt.subplot(subplot_dimension,subplot_dimension,idx)
        plt.imshow(mat,interpolation='none')
        plt.xticks(range(len(sigmas)),map(str,sigmas))
        plt.yticks(range(len(mutation_rates)),map(str,mutation_rates))
        #plt.yticks(rec_muts)
        plt.xlabel("sigma")
        plt.ylabel("mutation rate")
        plt.colorbar()
        title = names[idx]
        plt.title(title)
    plt.show()
