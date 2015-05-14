import numpy as np
import random
from simplex_sampling_experiment import make_kmers,h_np
from estremo_on_a_napkin import sample_eps,boltzmann
from utils import random_site,inverse_cdf_sample,mean,motif_ic,h,transpose,rslice,pairs,log2,total_motif_mi
from tqdm import tqdm,trange
from math import exp,log
from matplotlib import pyplot as plt
#import seaborn as sbn
import scipy
from collections import defaultdict
from sde import ou_relaxation_time_from_sample,is_stationary

n = 16
L = 5
beta = 1
K = 4**L
G = 5*10**5
# site_mut_prob = 10**-0.5
# rec_mut_prob = 10**-0.5

mean_site_muts = 10**0.5
#site_mut_prob = 10**-10
mean_rec_muts = 10**-0.25


def make_idx_of_word():
    return {w:i for i,w in enumerate(make_kmers(L))}

idx_of_word = make_idx_of_word()

def sample_sites(n=n,L=L):
    return [random_site(L) for i in range(n)]

def sample_rec():
    return np.random.randint(0,2,K)
    
def occs(sites,rec,sample=False):
    site_recs = np.array([rec[idx_of_word[site]] for site in sites])
    if sample:
        Zb = sample_Z(rec)
    else:
        Zb = background_Z(rec)
    occs = site_recs/(np.sum(site_recs) + Zb + 10**-10)
    return occs

def background_Z(rec):
    Z = np.sum(rec)
    return Z * G/float(len(rec))

def sample_Z(rec):
    background_recs = np.random.choice(rec,G)
    return np.sum(background_recs)
    
def test_background_Z():
    rec = sample_rec()
    sampled_Z = sample_Z(rec)
    pred_Z = background_Z(rec)
    return pred_Z,sampled_Z
    
def fitness((sites,rec)):
    return np.sum(occs(sites,rec))

def sites_recognized((sites,rec)):
    return sum([rec[idx_of_word[site]] for site in sites])
    
def mutate((sites,rec),site_mu,rec_mu):
    new_sites = mutate_sites(sites,site_mu)
    new_rec = mutate_rec(rec,rec_mu)
    return (new_sites,new_rec)
        
def mutate_rec_ref(rec,rec_mu):
    new_rec = np.copy(rec)
    for i in xrange(len(new_rec)):
        if random.random() < rec_mu:
            #print "mutating rec"
            new_rec[i] = 1 - new_rec[i]
    return new_rec

def mutate_rec(rec,rec_mu):
    """

    This algorithm is a slight approximation since it samples the
    number of mutations according to a Binom(K,rec_mu) random
    variable, then mutates randomly with replacement.  Collisions are
    technically possible, so effective mutation rate is slightly
    lower than advertised, by about 1/K.

    """
    new_rec = np.copy(rec)
    num_mutations = np.random.binomial(K,rec_mu)
    mut_idxs = random_combination(K,num_mutations)
    for i in mut_idxs:
        new_rec[i] = 1 - new_rec[i]
    return new_rec
    
def mutate_char(b,mu_prob):
    return b if random.random() > mu_prob else random.choice([c for c in "ACGT" if not c == b])

def mutate_char_det(b):
    return random.choice([c for c in "ACGT" if not c == b])
    
def random_combination(N,k):
    """select random k-combination of [0...N-1]

    Floyd,R.  Communications of the ACM, September 1987, Volume 30, Number 9
    """
    s = []
    for j in range(N-k,N):
        t = random.randrange(0,j+1)
        if not t in s:
            s.append(t)
        else:
            s.append(j)
    return s
    
def mutate_site(site,mu_prob):
    num_mutations = np.random.binomial(L,mu_prob)
    mut_idxs = random_combination(K,num_mutations)
    return "".join([mutate_char_det(b) if i in mut_idxs else b for i,b in enumerate(site)])

def mutate_sites(sites,mu_prob):
    return [mutate_site(site,mu_prob) for site in sites]

def sample_species():
    return (sample_sites(),sample_rec())

def make_ringer():
    return (["A"*L for _ in range(n)],np.array([1]+[0]*(K-1)))

def make_mi_ringer():
    sites = (["AT"+"A"*(L-2) for _ in range(n/2)] + 
             ["TA"+"A"*(L-2) for _ in range(n/2)])
    rec = np.zeros(K)
    for site in sites:
        rec[idx_of_word[site]] = 1
    return (sites, rec)
    
def rec_h(rec):
    p = np.sum(rec)/float(len(rec))
    return h([p,1-p])

def main_experiment(turns=10000,N=1000,hist_modulus=100,print_modulus=1000):
    site_muts = [10**-i for i in range(-2,4)]
    rec_muts = [10**-i for i in range(-2,4)]
    results_dict = {}
    for site_mut in site_muts:
        for rec_mut in rec_muts:
            print "starting on:",site_mut,rec_mut
            converged = False
            scratch_pop = None
            ring_pop = None
            scratch_hist = []
            ring_hist = []
            while not converged:
                print "ringer"
                pop,hist = moran_process(N=N,turns=turns,mean_site_muts=site_mut,mean_rec_muts=rec_mut,
                                                             init=make_ringer,pop=ring_pop,
                                                             hist_modulus=hist_modulus,print_modulus=print_modulus)
                converged = True
            results_dict[(site_mut,rec_mut)] = [pop,hist[:]]
    return results_dict

def interpret_main_experiment(results_dict,f=None):
    site_muts,rec_muts = map(lambda x:sorted(set(x)),transpose(results_dict.keys()))
    for idx in range(1,7+1):
        if idx == 6:
            f = recognizer_non_linearity
        elif idx == 7:
            f = motif_non_linearity
        mat = np.zeros((len(site_muts),len(rec_muts)))
        for i,site_mut in enumerate(sorted(site_muts)):
            for j,rec_mut in enumerate(sorted(rec_muts)):
                pop,hist = results_dict[(site_mut,rec_mut)]
                if f is None:
                    last = hist[-1]
                    mat[i,j] = last[idx]
                    print i,j,site_mut,rec_mut,mat[i,j]
                else:
                    mat[i,j] = mean([f(x) for x,fit in pop])
                    print i,j,mat[i,j]
        plt.subplot(3,3,idx)
        plt.imshow(mat,interpolation='none')
        plt.xticks(range(len(rec_muts)),map(str,rec_muts))
        plt.yticks(range(len(site_muts)),map(str,site_muts))
        #plt.yticks(rec_muts)
        plt.xlabel("rec mutation rate")
        plt.ylabel("site mutation rate")
        plt.colorbar()
        title = "turn f mean_fits mean_dna_ic mean_rec mean_recced rec_nonlinearity motif_nonlinearity".split()[idx]
        plt.title(title)
    plt.show()

def interpret_main_experiment2():
    pass
    
def moran_process(N=1000,turns=10000,mean_site_muts=1,mean_rec_muts=1,init=sample_species,mutate=mutate,
                  fitness=fitness,pop=None,print_modulus=100,hist_modulus=10):
    #ringer = (np.array([1]+[0]*(K-1)),sample_eps())
    if pop is None:
        pop = [(lambda spec:(spec,fitness(spec)))(init())
               for _ in trange(N)]
    # ringer = make_ringer()
    # pop[0] = (ringer,fitness(ringer))
    #pop = [(ringer,fitness(ringer)) for _ in xrange(N)]
    site_mu = min(1/float(n*L) * mean_site_muts,1)
    rec_mu = min(1/float(K) * mean_rec_muts,1)
    hist = []
    for turn in xrange(turns):
        fits = [f for (s,f) in pop]
        #print fits
        birth_idx = inverse_cdf_sample(range(N),fits,normalized=False)
        if birth_idx is None:
            return pop
        death_idx = random.randrange(N)
        #print birth_idx,death_idx
        mother,f = pop[birth_idx]
        daughter = mutate(mother,site_mu,rec_mu)
        #print "mutated"
        pop[death_idx] = (daughter,fitness(daughter))
        mean_fits = mean(fits)
        #hist.append((f,mean_fits))
        if turn % hist_modulus == 0:
            mean_dna_ic = mean([motif_ic(sites,correct=False) for ((sites,eps),_) in pop])
            mean_rec = mean([recognizer_promiscuity(x) for (x,f) in pop])
            mean_recced = mean([sites_recognized((dna,rec)) for ((dna,rec),_) in pop])
            hist.append((turn,f,mean_fits,mean_dna_ic,mean_rec,mean_recced))
            if turn % print_modulus == 0:
                print turn,"sel_fit:",f,"mean_fit:",mean_fits,"mean_dna_ic:",mean_dna_ic,"mean_rec_prom:",mean_rec
    return pop,hist

def rare_mutation_process(N=1000,turns=10000,mean_site_muts=1,mean_rec_muts=1,init=sample_species,x0=None):
    if x0 is None:
        x = init()
    else:
        x = x0
    f = fitness(x)
    site_mu = min(1/float(n*L) * mean_site_muts,1)
    rec_mu = min(1/float(K) * mean_rec_muts,1)
    proposed_mutants = 0
    for turn in xrange(turns):
        xp = mutate(x,site_mu,rec_mu)
        fp = fitness(xp)
        if f == fp:
            ar = 1.0/N
        else:
            proposed_mutants += 1
            ar = (1 - f/fp)/(1-(f/fp)**N)
            #print turn,f,fp,ar,proposed_mutants/float(turn+1)
        if random.random() < ar:
            print turn,"accepted:",f,"->",fp,ar,proposed_mutants/float(turn+1),sites_recognized(x),recognizer_promiscuity(x)
            x,f = xp,fp
    return x
    
def recognizer_non_linearity((sites,recognizer)):
    L = log(len(idx_of_word),4)
    motif = [w for w,i in idx_of_word.items() if recognizer[i]]
    if len(motif) == 0:
        return -1
    else:
        total_info = 2*L - log2(len(motif))
        col_info = motif_ic(motif,correct=False)
        return total_info - col_info
    
def motif_non_linearity((sites,recognizer)):
    return total_motif_mi(sites)
    
def collapsed_moran_process(N,turns,init=sample_species,mutate=mutate,fitness=fitness,ancestor=None,modulus=100):
    if ancestor is None:
        ancestor = sample_species()
    f = fitness(ancestor)
    hist = []
    for turn in xrange(turns):
        prop = mutate(ancestor)
        fp = fitness(prop)
        if f == fp:
            continue
        num = (1-f/fp)
        denom = (1-(f/fp)**N)
        transition_prob = num/denom
        # print f,fp
        # print num,denom
        # print transition_prob
        if random.random() < transition_prob:
            ancestor = prop
            f = fp
        if turn % modulus == 0:
            print (turn,f,f,motif_ic(ancestor[0],correct=False),rec_h(ancestor[1]))
            hist.append((turn,f,f,motif_ic(ancestor[0]),rec_h(ancestor[1])))
    return ancestor,hist

def recognizer_promiscuity((dna,rec)):
    return np.sum(rec)
    
def mh_moran_process(turns,beta=1,mean_site_muts=10**0.5,mean_rec_muts=10**-0.25,
                     init=sample_species,mutate=mutate,fitness=fitness,x=None,modulus=100):
    site_mu = min(1/float(n*L) * mean_site_muts,1)
    rec_mu = min(1/float(K) * mean_rec_muts,1)
    print "site_mu:",site_mu,"rec_mu:",rec_mu
    print "mean_site_muts:",mean_site_muts,"mean_rec_muts:",mean_rec_muts
    print "mut_prob:",mean_site_muts + mean_rec_muts

    if x is None:
        x = sample_species()
    f = fitness(x)
    hist = []
    accs = 0
    disads = 0
    for turn in xrange(turns):
        xp = mutate(x,site_mu,rec_mu)
        fp = fitness(xp)
        log_transition_prob = (beta*(fp-f)) # assuming fitness behaves as state energy;mutation probs cancel, since equal
        # print f,fp
        # print num,denom
        # print transition_prob
        log_ratio = (log(fp+10**-100)-log(f)) * beta
        #print ratio
        if log(random.random()) < log_ratio:#log(random.random()) < log_transition_prob:
            x = xp
            f = fp
            accs += 1
            if log_transition_prob < 0:
                disads += 1
        if turn % modulus == 0:
            mot_ic = motif_ic(x[0],correct=False)
            rec_spec = recognizer_promiscuity(x)
            sites_recced = sites_recognized(x)
            print (turn,f,f,mot_ic,rec_spec,sites_recced),accs/float(turn+1),disads/float(accs+1)
            hist.append((turn,f,f,mot_ic,rec_spec,sites_recced))
    return x,hist

def param_scan(mean_site_muts,n,turns=1000000):
    results = defaultdict(list)
    for mean_site_mut in mean_site_muts:
        for trial in xrange(n):
            results[mean_site_mut].append(mh_moran_process(turns,beta=10**8,modulus=10000
                                                           ,mean_site_muts=mean_site_mut))
        print len(results[mean_site_mut])
    return results

def plot_param_scan(results):
    colors = "bgrcmyk"
    jitter = lambda x:x+random.gauss(0,0.01)
    for i,mut_value in enumerate(results):
        for j,(x,hist) in enumerate(results[mut_value]):
            _,_,fit,_,rec_spec,sites_recced = hist[-1]
            print fit,log(fit)
            plt.scatter([jitter(rec_spec)],[jitter(sites_recced)],color=colors[i],
                        label=mut_value if j == 0 else None,linewidth=100*(fit))
    plt.xlabel("Recognize specificity (in sites)")
    plt.ylabel("Sites Recognized")
    plt.legend()
    plt.show()
    
def plot_hist(hist,show=True,labels=True):
    transposed_hist = transpose(hist)
        #hist.append((turn,f,mean_fits,mean_dna_ic,mean_rec,mean_recced))
    plt.plot(transposed_hist[0],transposed_hist[1],label="sampled fitness"*labels,color='b')
    plt.plot(transposed_hist[0],transposed_hist[2],label="mean fitness"*labels,color='g')
    plt.plot(transposed_hist[0],transposed_hist[3],label="mean motif ic"*labels,color='r')
    plt.plot(transposed_hist[0],transposed_hist[4],label="rec prom"*labels,color='y')
    plt.plot(transposed_hist[0],transposed_hist[5],label="sites recced"*labels,color='m')
    #plt.semilogy()
    if labels:
        plt.legend()
    if show:
        plt.show()

def brownian_test(xs):
    """given a time series, test for gaussian increments by Shapiro-Wilks test.  """
    dxs = [x1-x0 for (x0,x1) in pairs(xs)]
    return scipy.stats.shapiro(dxs)

def estimate_drift_diffusion(xs):
    """Given a sample path Xt, estimate mu, sigma for:
    dX = mu + sigma dW"""
    dxs = [x1-x0 for (x0,x1) in pairs(xs)]
    mu = mean(dxs)
    sigma = sd(dxs)
    return mu,sigma

def resample_drift_diffusion(xs):
    mu,sigma = estimate_drift_diffusion(xs)
    ys = [xs[0]]
    for x in xs[1:]:
        ys.append(ys[-1] + mu + sigma*random.gauss(0,1))
    return ys
    
def bm(n,t):
    """Return sample path of standard brownian motion, sampled n times up until time t"""
    W = [0]
    dt = t/float(n)
    for i in range(n-1):
        W.append(W[-1] + random.gauss(0,sqrt(dt)))
    return W

