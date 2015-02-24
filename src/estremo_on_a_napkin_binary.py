import numpy as np
import random
from simplex_sampling_experiment import make_kmers,h_np
from estremo_on_a_napkin import sample_eps,boltzmann
from utils import random_site,inverse_cdf_sample,mean,motif_ic,h,transpose
from tqdm import tqdm,trange
from math import exp,log
from matplotlib import pyplot as plt
import seaborn as sbn

n = 16
L = 5
beta = 1
K = 4**L
G = 5*10**6
# site_mut_prob = 10**-0.5
# rec_mut_prob = 10**-0.5

site_mut_prob = 10**-0.5 * 0
#site_mut_prob = 10**-10
rec_mut_prob = 10**-0.25

mut_prob = 1-(1-site_mut_prob)*(1-rec_mut_prob)
print "mut_prob:",mut_prob

site_mu = 1/float(n*L) * site_mut_prob
rec_mu = 1/float(K) * rec_mut_prob 

def make_idx_of_word():
    return {w:i for i,w in enumerate(make_kmers(L))}

idx_of_word = make_idx_of_word()

def sample_sites(n=n,L=L):
    return [random_site(L) for i in range(n)]

def sample_rec():
    return np.random.randint(0,2,K)
    
def occs(sites,rec):
    site_recs = np.array([rec[idx_of_word[site]] for site in sites])
    Zb = background_Z(rec)
    occs = site_recs/(np.sum(site_recs) + Zb + 10**-10)
    return occs
    
def background_Z(rec):
    Z = np.sum(rec)
    return Z * G/float(len(rec))

def sample_Z(rec):
    backround_recs = np.random.choice(rec,G)
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
    
def mutate((sites,rec)):
    new_sites = mutate_sites(sites,site_mu)
    new_rec = mutate_rec(rec,rec_mu)
    return (new_sites,new_rec)
        
def mutate_rec(rec,rec_mu):
    new_rec = np.copy(rec)
    for i in xrange(len(new_rec)):
        if random.random() < rec_mu:
            #print "mutating rec"
            new_rec[i] = 1 - new_rec[i]
    return new_rec
    
def mutate_char(b,mu_prob):
    return b if random.random() > mu_prob else random.choice([c for c in "ACGT" if not c == b])
    
def mutate_site(site,mu_prob):
    return "".join(mutate_char(b,mu_prob) for b in site)

def mutate_sites(sites,mu_prob):
    return [mutate_site(site,mu_prob) for site in sites]

def sample_species():
    return (sample_sites(),sample_rec())

def make_ringer():
    return (["A"*L for _ in range(n)],np.array([1]+[0]*(K-1)))

def rec_h(rec):
    p = np.sum(rec)/float(len(rec))
    return h([p,1-p])
    
def moran_process(N=1000,turns=10000,init=sample_species,mutate=mutate,fitness=fitness,pop=None,modulus=100):
    #ringer = (np.array([1]+[0]*(K-1)),sample_eps())
    if pop is None:
        pop = [(lambda spec:(spec,fitness(spec)))(sample_species())
               for _ in trange(N)]
    # ringer = make_ringer()
    # pop[0] = (ringer,fitness(ringer))
    #pop = [(ringer,fitness(ringer)) for _ in xrange(N)]
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
        daughter = mutate(mother)
        #print "mutated"
        pop[death_idx] = (daughter,fitness(daughter))
        mean_fits = mean(fits)
        #hist.append((f,mean_fits))
        if turn % modulus == 0:
            mean_dna_ic = mean([motif_ic(sites,correct=False) for ((sites,eps),_) in pop])
            mean_rec = mean([rec_h(rec) for ((dna,rec),_) in pop])
            hist.append((turn,f,mean_fits,mean_dna_ic,mean_rec))
            print turn,"sel_fit:",f,"mean_fit:",mean_fits,"mean_dna_ic:",mean_dna_ic,"mean_rec_h:",mean_rec
    return pop,hist

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

def mh_moran_process(turns,beta=1,init=sample_species,mutate=mutate,fitness=fitness,x=None,modulus=100):
    if x is None:
        x = sample_species()
    f = fitness(x)
    hist = []
    accs = 0
    disads = 0
    for turn in xrange(turns):
        xp = mutate(x)
        fp = fitness(xp)
        log_transition_prob = (beta*(fp-f)) # assuming fitness behaves as state energy;mutation probs cancel, since equal
        # print f,fp
        # print num,denom
        # print transition_prob
        if log(random.random()) < log_transition_prob:
            x = xp
            f = fp
            accs += 1
            if log_transition_prob < 0:
                disads += 1
        if turn % modulus == 0:
            mot_ic = motif_ic(x[0],correct=False)
            rec_spec = np.sum(x[1])
            sites_recced = sites_recognized(x)
            print (turn,f,f,mot_ic,rec_spec,sites_recced),accs/float(turn+1),disads/float(accs+1)
            hist.append((turn,f,f,mot_ic,rec_spec,sites_recced))
    return x,hist

def plot_hist(hist,show=True):
    plt.plot(transpose(hist)[0],transpose(hist)[1])
    plt.plot(transpose(hist)[0],transpose(hist)[2],label="fitness")
    plt.plot(transpose(hist)[0],transpose(hist)[3],label="motif ic")
    plt.plot(transpose(hist)[0],transpose(hist)[4],label="rec spec")
    plt.plot(transpose(hist)[0],transpose(hist)[5],label="sites recced")
    plt.semilogy()
    plt.legend()
    if show:
        plt.show()
