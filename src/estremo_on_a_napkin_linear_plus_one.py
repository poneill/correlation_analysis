import random
from estremo_on_a_napkin_pairwise import moran_process,sample_species,mutate,fitness,sample_bd,sample_sites,ln_mean,interpret_main_experiment,extract_sites
from tqdm import *
from utils import pairs,mean,variance,inverse_cdf_sample,motif_ic,total_motif_mi,choose2
from math import sqrt,exp,log

n = 16
L = 5
beta = 1
K = 4**L
G = 5*10**6
num_aas = 20
aas = range(num_aas)
nucs = "ACGT"
nuc_pairs = [(b1,b2) for b1 in nucs for b2 in nucs]
nuc_trips = [(b1,b2,b3) for b1 in nucs for b2 in nucs for b3 in nucs]
log10 = lambda x:log(x,10)

def sample_species():
    bd = sample_bd(L+1)
    sites = sample_sites(n,L)
    return (bd,sites)

def sample_species2():
    bd = sample_bd(L+1)
    site = sample_sites(1,L)[0]
    sites = [site for i in range(n)]
    return (bd,sites)
    
def sample_code(li_sigma=1,bi_sigma=1):
    li_code = {(aa,b):random.gauss(0,li_sigma) for aa in aas for b in nucs}
    bi_code = {(aa,b1,b2):random.gauss(0,bi_sigma) for aa in aas for (b1,b2) in nuc_pairs}
    return li_code,bi_code

def make_ringer((li_code,bi_code)):
    def li_aa_mu(aa):
        return mean([li_code[aa,b] for b in "ACGT"])
    def li_aa_sigma(aa):
        return sqrt(variance([li_code[aa,b] for b in "ACGT"]))
    def bi_aa_mu(aa1,aa2,aa12):
        return mean([bi_code[aa12,b1,b2] + li_code[aa1,b2] + li_code[aa2,b2] for b1,b2 in nuc_pairs])
    def bi_aa_sigma(aa1,aa2,aa12):
        return sqrt(variance([bi_code[aa12,b1,b2] + li_code[aa1,b2] + li_code[aa2,b2] for b1,b2 in nuc_pairs]))
    li_f = lambda ((aa,b),score):score - li_aa_mu(aa) + (li_aa_sigma(aa)**2)/2.0
    (li_aa,li_b),min_score = min(li_code.items(),key=li_f)
    bi_f = lambda (aa1,aa2,aa12,b1,b2):(bi_code[aa12,b1,b2] + li_code[aa1,b1] + li_code[aa2,b2]
                                                - bi_aa_mu(aa1,aa2,aa12)
                                                + (bi_aa_sigma(aa1,aa2,aa12)**2)/2.0)
    (aa1,aa2,aa12,b1,b2) = min([(aa1,aa2,aa12,b1,b2) for aa1 in aas for aa2 in aas for aa12 in aas
                                  for b1 in nucs for b2 in nucs],key=bi_f)

    bd = [li_aa]*(L-2) + [aa1,aa2,aa12]
    site = "".join([li_b]*(L-2) + [b1,b2])
    sites = [site for i in range(n)]
    return bd,sites
    
def score_site((li_code,bi_code),bd,site):
    return sum(li_code[aa,n] for (aa,n) in zip(bd,site)) + bi_code[bd[-1],site[-2],site[-1]]

def occs((li_code,bi_code),bd,sites):
    site_energies = [score_site((li_code,bi_code),bd,site) for site in sites]
    #print "test background"
    #background = np.matrix([score_site(code,bd,random_site(L)) for i in trange(G)])
    #print "finish test background"
    mu = sum([mean([li_code[aa,b] for b in nucs]) for aa in bd]) + mean(bi_code[bd[-1],b1,b2] for b1,b2 in nuc_pairs)
    sigma = sqrt(sum([variance([li_code[aa,b] for b in nucs]) for aa in bd]) +
                 variance([bi_code[bd[-1],b1,b2] for b1,b2 in nuc_pairs])) # XXX revisit w/ bd_variance
    fg = sum(exp(-ep) for ep in site_energies)
    #test_bg = np.sum(np.exp(-background))
    bg = ln_mean(-mu,sigma)*G
    #print "error: %1.2f" % ((bg - test_bg)/test_bg * 100)
    return fg/(fg+bg)

def fitness(code,(bd,sites)):
    return occs(code,bd,sites)
    
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

def main_experiment():
    taus = [0.01,0.1,1,10]
    d = {}
    #turns = 10
    turns=1000000
    for tau in taus:
        code = sample_code(bi_sigma=tau)
        ringer = make_ringer(code)
        pop,hist = moran_process(code,0.1,init=lambda:ringer,turns=turns)
        d[tau] = (pop,hist)
    return d

def plot_main_experiment_trajectories(results_dict):
    for tau,(pop,hist) in results_dict.items():
        print tau
        plt.plot(transpose(hist)[1],label=tau)
    plt.legend()
    plt.semilogy()
    plt.show()

def interpret_main_experiment(results_dict):
    taus = sorted(results_dict.keys())
    print taus
    data = [(tau,f,motif_ic(extract_sites(s)),total_motif_mi(extract_sites(s)))
            for tau in taus for (s,f) in results_dict[tau][0]]
    cols = transpose(data)
    names = "tau,f,motif_ic,total_motif_mi".split(",")
    for (i,name1),(j,name2) in choose2(list(enumerate(names))):
        xs = cols[i]
        ys = cols[j]
        print name1,name2,pearsonr(xs,ys),spearmanr(xs,ys)

