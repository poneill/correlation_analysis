# Tue Oct 20 19:56:06 EDT 2015

# The purpose of this analysis is to determine whether IC-matched
# evosims have higher Gini coefficients than their biological
# counterparts.

from utils import motif_ic, secant_interval, make_pssm, mmap, choose, log_choose, mean, transpose, sd
from utils import pairs, bisect_interval, show, h, normalize, inverse_cdf_sample, motif_hamming_distance
from utils import mh
from pwm_utils import matrix_from_motif, sigma_from_matrix
from linear_gaussian_ensemble import ringer_motif, fitness, log_fitness, sample_matrix
from linear_gaussian_ensemble_gini_analysis import mutate_motif_k_times,sella_hirsch_mh
from math import log, exp
from motifs import Escherichia_coli
from scipy.stats import pearsonr
from tqdm import *
from matplotlib import pyplot as plt

G = 5*10**6

def rho_weight(rho,n,L):
    return choose(n*L,rho)*3**rho

def log_rho_weight(rho, n, L):
    return log_choose(n*L,rho) + rho*3

def logsum(log_xs):
    """given an input array of the form log(x_i), return log(sum(xs))"""
    return reduce(lambda logx,logy:logx + log(1+exp(logy-logx)),log_xs)

def logmean(log_xs):
    """given an input array of the form log(x_i), return log(mean(xs))"""
    n = len(log_xs)
    return logsum(log_xs) - log(n)
    
def estimate_stationary_statistic(matrix,n,Ne,T,samples_per_bin=10):
    """given matrix, Ne and statistic T, estimate <T> under stationary
    distribution"""
    L = len(matrix)
    N = n*L
    nu = Ne - 1
    ringer = ringer_motif(matrix,n)
    all_sampless = [[mutate_motif_k_times(ringer,k) for i in range(samples_per_bin)] for k in trange(N)]
    Tss = mmap(T,all_sampless)
    log_fss = mmap(lambda motif:log_fitness(matrix,motif,G),all_sampless)
     # better expressed as exp(nu*log(f)), but numeric issues
    log_bz_weightss = [[(nu*log_f) for log_f in log_fs] for rho,log_fs in enumerate(log_fss)]
    #Z = sum([mean(bz_weights)*rho_weight(rho,n,L) for rho,bz_weights in enumerate(bz_weightss)])
    log_Z = logsum([logmean(log_bz_weights) + log_rho_weight(rho,n,L)
                    for rho,log_bz_weights in enumerate(log_bz_weightss)])
    # summands = [rho_weight(rho,n,L)*mean(t*bz_weight/Z for t,bz_weight in zip(ts,bz_weights))
    #             for rho,(ts,bz_weights) in enumerate(zip(Tss,bz_weightss))]
    log_summands = logsum([logmean([log(t*exp(log_bz_w)) for t,log_bz_w in zip(ts,log_bz_weights)])
                           + log_rho_weight(rho,n,L)
                           for rho,(ts,log_bz_weights) in enumerate(zip(Tss,log_bz_weightss))])
    return exp(log_summands - log_Z)
    #return sum(summands)

def estimate_stationary_statistic_ref(matrix,n,Ne,T,samples_per_bin=10):
    """given matrix, Ne and statistic T, estimate <T> under stationary
    distribution"""
    L = len(matrix)
    N = n*L
    nu = Ne - 1
    ringer = ringer_motif(matrix,n)
    all_samples = [(rho,mutate_motif_k_times(ringer,rho)) for i in trange(samples_per_bin) for rho in xrange(N)]
    log_accs = []
    log_Zs = []
    for rho,motif in all_samples:
        log_f = log_fitness(matrix,motif,G)
        t = T(motif)
        if t < 0:
            #print "continuing"
            continue
        log_t = log(t)
        log_w = log_rho_weight(rho,n,L)
        log_term = log_w + (nu*log_f)
        numer = log_term + log_t
        log_accs.append(numer)
        log_Zs.append(log_term)
    return exp(logsum(log_accs) - logsum(log_Zs))

def test_estimate_stationary_statistic_ref(matrix,n,Ne,T,samples_per_bin=10):
    matrix,chain = sella_hirsch_mh(Ne=Ne,n=n,matrix=matrix,init='ringer')
    mcmc_ics = map(motif_ic,chain)
    pred_ic = estimate_stationary_statistic_ref(matrix,n,Ne,T=motif_ic)
    return pred_ic,mean(mcmc_ics)
    
def test_estimate_stationary_statistic_ref_framework():
    matrix = make_pssm(Escherichia_coli.LexA)
    n = len(Escherichia_coli.LexA)
    Nes = np.linspace(1,5,10)
    pred,obs = transpose([test_estimate_stationary_statistic_ref(matrix,n,Ne,T=motif_ic) for Ne in Nes])
    plt.plot(Nes,pred)
    plt.plot(Nes,obs)
    return pred,obs
    
def Ne_from_motif(bio_motif,interp_rounds,iterations=50000):
    """Given a motif, return Ne that matches mean IC"""
    bio_ic = motif_ic(bio_motif)
    n = len(bio_motif)
    L = len(bio_motif[0])
    matrix = [[-ep for ep in row] for row in  make_pssm(bio_motif)]
    print len(matrix)
    def f(Ne,iterations=iterations):
        print "Ne",Ne
        _,chain = sella_hirsch_mh(matrix=matrix,n=n,Ne=Ne,iterations=iterations,init='ringer')
        return mean(map(motif_ic,chain[iterations/2:])) - bio_ic
    # lo,hi = 1,5
    # data = []
    # for _ in xrange(interp_rounds):
    #     guess = (lo + hi)/2.0
    #     y = f(guess)
    #     print lo,hi,guess,y
    #     data.append((guess,y))
    #     if y > 0:
    #         hi = guess
    #     else:
    #         lo = guess
    # return data
    Ne_min = 1
    Ne_max = 5
    while f(Ne_max) < 0:
        print "increasing Ne max"
        Ne_max *= 2
    xs, ys= transpose([(Ne,f(Ne)) for Ne in np.linspace(Ne_min,Ne_max,interp_rounds)])
    # now find an interpolant.  We desire smallest sigma of gaussian
    # interpolant such that function has at most one inflection point
    interp_sigmas = np.linspace(0.01,1,100)
    interps = [gaussian_interp(xs,ys,sigma=s) for s in interp_sigmas]
    for i,(sigma, interp) in enumerate(zip(interp_sigmas,interps)):
        print i,sigma
        if num_inflection_points(map(interp,np.linspace(Ne_min,Ne_max,100))) == 1:
            "found 1 inflection point"
            break
    print sigma
    Ne = bisect_interval(interp,Ne_min,Ne_max)
    return Ne
    
def gaussian_interp(xs,ys,sigma=1):
    def f(xp):
        Z = sum(exp(-(x-xp)**2/(2*sigma**2)) for x in xs)
        return sum(y*exp(-(x-xp)**2/(2*sigma**2)) for x,y in zip(xs,ys))/Z
    return f

def num_inflection_points(ys):
    ysp = [y2 - y1 for y1,y2 in pairs(ys)]
    yspp = [y2 - y1 for y1,y2 in pairs(ysp)]
    sign_changes = 0
    for y1,y2 in pairs(yspp):
        if y1*y2 < 0:
            sign_changes += 1
    return sign_changes

def find_all_Nes():
    Nes = {}
    for i, tf in enumerate(Escherichia_coli.tfs):
        print "starting on: %s (%s/%s)" % (tf,i,len(Escherichia_coli.tfs))
        Nes[tf] = Ne_from_motif(getattr(Escherichia_coli,tf),interp_rounds=10)
        print "found:",tf,Nes[tf]
    return Nes

def analyze_bio_motifs(Nes,trials=20):
    results = {}
    for tf_idx,tf in enumerate(Escherichia_coli.tfs):
        Ne = Nes[tf]
        bio_motif = getattr(Escherichia_coli,tf)
        n,L = len(bio_motif),len(bio_motif[0])
        bio_matrix = matrix_from_motif(bio_motif)
        sigma = sigma_from_matrix(bio_matrix)
        matrix_chains = [sella_hirsch_mh(n=n,L=L,sigma=sigma,Ne=Ne,init='ringer') for i in range(trials)]
        ics = [mean(map(motif_ic,chain[-1000:])) for (matrix,chain) in matrix_chains]
        ginis = [mean(map(motif_gini,chain[-1000:])) for (matrix,chain) in matrix_chains]
        mis = [mean(map(total_motif_mi,chain[-1000:])) for (matrix,chain) in matrix_chains]
        print "results for:",tf,tf_idx
        print motif_ic(bio_motif),mean(ics),sd(ics)
        print motif_gini(bio_motif),mean(ginis),sd(ginis)
        print total_motif_mi(bio_motif),mean(mis),sd(mis)
        results[tf] = (mean(ics),sd(ics),mean(ginis),sd(ginis),mean(mis),sd(mis))
    return results

def results_of_analyze_bio_motifs(results):
    # IC
    Ls = np.array([len(getattr(Escherichia_coli,tf)[0]) for tf in Escherichia_coli.tfs])
    Ls_choose_2 = np.array([choose(L,2) for L in Ls])
    bio_ics = np.array([motif_ic(getattr(Escherichia_coli,tf)) for tf in Escherichia_coli.tfs])
    sim_ics = np.array([results[tf][0] for tf in Escherichia_coli.tfs])
    sim_ic_errs = np.array([1.96*results[tf][1] for tf in Escherichia_coli.tfs])
    bio_ics_norm = bio_ics/Ls
    sim_ics_norm = sim_ics/Ls
    sim_ic_norm_errs = sim_ic_errs/Ls
    bio_ginis = np.array([motif_gini(getattr(Escherichia_coli,tf)) for tf in Escherichia_coli.tfs])
    sim_ginis = np.array([results[tf][2] for tf in Escherichia_coli.tfs])
    sim_gini_errs = np.array([1.96*results[tf][3] for tf in Escherichia_coli.tfs])
    bio_mis_norm = np.array([total_motif_mi(getattr(Escherichia_coli,tf))/choose(L,2)
                    for tf,L in zip(Escherichia_coli.tfs,Ls)])
    sim_mis_norm = np.array([results[tf][4]/choose(L,2) for tf,L in zip(Escherichia_coli.tfs,Ls)])
    sim_mis_norm_errs = np.array([1.96*results[tf][5]/choose(L,2) for tf,L in zip(Escherichia_coli.tfs,Ls)])
    plt.subplot(1,4,1)
    plt.errorbar(bio_ics,sim_ics,
                  yerr=sim_ic_errs,fmt='o')
    plt.plot([0,20],[0,20])
    plt.xlabel("IC")
    
    plt.subplot(1,4,2)
    plt.errorbar(bio_ics_norm,sim_ics_norm,
                  yerr=sim_ic_norm_errs,fmt='o')
    plt.plot([0,2],[0,2])
    plt.xlabel("IC/base")
    
    plt.subplot(1,4,3)
    plt.errorbar(bio_ginis,sim_ginis,
                yerr=sim_gini_errs,fmt='o')
    plt.plot([0,1],[0,1])
    plt.xlabel("Gini coefficient")
    
    plt.subplot(1,4,4)
    plt.errorbar(bio_mis_norm,sim_mis_norm,
                 yerr=sim_mis_norm_errs,fmt='o')
    plt.plot([0,0.5],[0,0.5])
    plt.xlabel("MI/pair")
    print "IC:", pearsonr(bio_ics, sim_ics)
    print "normalized IC:", pearsonr(bio_ics_norm, sim_ics_norm)
    print "Gini:", pearsonr(bio_ginis, sim_ginis)
    print "normalized MI:", pearsonr(bio_mis_norm, sim_mis_norm)

def sample_motif_neglect_fg(matrix,n,Ne,pss=None):
    nu = Ne - 1
    if pss == None:
        pss = [normalize([exp(-nu*ep) for ep in col]) for col in matrix]
    def sample_site():
        return "".join([inverse_cdf_sample("ACGT",ps) for ps in pss])
    return [sample_site() for i in range(n)]

def dsample_motif_neglect_fg(matrix,motif,Ne,pss=None):
    """return log probability of motif"""
    # could stand to optimize this if need be!
    nu = Ne - 1
    if pss == None:
        pss = [normalize([exp(-nu*ep) for ep in col]) for col in matrix]
    def log_prob_site(site):
        return sum(log(ps["ACGT".index(b)]) for b,ps in zip(site,pss))
    return sum(map(log_prob_site,motif))
    
def validate_sample_motif_neglect_fg():
    """compare fg_neglect sampling to random mutation: indeed shows better fitness at given rho"""
    bio_motif = Escherichia_coli.LexA
    n = len(bio_motif)
    L = len(bio_motif[0])
    matrix = [[-ep for ep in row] for row in make_pssm(bio_motif)]
    ringer = ringer_motif(matrix,n)
    random_motifs = [mutate_motif_k_times(ringer,k) for k in range(n*L)]
    random_motifs2 = [sample_motif_neglect_fg(matrix,n,Ne) for Ne in np.linspace(1,10,n*L)]
    random_rhos = [motif_hamming_distance(ringer,motif) for motif in random_motifs]
    random_log_fs = [log_fitness(matrix,motif,G) for motif in random_motifs]
    random_rhos2 = [motif_hamming_distance(ringer,motif) for motif in random_motifs2]
    random_log_fs2 = [log_fitness(matrix,motif,G) for motif in random_motifs2]
    plt.plot(random_rhos,random_log_fs)
    plt.plot(random_rhos2,random_log_fs2)
    plt.plot(random_rhos3,random_log_fs3)

def validate_sample_motif_neglect_fg2(iterations=50000):
    """compare fg_neglect sampling to MCMC"""
    bio_motif = Escherichia_coli.LexA
    n = len(bio_motif)
    L = len(bio_motif[0])
    matrix = [[-ep for ep in row] for row in make_pssm(bio_motif)]
    ringer = ringer_motif(matrix,n)
    Ne = 2.375 
    random_motifs = [sample_motif_neglect_fg(matrix,n,Ne) for i in trange(iterations)]
    random_rhos = [motif_hamming_distance(ringer,motif) for motif in tqdm(random_motifs)]
    random_log_fs = [log_fitness(matrix,motif,G) for motif in tqdm(random_motifs)]
    random_ics = map(motif_ic,random_motifs)
    _, chain = sella_hirsch_mh(matrix=matrix,init="ringer",Ne=Ne,n=n,iterations=iterations)
    chain_rhos = [motif_hamming_distance(ringer,motif) for motif in tqdm(chain)]
    chain_log_fs = [log_fitness(matrix,motif,G) for motif in tqdm(chain)]
    chain_ics = map(motif_ic,chain)
    plt.subplot(1,2,1)
    plt.scatter(random_rhos,random_log_fs)
    plt.scatter(chain_rhos,chain_log_fs,color='g')
    plt.xlabel("rho")
    plt.ylabel("log fitness")
    plt.subplot(1,2,2)
    plt.scatter(random_rhos,random_ics)
    plt.scatter(chain_rhos,chain_ics,color='g')
    plt.xlabel("rho")
    plt.ylabel("IC")

def stationary_stat_neglect_fg(matrix,n,Ne,T,samples=1000):
    acc = 0
    Z = 0
    ws = []
    for sample in trange(samples):
        motif = sample_motif_neglect_fg(matrix,n,Ne)
        log_f = log_fitness(matrix,motif,G)
        log_q = dsample_motif_neglect_fg(matrix,motif,Ne)
        t = T(motif)
        w = exp(log_f - log_q)
        acc += t * w
        Z += w
        print acc/Z
        ws.append(w)
    print "entropy of samples:",h(normalize(ws)), log(samples) - h(normalize(ws))
    return acc/Z

def visualize_stationary_sum(matrix,n,Ne,T,samples_per_bin=100):
    L = len(matrix)
    nu = Ne - 1
    ringer = ringer_motif(matrix,n)
    motifss = [[mutate_motif_k_times(ringer,k) for i in range(samples_per_bin)] for k in trange(n*L)]
    log_fss = mmap(lambda motif:log_fitness(matrix,motif,G),tqdm(motifss))
    Tss = mmap(T,tqdm(motifss))
    log_ws = [log_rho_weight(rho,n,L) for rho in range(n*L)]
    terms = [mean(exp(nu*log_f + log_w)*T for log_f,T in zip(log_fs,Ts))
             for log_w,log_fs,Ts in zip(log_ws,log_fss,Tss)]
    Z = sum([mean(exp(nu*log_f + log_w) for log_f,T in zip(log_fs,Ts))
             for log_w,log_fs,Ts in zip(log_ws,log_fss,Tss)])
    print sum(terms)/Z
    plt.plot(range(n*L),terms)
    #plt.semilogy()

def sella_hirsch_imh(matrix,n,Ne,iterations=50000):
    f = lambda motif:log_fitness(matrix,motif,G)
    nu = Ne - 1
    pss = [normalize([exp(-nu*ep) for ep in col]) for col in matrix]
    rq = lambda motif:sample_motif_neglect_fg(matrix,n,Ne,pss=pss)
    dq = lambda motif_prime,motif: dsample_motif_neglect_fg(matrix,motif_prime,Ne,pss=pss)
    return matrix, mh(f,rq,rq(None),dprop=dq,use_log=True)

def naive_spoof(motif):
    n = len(motif)
    pss = [[col.count(b)/float(n) for b in "ACGT"] for col in transpose(motif)]
    def sample_site():
        return "".join(inverse_cdf_sample("ACGT",ps) for ps in pss)
    return [sample_site() for site in motif]

def bounded_interval_ci(xs,a,b,alpha=0.05):
    """get ci for x bounded on [a,b]"""
    # see http://arxiv.org/pdf/0802.3458v6.pdf for details
    zs = [(x-a)/float(b-a) for x in xs]
    n = len(zs)
    zhat = mean(zs)
    c = 9/(2*log(2/alpha))
    L = zhat + 3/(4+n*c)*(1 - 2*zhat - sqrt(1 + n*zhat*(1-zhat)))
    U = zhat + 3/(4+n*c)*(1 - 2*zhat + sqrt(1 + n*zhat*(1-zhat)))
    return L*(b-a) + a,U*(b-a) + a

