from utils import motif_ic,motif_gini,total_motif_mi,mutate_motif_p,mean,transpose,iterate,iterate_list
from utils import motif_hamming_distance,gini, pl, normalize, inverse_cdf_sample, subst_motif, log_choose
from utils import mutate_motif_p_exact, choose, subst, concat, find, mmap,binary_find
from linear_gaussian_ensemble import *
import sys
sys.path.append("/home/pat/Dropbox/weighted_ensemble_motif_analysis")
from maxent_motif_sampling import maxent_sample_motifs_with_ic
from matplotlib import pyplot as plt
import itertools
from scipy import polyfit,poly1d
from math import sqrt

G = 5*10**6
n,L = 16,16

def sella_hirsch_mh(Ne=5,n=16,L=16,G=5*10**6,sigma=1,init="random",
                                             matrix=None,x0=None,iterations=50000,p=None):
    print "p:",p
    if matrix is None:
        matrix = sample_matrix(L,sigma)
    if x0 is None:
        if init == "random":
            x0 = random_motif(L,n)
        elif init == "ringer":
            x0 = ringer_motif(matrix,n)
        elif init == "anti_ringer":
            x0 = anti_ringer_motif(matrix,n)
        else:
            x0 = init
    if p is None:
        p = 1.0/(n*L)
    nu = Ne - 1
    def log_f(motif):
        return nu * log_fitness(matrix,motif,G)
    def prop(motif):
        motif_p = mutate_motif_p(motif,p) # probability of mutation per basepair
        return motif_p
    chain = mh(log_f,prop,x0,use_log=True,iterations=iterations)
    return matrix,chain
    
def sella_hirsch_imh(Ne=5,n=16,L=16,G=5*10**6,sigma=1,init="random",
                                             matrix=None,x0=None,iterations=50000,p=None,lamb=1,return_ar=False):
    """Independent Metropolis Hastings with proposal density geometrically
    distributed in # mutations from ringer"""
    print "p:",p
    if matrix is None:
        matrix = sample_matrix(L,sigma)
    if x0 is None:
        if init == "random":
            x0 = random_motif(L,n)
        elif init == "ringer":
            x0 = ringer_motif(matrix,n)
        else:
            x0 = init
    if p is None:
        p = 1.0/(n*L)
    nu = Ne - 1
    N = n * L
    ringer = ringer_motif(matrix,n)
    ps = normalize([exp(-lamb*i) for i in range(N)])
    def log_f(motif):
        return nu * log_fitness(matrix,motif,G)
    # def log_f(motif):
    #     return log(random.random())
    def prop(motif):
        # determine number of mutations to perform
        #k = discrete_exponential_sampler(N,lamb)
        k = inverse_cdf_sample(range(N),ps)
        #motif_p = mutate_motif_p_exact(ringer,p) # probability of mutation per basepair
        motif_p = mutate_motif_k_times(ringer,k)
        return motif_p
    matrix_probs = [normalize([exp(-lamb*ep) for ep in row]) for row in matrix]
    def prop_fanciful(motif):
        return ["".join([inverse_cdf_sample("ACGT",probs) for probs in matrix_probs]) for i in xrange(n)]
    def dprop(motif,_):
        k = motif_hamming_distance(ringer,motif) # number of mutations from ringer
        #return choose_reference(N,k) * p**k * (1-p)**(N-k) * (1/3.0)**k
        #return 1.0/N * (1/3.0)**k
        return ps[k] * choose(N,k) * (1/3.0)**k
    def log_dprop(motif,_):
        k = motif_hamming_distance(ringer,motif) # number of mutations from ringer
        #return log_choose(N,k) + k * log(p) + (N-k)*log(1-p) + k*log(1/3.0)
        #return -log(N) + k * log(1/3.0)
        return log(ps[k]) - log_choose(N,k) + k * log(1/3.0)
    def log_dprop_fanciful(motif,_):
        return sum(log(matrix_probs[i]["ACGT".index(b)]) for site in motif for i,b in enumerate(site))
    chain = mh(log_f,prop_fanciful,x0,dprop=log_dprop_fanciful,use_log=True,iterations=iterations,verbose=False,
               return_ar=return_ar)
    return matrix,chain

def sample_log_odds(matrix,n,lamb=1):
    matrix_probs = [normalize([exp(-lamb*ep) for ep in row]) for row in matrix]
    return ["".join([inverse_cdf_sample("ACGT",probs) for probs in matrix_probs]) for i in xrange(n)]

def log_odds_prob(matrix,motif,lamb=1):
    matrix_probs = [normalize([exp(-lamb*ep) for ep in row]) for row in matrix]
    log_p = sum(log(matrix_probs[i]["ACGT".index(b)]) for site in motif for i,b in enumerate(site))
    return log_p
    
def acceptance_ratio_study(matrix=None,sigma=1,iterations=10000,start=0,stop=10,steps=100):
    lambs = np.linspace(start,stop,steps)
    if matrix is None:
        matrix = sample_matrix(L,sigma)
    ars = []
    for i,lamb in enumerate(lambs):
        print i,lamb
        matrix,ar = sella_hirsch_imh(matrix=matrix,sigma=sigma,iterations=iterations,lamb=lamb,return_ar=True)
        ars.append(ar)
    def conf_int(p):
        z = 1.96
        return z*sqrt(1.0/iterations*p*(1-p))
    plt.errorbar(lambs,ars,yerr=map(conf_int,ars))
    return lambs,ars
        
def mutate_motif_k_times_ref(motif,k):
    motif_ = motif[:]
    n = len(motif)
    L = len(motif[0])
    N = n * L
    k_so_far = 0
    choices = []
    while k_so_far < k:
        i = random.randrange(n)
        j = random.randrange(L)
        if (i,j) in choices:
            continue
        else:
            choices.append((i,j))
            k_so_far += 1
            b = motif[i][j]
            new_b = random.choice([c for c in "ACGT" if not c == b])
            motif_[i] = subst(motif_[i],new_b,j)
    return motif_

def mutate_motif_k_times(motif,k):
    motif_ = motif[:]
    n = len(motif)
    L = len(motif[0])
    N = n * L
    rs = range(N)
    choices = []
    for _ in range(k):
        r = random.choice(rs)
        choices.append(r)
        rs.remove(r)
    for r in choices:
        i = r / L
        j = r % L
        b = motif[i][j]
        new_b = random.choice([c for c in "ACGT" if not c == b])
        motif_[i] = subst(motif_[i],new_b,j)
    return motif_
    
def discrete_exponential_sampler(N,lamb):
    ps = normalize([exp(-lamb*i) for i in range(N)])
    return inverse_cdf_sample(range(N),ps)

def sella_hirsch_gibbs_sampler(Ne=1000,n=16,L=16,G=5*10**6,sigma=1,init="random",
                                             matrix=None,x0=None,iterations=50000):
    if matrix is None:
        matrix = sample_matrix(L,sigma)
    if x0 is None:
        if init == "random":
            x0 = random_motif(L,n)
        elif init == "ringer":
            x0 = ringer_motif(matrix,n)
        else:
            x0 = init
    nu = Ne - 1
    def log_f(motif):
        return nu * log_fitness(matrix,motif,G)
    def prop(motif):
        #return mutate_motif_p(motif,1) # on average, 1 mutation per motif, (binomially distributed)
        return mutate_motif_p(motif,4) # on average, 4 mutation per motif, (binomially distributed)
    motif = x0
    chain = [motif]
    for iteration in xrange(iterations):
        for i in range(n):
            for j in range(L):
                prop_motifs = [subst_motif(motif,i,j,b) for b in "ACGT"]
                log_fs = map(log_f,prop_motifs)
                log_f_hat = mean(log_fs)
                log_fs_resid = [lf - log_f_hat for lf in log_fs]
                ps = normalize(map(exp,log_fs_resid))
                idx = inverse_cdf_sample(range(4),ps)
                motif = prop_motifs[idx]
                chain.append(motif)
        if iteration % 10 == 0:
            print "iterations:(%s/%s)" % (iteration,iterations),"log_f:",log_fs[idx]
    return matrix,chain

def sella_hirsch_converged_chain(Ne=1000,n=16,L=16,G=5*10**6,sigma=1,
                                 matrix=None,iterations=50000,return_iterations=False):
    if matrix is None:
        matrix = sample_matrix(L,sigma)
    nu = Ne - 1
    def log_f(motif):
        return nu * log_fitness(matrix,motif,G)
    x0_random = random_motif(L,n)
    x0_ringer = ringer_motif(matrix,n)
    converged = False
    total_iterations = 0
    while not converged:
        random_chain = mh(log_f,mutate_motif,x0_random,use_log=True,iterations=iterations)
        ringer_chain = mh(log_f,mutate_motif,x0_ringer,use_log=True,iterations=iterations)
        total_iterations += iterations
        x0_random = random_chain[-1]
        x0_ringer = ringer_chain[-1]
        random_fitness,ringer_fitness = log_f(x0_random), log_f(x0_ringer)
        print "random,ringer:",random_fitness,ringer_fitness
        if random_fitness >= ringer_fitness:
            converged = True
    print "total iterations:",total_iterations
    if not return_iterations:
        return matrix,random.choice([x0_random,x0_ringer])
    else:
        return total_iterations

def sella_hirsch_converged_chain_runtime_experiment():
    sigmas = np.linspace(0,3,10)
    trials = 5
    results = []
    for sigma in sigmas:
        print "sigma:",sigma
        results.append([sella_hirsch_converged_chain(Ne=5,sigma=sigma,return_iterations=True,iterations=1000)
                        for i in range(trials)])
    return results

def ringer_vs_random(Ne=5,sigma=1,matrix=None,p=None,iterations=50000,function=sella_hirsch_imh):
    L = 16
    n = 16
    if matrix is None:
        matrix = sample_matrix(L,sigma)
    ringer = ringer_motif(matrix,n)
    rand = random_motif(L,n)
    half = anti_ringer_motif(matrix,n)
    # _, ringer_chain = sella_hirsch_mh(Ne=Ne,sigma=sigma,matrix=matrix,init=ringer,p=p,
    #                                                            iterations=iterations)
    # _, random_chain = sella_hirsch_mh(Ne=Ne,sigma=sigma,matrix=matrix,init=random,p=p,
    #                                                            iterations=iterations)
    # _, half_chain = sella_hirsch_mh(Ne=Ne,sigma=sigma,matrix=matrix,init=half,p=p,
    #                                                          iterations=iterations)
    _, ringer_chain = function(Ne=Ne,sigma=sigma,matrix=matrix,init=ringer,p=p,
                                                               iterations=iterations)
    _, rand_chain = function(Ne=Ne,sigma=sigma,matrix=matrix,init=rand,p=p,
                                                               iterations=iterations)
    _, half_chain = function(Ne=Ne,sigma=sigma,matrix=matrix,init=half,p=p,
                                                             iterations=iterations)
    plt.subplot(2,1,1)
    print "mapping fitnesses"
    plot_matrix_chain_log_fitness((matrix,ringer_chain))
    plot_matrix_chain_log_fitness((matrix,rand_chain))
    plot_matrix_chain_log_fitness((matrix,half_chain))
    plt.subplot(2,1,2)
    print "mapping IC"
    plot_matrix_chain_ic((matrix,ringer_chain))
    plot_matrix_chain_ic((matrix,rand_chain))
    plot_matrix_chain_ic((matrix,half_chain))
    
def mh_vs_imh(Ne=5,sigma=1,matrix=None,p=None,iterations=50000,lamb=1):
    L = 16
    n = 16
    if matrix is None:
        matrix = sample_matrix(L,sigma)
    
    _, imh_chain = sella_hirsch_imh(Ne=Ne,sigma=sigma,matrix=matrix,init="random",p=p,
                                                               iterations=iterations,lamb=lamb)
    _, mh_anti_ringer_chain = sella_hirsch_mh(Ne=Ne,sigma=sigma,matrix=matrix,init="anti_ringer",p=p,
                                                               iterations=iterations)
    _, mh_ringer_chain = sella_hirsch_mh(Ne=Ne,sigma=sigma,matrix=matrix,init="ringer",p=p,
                                                             iterations=iterations)
    plt.subplot(1,2,1)
    plot_matrix_chain_log_fitness((matrix,mh_anti_ringer_chain))
    plot_matrix_chain_log_fitness((matrix,mh_ringer_chain))
    plot_matrix_chain_log_fitness((matrix,imh_chain))
    plt.subplot(1,2,2)
    plot_matrix_chain_ringer_distance((matrix,mh_anti_ringer_chain))
    plot_matrix_chain_ringer_distance((matrix,mh_ringer_chain))
    plot_matrix_chain_ringer_distance((matrix,imh_chain))
    
def sella_hirsch_reduction(Ne=1000,n=16,L=16,G=5*10**6,sigma=1,matrix=None,x0=None,iterations=50000):
    """if fg << Zb, as appears to be so, log(fitness) reduces to fg.  Test
this assumption by comparing to sella_hirsch_chain..."""
    if matrix is None:
        matrix = sample_matrix(L,sigma)
    if x0 is None:
        x0 = random_motif(L,n)
    nu = Ne - 1
    def log_f(motif):
        eps = [score_seq(matrix,site) for site in motif]
        fg = sum(exp(-ep) for ep in eps)
        return nu * log(fg)
    chain = mh(log_f,mutate_motif,x0,use_log=True,iterations=iterations)
    return matrix,chain
### Results: NOT THE CASE THAT fg << Zb IN GENERAL!  Assumption rested on failure to normalize Zb_hat by 4**L.

def sella_hirsch_reduction2(Ne=1000,n=16,L=16,G=5*10**6,sigma=1,matrix=None,x0=None,iterations=50000):
    """if fg >> Zb for genotypes likely at steady state, log(fitness) approx proportional to 1/fg.  Test
    this assumption by comparing to sella_hirsch_chain..."""
    if matrix is None:
        matrix = sample_matrix(L,sigma)
    if x0 is None:
        x0 = random_motif(L,n)
    nu = Ne - 1
    def log_f(motif):
        eps = [score_seq(matrix,site) for site in motif]
        fg = sum(exp(-ep) for ep in eps)
        return nu * 1/fg
    chain = mh(log_f,mutate_motif,x0,use_log=True,iterations=iterations)
    return matrix,chain
### Results: Doesn't work when far from stationarity, hence worthless

    
def gini_of_LGEs_experiment(iterations = 50000,Ne=200,sigma=1,lge_replicates=20):
    """Do motifs evolved under LGEs show more or less gini coefficient
    than IC-matched maxent counterparts?"""
    n = 16
    L = 16
    maxent_replicates= 1000
    lge_motifs = []
    lge_matrices = []
    maxent_motifss = []
    for i in range(lge_replicates):
        matrix,chain = sella_hirsch_mh(Ne=Ne,n=n,L=L,G=G,sigma=sigma,
                                                                iterations=iterations,init='ringer')
        lge_motif = chain[-1]
        desired_ic = motif_ic(lge_motif)
        maxent_motifs = maxent_sample_motifs_with_ic(n,L,desired_ic,replicates = maxent_replicates)
        lge_matrices.append(matrix)
        lge_motifs.append(lge_motif)
        maxent_motifss.append(maxent_motifs)
    return lge_matrices,lge_motifs,maxent_motifss
        
def plot_gini_of_LGEs_experiment(lge_matrices,lge_motifs,maxent_motifss):
    plt.subplot(1,4,1)
    print "plotting fitness"
    plt.boxplot([map(lambda m:log_fitness(matrix,m,G),maxent_motifs)
                 for matrix,maxent_motifs in zip(lge_matrices,maxent_motifss)])
    plt.scatter(range(1,len(lge_motifs) + 1),[log_fitness(matrix,lge_motif,G)
                                              for matrix,lge_motif in zip(lge_matrices,lge_motifs)])
    plt.subplot(1,4,2)
    print "plotting ic"
    plt.boxplot([map(motif_ic,maxent_motifs) for maxent_motifs in tqdm(maxent_motifss)])
    plt.scatter(range(1,len(lge_motifs) + 1),map(motif_ic,tqdm(lge_motifs)))
    plt.subplot(1,4,3)
    print "plotting gini"
    plt.boxplot([map(motif_gini,maxent_motifs) for maxent_motifs in tqdm(maxent_motifss)])
    plt.scatter(range(1,len(lge_motifs) + 1),map(motif_gini,tqdm(lge_motifs)))
    plt.subplot(1,4,4)
    print "plotting mi"
    plt.boxplot([map(total_motif_mi,maxent_motifs) for maxent_motifs in tqdm(maxent_motifss)])
    plt.scatter(range(1,len(lge_motifs) + 1),map(total_motif_mi,tqdm(lge_motifs)))

def plot_matrix_chain_log_fitness(matrix_chain,label=None):
    matrix,chain = matrix_chain
    plt.plot([log_fitness(matrix,motif,G) for motif in tqdm(chain)],label=label)

def log_fitnesses_from_matrix_chain(matrix_chain):
    matrix,chain = matrix_chain
    return [log_fitness(matrix,motif,G) for motif in tqdm(chain)]
    
def plot_matrix_chain_additive_fitness(matrix_chain,label=None):
    matrix,chain = matrix_chain
    plt.plot([fitness_additive(matrix,motif,G) for motif in tqdm(chain)],label=label)

def plot_matrix_chain_gini(matrix_chain):
    matrix,chain = matrix_chain
    plt.plot([motif_gini(motif) for motif in tqdm(chain)])

def plot_matrix_chain_ic(matrix_chain):
    matrix,chain = matrix_chain
    plt.plot([motif_ic(motif) for motif in tqdm(chain)])

def plot_matrix_chain_ringer_distance((matrix,chain)):
    n = len(chain[0])
    ringer = ringer_motif(matrix,n)
    plt.plot([motif_hamming_distance(m,ringer) for m in tqdm(chain)])

def plot_matrix_chain_distance_from_first((matrix,chain)):
    init = chain[0]
    plt.plot([motif_hamming_distance(m,init) for m in tqdm(chain)])
    
def plot_matrix_chain_ringer_distance((matrix,chain)):
    n = len(chain[0])
    ringer = ringer_motif(matrix,n)
    plt.plot([motif_hamming_distance(motif,ringer) for motif in tqdm(chain)])

def plot_matrix_chain_occupation_gini((matrix,chain)):
    plt.plot([occupation_gini(matrix,motif,G) for motif in tqdm(chain)])

def occupation_gini(matrix,motif,G):
    """return gini of occupancy over sites"""
    eps = [score_seq(matrix,site) for site in motif]
    fgs = [exp(-ep) for ep in eps]
    Zb = Zb_from_matrix(matrix,G)
    Z = sum(fgs) + Zb
    return gini([fg/Z for fg in fgs])
    
def Ne_sigma_scan(replicates=3,iterations=50000):
    """How does the rate of convergence of SH simulations depend on Ne and
    sigma?  A parameter scan to find out."""
    Nes = [10,100,1000]
    sigmas = [1,2,5,10]
    results = []
    for Ne in Nes:
        for sigma in sigmas:
            print "Ne,sigma:",Ne,sigma
            for replicate in range(replicates):
                matrix,chain = sella_hirsch_mh(Ne=Ne,sigma=sigma,init='ringer',
                                                                        iterations=iterations)
                results.append((Ne,sigma,matrix,chain))
    return results

def Ne_sigma_scan(iterations=50000):
    lge_replicates = 3
    lge_matrices = []
    lge_motifs = []
    maxent_motifs = []
    Nes = [10,100,1000]
    sigmas = [1,2,5,10]
    for (Ne,sigma) in itertools.product(Nes,sigmas):
        print Ne,sigma
        lge_mats, lge_mots,maxent_mots = gini_of_LGEs_experiment(Ne=Ne,sigma=sigma,
                                                                 iterations=iterations,lge_replicates=lge_replicates)
        lge_matrices.extend(lge_mats)
        lge_motifs.extend(lge_mots)
        maxent_motifs.extend(maxent_mots)
    plot_gini_of_LGEs_experiment(lge_matrices,lge_motifs,maxent_motifs)
    return lge_matrices,lge_motifs,maxent_motifs

def entropy_drift_analysis(sigma=2, color='b', color_p='g'):
    """why is convergence so difficult to obtain for, say, sigma = 2?  Explore selection/mutation balance."""
    n = 16
    L = 16
    matrix = sample_matrix(L,sigma)
    ringer = ringer_motif(matrix,n)
    mutants = [iterate(mutate_motif,ringer,i) for i in trange(256) for j in range(10)]
    dists = [motif_hamming_distance(ringer,mutant) for mutant in tqdm(mutants)]
    fs = [log_fitness(matrix,mutant,G) for mutant in tqdm(mutants)]
    fps = []
    trials = 100
    for mutant in tqdm(mutants):
        nexts = []
        f = log_fitness(matrix,mutant,G)
        for i in range(trials):
            mutant_p = mutate_motif(mutant)
            fp = log_fitness(matrix,mutant_p,G)
            if log(random.random()) < fp - f:
                nexts.append(fp)
            else:
                nexts.append(f)
        fps.append(mean(nexts))
    plt.subplot(3,1,1)
    plt.scatter(dists,fs,color=color,marker='.')
    plt.scatter(dists,fps,color=color_p,marker='.')
    #plt.semilogy()
    plt.subplot(3,1,2)
    plt.scatter(dists,[(f-fp)/f for (f,fp) in zip(fs,fps)],color=color,marker='.')
    plt.plot([0,len(fs)],[0,0],linestyle='--',color='black')
    plt.subplot(3,1,3)
    diffs = [fp - f for f,fp in zip(fs,fps)]
    plt.scatter(fs,diffs,marker='.',color=color)
    interpolant = poly1d(polyfit(fs,diffs,1))
    plt.plot(*pl(interpolant,[min(fs),max(fs)]))
    plt.plot([min(fs),max(fs)],[0,0],linestyle='--',color='black')
    minx,maxx = min(fs + fs),max(fs+fps)
    #plt.plot([minx,maxx],[minx,maxx],linestyle='--',color='black')
    #plt.show()

def convergence_wall_exploration(sigma=3,mutations=range(0,100,10),iterations=10000,Ne=5,p=None,
                                 function=sella_hirsch_imh):
    n = 16
    L = 16
    if p is None:
        p = 1.0/(n*L)
    matrix = sample_matrix(L,sigma)
    ringer = ringer_motif(matrix,n)
    mutants = [iterate(mutate_motif,ringer,i) for i in mutations]
    matrix_chains = [function(Ne=Ne,matrix=matrix,init=mutant,iterations=iterations,
                                                              p=p)
              for mutant in tqdm(mutants)]
    for matrix_chain in matrix_chains:
        plot_matrix_chain_log_fitness(matrix_chain)
    stats = [map(lambda m:log_fitness(matrix,m,G),chain) for (matrix,chain) in tqdm(matrix_chains)]
    return stats

def sella_hirsch_mh_gr(matrix,Ne=5,iterations=1000,n=16,x0s=None):
    """MH with gelman rubin criterion"""
    print "starting with iterations:",iterations
    L = len(matrix)
    if x0s is None:
        ringer = ringer_motif(matrix,n)
        anti_ringer = anti_ringer_motif(matrix,n)
        x0s = ([ringer] + [mutate_motif_k_times(ringer,k) for k in [1,2,4,8,16]] +
               [random_motif(L,n)] + [anti_ringer])
    matrix_chains = [sella_hirsch_mh(Ne=5,matrix=matrix,iterations=iterations,x0=x0) for x0 in x0s]
    fits = map(log_fitnesses_from_matrix_chain,matrix_chains)
    gr_criterion, neff = gelman_rubin(fits)
    if gr_criterion < 1.1:
        print "success:",gr_criterion,neff
        return matrix_chains
    else:
        print "not converged yet:",gr_criterion,neff,"at %s iterations" % iterations
        x0s_new = [matrix_chain[1][-1] for matrix_chain in matrix_chains]
        return sella_hirsch_mh_gr(matrix=matrix,Ne=Ne,iterations=2*iterations,n=n,x0s=x0s_new)

def overdispersion_plot_ref(matrix,Ne=5,iterations=50000,x0s=None):
    if x0s is None:
        ringer = ringer_motif(matrix,n)
        anti_ringer = anti_ringer_motif(matrix,n)
        x0s = ([ringer] + [mutate_motif_k_times(ringer,k) for k in [1,2,4,8,16,32,64]] +
               [random_motif(L,n)] + [anti_ringer])
    matrix_chains = [sella_hirsch_mh(Ne=5,matrix=matrix,iterations=iterations,x0=x0) for x0 in x0s]
    return matrix_chains

def pick_next_best(matrix,motif):
    f = lambda motif:fitness(matrix,motif,G)
    n,L = len(matrix),len(matrix[0])
    f0 = f(motif)
    print "f0:",f0
    if f0 == 0:
        print "motif has minimum fitness"
        return motif
    cur_motif = None
    fcur = None
    for i in range(n):
        for j in range(L):
            for bp in "ACGT":
                prop_motif = subst_motif(motif,i,j,bp)
                fprop = f(prop_motif)
                if fcur < fprop < f0:
                    cur_motif = prop_motif
                    fcur = fprop
    if cur_motif is None:
        return motif
    else:
        return cur_motif

def overdispersion_chain(matrix,n,iterations=None):
    if iterations is None:
        L = len(matrix)
        iterations = n*L
    ringer = ringer_motif(matrix,n)
    return [mutate_motif_k_times(ringer,k) for k in xrange(iterations)]

def overdispersion_chain2(matrix,n,iterations=None):
    if iterations is None:
        L = len(matrix)
        iterations = n*L
    ringer = ringer_motif(matrix,n)
    f = lambda motif:pick_next_best(matrix,motif)
    return iterate_list(f,ringer,iterations)

def overdispersion_plot(matrix,Ne=5,n=16,iterations=10000):
    L = len(matrix)
    N = L*n
    x0s = overdispersion_chain(matrix,n,iterations=N)
    matrix_chains = [sella_hirsch_mh(Ne=5,matrix=matrix,iterations=iterations,x0=x0) for x0 in x0s]
    map(plot_matrix_chain_log_fitness,matrix_chains)
    return matrix_chains

def overdispersion_plot2(matrix,Ne=5,n=16,iterations=10000):
    L = len(matrix)
    N = L*n
    x0s = overdispersion_chain2(matrix,n,iterations=N)
    matrix_chains = [sella_hirsch_mh(Ne=5,matrix=matrix,iterations=iterations,x0=x0) for x0 in x0s]
    map(plot_matrix_chain_log_fitness,matrix_chains)
    return matrix_chains
    
def gelman_rubin(chains):
    N = len(chains[0])
    burned_chains = [chain[N/2:] for chain in chains] # eliminate burn-in
    # now split each one in half
    halved_chains = concat([(chain[:len(chain)/2],chain[len(chain)/2:]) for chain in burned_chains])
    min_len = min(map(len,halved_chains))
    halved_chains = [hc[:min_len] for hc in halved_chains]
    m = len(halved_chains)
    n = len(halved_chains[0])
    psi = np.matrix(halved_chains).transpose()
    psi_bar = np.mean(psi)
    B = n/float(m-1)*sum((np.mean(psi[:,j]) - psi_bar)**2 for j in range(m))
    def sj_sq(j):
        psi_j = np.mean(psi[:,j])
        return 1.0/(n-1)*sum((psi[i,j] - psi_j)**2 for i in range(n))
    W = 1.0/m * sum(sj_sq(j)for j in range(m))
    var_hat_plus = (n-1)/float(n)*W + 1.0/n * B
    R_hat = sqrt(var_hat_plus/W)
    def V(t):
        return 1.0/(m*(n-t))*sum((psi[i,j] - psi[i-t,j])**2 for i in range(t+1,n) for j in range(m))
    def rho_hat(t):
        return 1 - V(t)/(2*var_hat_plus)
    crit = lambda t:(t % 2 == 1) and rho_hat(t+1) + rho_hat(t+2) < 0
    #T = find(lambda t:(t % 2 == 1) and rho_hat(t+1) + rho_hat(t+2) < 0,range(n-1))
    T = binary_find(crit,range(n))
    if not T is None:
        neff = m*n/(1 + 2*sum(rho_hat(t) for t in range(1,T+1)))
    else:
        neff = None
    return R_hat,neff
    
def approximate_mean_log_f(matrix,Ne,mutations=n*L):
    nu = Ne - 1
    ringer = ringer_motif(matrix,n)
    replicates = 1000
    Z = 0
    acc = 0
    mean_log_fs = []
    running_totals = []
    log_contribs = []
    log_ws = []
    mean_log_bzs = []
    log_weights = []
    def w(k):
        return 4**k
    for k in range(mutations):
        motifs = [mutate_motif_k_times(ringer,k) for i in xrange(replicates)]
        log_fs = [log_fitness(matrix,motif,G) for motif in motifs]
        bzs = [exp(nu*log_f) for log_f in log_fs]
        contrib = sum(log_f * bz for log_f,bz in zip(log_fs,bzs))
        acc += w(k)*contrib
        Z += w(k)*sum(bzs)
        print k,mean(log_fs),acc,Z,acc/Z
        mean_log_fs.append(mean(log_fs))
        running_totals.append(acc/Z)
        log_ws.append(log(w(k)))
        log_weights.append(log(w(k)*mean(bzs)))
        log_contribs.append(log(mean(log_fs)*w(k)*mean(bzs)))
    plt.plot(mean_log_fs,label="mean log f")
    # plt.plot(mean_log_bzs,label="mean log boltzmann weight")
    # plt.plot(log_ws,label="mean weight")
    # plt.plot(running_totals,label='running total')
    plt.plot(log_weights,label='log weights')
    plt.plot(log_contribs,label='log contribs')
    plt.legend()
    
def sella_hirsch_imh_sanity(Ne=5,n=16,L=16,G=5*10**6,sigma=1,init="random",
                                             matrix=None,x0=None,iterations=50000,p=None,lamb=1):
    """Independent Metropolis Hastings with proposal density geometrically
    distributed in # mutations from ringer"""
    if matrix is None:
        matrix = sample_matrix(L,sigma)
    x = random_motif(L,n)
    nu = Ne - 1
    N = n * L
    ringer = ringer_motif(matrix,n)
    ps = normalize([exp(-lamb*i) for i in range(N)])
    def log_f(motif):
        return nu * log_fitness(matrix,motif,G)
    def prop(motif):
        k = inverse_cdf_sample(range(N),ps)
        motif_p = mutate_motif_k_times(ringer,k)
        return motif_p
    def log_dprop(motif,_):
        k = motif_hamming_distance(ringer,motif) # number of mutations from ringer
        return log(ps[k]) - log_choose(N,k) + k * log(1/3.0)
    chain = []
    log_fx = log_f(x)
    log_dpropx = log_dprop(x,None)
    acceptances = 0
    for i in trange(iterations):
        xp = prop(None)
        log_fxp = log_f(xp)
        log_dpropxp = log_dprop(xp,None)
        log_r = log(random.random())
        ar = log_fxp - log_fx + log_dpropx - log_dpropxp
        accept = log_r < ar
        #print log_fx,log_fxp,log_dpropx,log_dpropxp,ar,accept
        if accept:
            acceptances +=1 
            x = xp
            log_fx = log_fxp
            log_dpropx = log_dpropxp
        chain.append(x)
    #chain = mh(log_f,prop,x0,dprop=log_dprop,use_log=True,iterations=iterations,verbose=False)
    print "Acceptance Ratio:",acceptances/float(iterations)
    return matrix,chain

def importance_sample_mean_fitness(Ne=5,n=16,L=16,G=5*10**6,sigma=1,
                                             matrix=None,x0=None,iterations=50000,lamb=1):
    """Independent Metropolis Hastings with proposal density geometrically
    distributed in # mutations from ringer"""
    if matrix is None:
        matrix = sample_matrix(L,sigma)
    nu = Ne - 1
    N = n * L
    ringer = ringer_motif(matrix,n)
    #ps = normalize([exp(-lamb*i) for i in range(N)])
    ps = [1.0/N]*N
    def log_f(motif):
        return log_fitness(matrix,motif,G)
    def prop(motif):
        k = inverse_cdf_sample(range(N),ps)
        motif_p = mutate_motif_k_times(ringer,k)
        return motif_p
    def log_dprop(motif,_):
        k = motif_hamming_distance(ringer,motif) # number of mutations from ringer
        return log(ps[k]) - log_choose(N,k) + k * log(1/3.0)
    def weight(motif):
        k = motif_hamming_distance(ringer,motif) # number of mutations from ringer
        return 4**k
    motifs = [prop(None) for i in trange(iterations)]
    log_fs = map(log_f,tqdm(motifs))
    log_dprops = [log_dprop(motif,None) for motif in tqdm(motifs)]
    ws = [exp(nu*lf - ldp) for (lf,ldp) in zip(log_fs,log_dprops)]
    Z = mean(ws)
    #print "Z:",Z,bs_ci(mean,ws)
    print "min w:",min(ws)
    ps = normalize(ws)
    mean_log_f = sum([lf*p for lf,p in zip(log_fs,ps)])
    return mean_log_f

def plot_log_fs_vs_log_dprops(Ne=5,n=16,L=16,G=5*10**6,sigma=1,
                                             matrix=None,x0=None,iterations=50000,lamb=1):
    if matrix is None:
        matrix = sample_matrix(L,sigma)
    nu = Ne - 1
    N = n * L
    ringer = ringer_motif(matrix,n)
    ps = normalize([exp(-lamb*i) for i in range(N)])
    #ps = [1.0/N]*N
    def log_f(motif):
        return log_fitness(matrix,motif,G)
    def prop(motif):
        k = inverse_cdf_sample(range(N),ps)
        motif_p = mutate_motif_k_times(ringer,k)
        return motif_p
    def log_dprop(motif,_):
        k = motif_hamming_distance(ringer,motif) # number of mutations from ringer
        ### return log(ps[k]) + k * log(1/3.0)  ### XXX BE LESS RETARDED ABOUT THIS XXX
        # p(k)*(1/choose(N,k)) * (1/3.0)**k
        return log(ps[k]) - log_choose(N,k) + k*log(1.0/3)
    def weight(motif):
        k = motif_hamming_distance(ringer,motif) # number of mutations from ringer
        return 4**k
    matrix_probs = [normalize([exp(-ep) for ep in row]) for row in matrix]
    def prop_fanciful(motif):
        return ["".join([inverse_cdf_sample("ACGT",probs) for probs in matrix_probs]) for i in xrange(n)]
    def dprop(motif,_):
        k = motif_hamming_distance(ringer,motif) # number of mutations from ringer
        #return choose_reference(N,k) * p**k * (1-p)**(N-k) * (1/3.0)**k
        #return 1.0/N * (1/3.0)**k
        return ps[k] * choose(N,k) * (1/3.0)**k
    def log_dprop(motif,_):
        k = motif_hamming_distance(ringer,motif) # number of mutations from ringer
        #return log_choose(N,k) + k * log(p) + (N-k)*log(1-p) + k*log(1/3.0)
        #return -log(N) + k * log(1/3.0)
        return log(ps[k]) - log_choose(N,k) + k * log(1/3.0)
    def log_dprop_fanciful(motif,_):
        return sum(log(matrix_probs[i]["ACGT".index(b)]) for site in motif for i,b in enumerate(site))
    motifs = [prop_fanciful(None) for i in trange(iterations)]
    log_fs = np.array(map(log_f,tqdm(motifs)))
    log_dprops = np.array([log_dprop_fanciful(motif,None) for motif in tqdm(motifs)])
    ws = np.exp(log_fs - log_dprops)
    ws = ws/np.sum(ws)
    rhos = np.array([motif_hamming_distance(motif,ringer) for motif in motifs])
    print "mean log_f:", ws.dot(log_fs)
    print "mean rho:", ws.dot(rhos)
    plt.subplot(2,3,1)
    plt.title("log fitness vs proposal")
    plt.scatter(log_fs,log_dprops)
    minval,maxval = min(log_fs + log_dprops),max(log_fs + log_dprops)
    plt.plot([minval,maxval],[minval,maxval],linestyle='--')
    plt.subplot(2,3,2)
    #plt.scatter(exp,log_fs,rhos)
    plt.title("ringer dstance vs weight")
    plt.scatter(rhos,np.log10(ws))
    plt.subplot(2,3,3)
    #plt.scatter(exp,log_fs,rhos)
    plt.title("log fitness vs wieght")
    plt.scatter(log_fs,np.log10(ws))
    plt.subplot(2,3,5)
    #plt.scatter(exp,log_fs,rhos)
    plt.scatter(rhos,log_fs)
    plt.subplot(2,3,6)
    #plt.scatter(exp,log_fs,rhos)
    plt.scatter(log_fs,np.log10(ws))
    
    
def uniform_sample_mean_fitness(Ne=5,n=16,L=16,G=5*10**6,sigma=1,
                                             matrix=None,x0=None,iterations=50000,lamb=1):
    if matrix is None:
        matrix = sample_matrix(L,sigma)
    nu = Ne - 1
    N = n * L
    ringer = ringer_motif(matrix,n)
    #ps = normalize([exp(-lamb*i) for i in range(N)])
    ps = [1.0/N]*N
    def log_f(motif):
        return log_fitness(matrix,motif,G)
    def prop(motif):
        return random_motif(L,n)
    def log_dprop(motif,_):
        # log(1/(4**(n*L)))
        return - n*L * log(4)
    motifs = [prop(None) for i in trange(iterations)]
    log_fs = map(log_f,tqdm(motifs))
    log_dprops = [log_dprop(motif,None) for motif in tqdm(motifs)]
    ws = [exp(nu*lf - ldp) for (lf,ldp) in zip(log_fs,log_dprops)]
    Z = mean(ws)
    print "Z:",Z,bs_ci(mean,ws)
    print "min w:",min(ws)
    ps = normalize(ws)
    mean_log_f = sum([lf*p for lf,p in zip(log_fs,ps)])
    return mean_log_f

    
def entropic_sampling(matrix,n=16):
    """sample motifs uniformly wrt fitness"""
    L = len(matrix)
    ringer = ringer_motif(matrix,n)
    N = n * L
    ps = [1.0/N for i in range(N)]
    ks = range(N)
    replicates = 10000
    motifs = [mutate_motif_k_times(ringer,inverse_cdf_sample(ks,ps)) for i in trange(replicates)]
    log_fs = [log_fitness(matrix,motif,G) for motif in motifs]

def main_experiment(motif_obj):
    """compare gini biological motifs to:
    (1) null ic-matched ensembles,and
    (2) sigma and IC matched evosims.
    Conduct evosims by matching sigma to pssm (fair?) and sweeping Ne in order to match IC.
    """
    evosim_trials = 10
    for tf in motif_obj.tfs:
        bio_motif = getattr(motif_obj,tf)
        n,L = len(bio_motif),len(bio_motif[0])
        bio_gini = motif_gini(bio_motif)
        bio_ic = motif_ic(bio_motif)
        bio_mi = total_motif_mi(bio_motif)
        ###############
        ### null ensemble stuff here
        ###############
        pssm = make_pssm(bio_motif)
        sigma = mean(map(sd,pssm)) # revisit this, see Djordjevic's paper
        # determine Ne
        Ne_ic = {}
        lo = 0
        hi = 5
        #Ne_ic[lo] = 
        chain = sella_hirsch_mh_gr(matrix,Ne=5,iterations=1000,n=16,x0s=None)
        print "sigma:",sigma
        for trial in trange(evosim_trials):
            matrix = sample_matrix(L,sigma)

def estimate_stationary_statistic(matrix,n,Ne,T,samples_per_bin=10):
    """given matrix, Ne and statistic T, estimate <T> under stationary
    distribution by importance sampling perturbations from ringer"""
    L = len(matrix)
    N = n*L
    nu = Ne - 1
    ringer = ringer_motif(matrix,n)
    all_sampless = [[mutate_motif_k_times(ringer,k) for i in range(samples_per_bin)] for k in trange(N)]
    Tss = mmap(T,all_sampless)
    fss = mmap(lambda motif:fitness(matrix,motif,G),all_sampless)
     # better expressed as exp(nu*log(f)), but numeric issues
    bz_weightss = [[(f**nu) for f in fs] for rho,fs in enumerate(fss)]
    Z = sum([mean(bz_weights)*4**rho for rho,bz_weights in enumerate(bz_weightss)])
    summands = [4**rho*mean(t*bz_weight/Z for t,bz_weight in zip(ts,bz_weights))
                for rho,(ts,bz_weights) in enumerate(zip(Tss,bz_weightss))]
    return sum(summands)

def estimate_stationary_statistic2(matrix,n,Ne,T,num_samples=10000,lamb=1):
    L = len(matrix)
    N = n*L
    nu = Ne - 1
    ringer = ringer_motif(matrix,n)
    all_samples = [sample_log_odds(matrix,n,lamb) for _ in trange(num_samples)]
    Ts = map(T,all_samples)
    fs = map(lambda motif:fitness(matrix,motif,G),all_samples)
     # better expressed as exp(nu*log(f)), but numeric issues
    p_hats = [(f**nu) for f in tqdm(fs)]
    Z = sum(p_hats)
    ps = [p_hat/Z for p_hat in p_hats]
    qs = normalize([exp(log_odds_prob(matrix,sample,lamb=lamb)) for sample in all_samples])
    return sum(t*(p/q) for t,p,q in zip(Ts,ps,qs))

def test_estimate_stationary_stat(trials=10):
    Ls = [5,10,15]
    ns = [10,20,50]
    Nes = [1,2,3,4,5]
    sigmas = [1,2,3,4,5]
    predicted_stats = []
    for trial in trange(trials):
        L = random.choice(Ls)
        n = random.choice(ns)
        Ne = random.choice(Nes)
        sigma = random.choice(sigmas)
        print "L:",L,"n:",n,"Ne:",Ne,"sigma:",sigma
        matrix = sample_matrix(L,sigma)
        Ts = [motif_ic,motif_gini,total_motif_mi,lambda motif:log10(fitness(matrix,motif,G))]
        matrix_chains = sella_hirsch_mh_gr(matrix,Ne=Ne,n=n,iterations=1000)
        observed_stats = [[T(matrix_chain[1][-1]) for matrix_chain in matrix_chains] for T in Ts]
        predicted_stats = [estimate_stationary_statistic(matrix,n=n,Ne=Ne,T=T,samples_per_bin=10) for T in Ts]
    return (L,n,Ne,sigma,matrix),predicted_stats,observed_stats

log10 = lambda x:log(x+10**-300,10)

def rho_fitness_plot(matrix_chain):
    matrix,chain = matrix_chain
    n = len(chain[0])
    L = len(chain[0][0])
    N = n * L
    ringer = ringer_motif(matrix,n)
    print "fs"
    fs = [log10(fitness(matrix,motif,G)) for motif in tqdm(chain)]
    print "rhos"
    rhos = [motif_hamming_distance(ringer,motif) for motif in tqdm(chain)]
    print "ics"
    ics = [motif_ic(motif) for motif in tqdm(chain)]
    plt.xlabel("Distance from ringer")
    plt.ylabel("fitness")
    print "perturbations"
    perturbations = [mutate_motif_k_times(ringer,random.randrange(N)) for _ in tqdm(chain)]
    print "fs"
    perturb_fs = [log10(fitness(matrix,motif,G)) for motif in tqdm(perturbations)]
    print "rhos"
    perturb_rhos = [motif_hamming_distance(ringer,motif) for motif in tqdm(perturbations)]
    print "ics"
    perturb_ics = [motif_ic(motif) for motif in tqdm(perturbations)]
    print "log odds"
    log_odds = [sample_log_odds(matrix,n,3) for lamb in tqdm(np.linspace(0.5,5,10000))]
    print "fs"
    log_odds_fs = [log10(fitness(matrix,motif,G)) for motif in tqdm(log_odds)]
    print "rhos"
    log_odds_rhos = [motif_hamming_distance(ringer,motif) for motif in tqdm(log_odds)]
    print "ics"
    log_odds_ics = [motif_ic(motif) for motif in tqdm(log_odds)]
    plt.subplot(1,3,1)
    plt.scatter(perturb_rhos,perturb_fs,color='g')
    plt.scatter(log_odds_rhos,log_odds_fs,color='r')
    plt.scatter(rhos,fs,color='b')
    plt.xlabel("rho")
    plt.ylabel("log f")
    plt.subplot(1,3,2)
    plt.scatter(perturb_rhos,perturb_ics,color='g')
    plt.scatter(log_odds_rhos,log_odds_ics,color='r')
    plt.scatter(rhos,ics,color='b')
    plt.xlabel("rho")
    plt.ylabel("IC")
    plt.subplot(1,3,3)
    plt.scatter(perturb_ics,perturb_fs,color='g')
    plt.scatter(log_odds_ics,log_odds_fs,color='r')
    plt.scatter(ics,fs,color='b')
    plt.xlabel("IC")
    plt.ylabel("fs")

def running_estimate(matrix_chain,Ne=5,T=motif_ic):
    matrix, chain = matrix_chain
    Z = 0
    nu = Ne - 1
    bz = lambda motif:fitness(matrix,motif,G)**nu
    Z = 0
    accs = [0]
    foo = []
    for i,motif in enumerate(chain):
        tm = T(motif)
        w = bz(motif)
        Z += w
        acc = accs[-1]
        foo.append(tm*w)
        accs.append(acc*(i/(i+1.0)) + tm*w/Z * 1/(i+1.0))
    return accs,foo
