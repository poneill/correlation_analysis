from utils import random_site, inverse_cdf_sample, inverse_cdf_sampler
from utils import motif_ic, mean, mutate_site, mh, prod, mean_ci
from utils import score_seq, argmin, argmax, mmap, choose2
from utils import normalize, subst, bisect_interval_noisy_ref
from utils import rslice, sorted_indices, seq_scorer
from pwm_utils import ringer_motif, sigma_from_matrix, pssm_from_motif, approx_mu
from why_linear_recognition_sanity_check import sample_matrix
from math import exp, log
import random
from matplotlib import pyplot as plt
from tqdm import *
import numpy as np
import time
from scipy import polyfit
from scipy.stats import mannwhitneyu
from scipy.interpolate import griddata, interp2d

def sample_site_vb(matrix, mu, Ne):
    nu = Ne - 1
    alpha = nu*exp(-mu)
    log_new_mat = [([(-(1+alpha*(exp(ep)))) for ep in row])
               for row in matrix]
    return "".join(inverse_cdf_sample("ACGT",ps) for ps in new_mat)

def online_mh_approximation(matrix, mu, Ne, iterations=50000):
    """find best linear approximation using online MH"""
    nu = Ne - 1
    def phat(s):
        ep = score_seq(matrix,s)
        return (1 + exp(ep - mu))**(-nu)
    pss = [[1]*4 for _ in matrix]
    def rQ():
        return "".join([inverse_cdf_sample("ACGT",ps,normalized=False)
                        for ps in pss])
    def dQ(s):
        ds = prod([ps["ACGT".index(sj)]/float(sum(ps)) # not quite right but...
                   for ps, sj in zip(pss,s)])
        return ds
    x = rQ()
    for i in trange(iterations):
        xp = rQ()
        r = random.random()
        ar = phat(xp)/phat(x)*dQ(x)/dQ(xp)
        if r < ar:
            x = xp
        for j, b in enumerate(x):
            pss[j]["ACGT".index(b)] += 1
    return [normalize(ps) for ps in pss]

def sample_site_spec(matrix, mu, Ne):
    nu = Ne - 1
    L = len(matrix)
    best_site = "".join(["ACGT"[argmin(col)] for col in matrix])
    worst_site = "".join(["ACGT"[argmax(col)] for col in matrix])
    def phat(s):
        assert len(s) == L
        ep = score_seq(matrix,s)
        return (1 + exp(ep - mu))**(-nu)
    chosen_site = ""
    def best_completion(s):
        l = len(s)
        return phat(s + best_site[l:])
        
    def worst_completion(s):
        l = len(s)
        return s + worst_site[l:]
    return chosen_site

def langrangian_approximation(matrix, mu, Ne):
    nu = Ne - 1
    alpha = nu*exp(-mu)
    qss = [[1/4.0]*4 for row in matrix]
    def dLdqij(qss,i,j):
        term0 = alpha*exp(matrix[i][j])/sum(qij*exp(epij)
                                        for qij,epij in zip(qss[i],matrix[i]))
        expectation = prod([sum(qij*exp(epij) for qij,epij in zip(qsi,epsi))
                            for qsi,epsi in zip(qss,matrix)])
        return term0*expectation + log(qss[i][j]) + 1
def sample_site_omh_approx(matrix, mu, Ne, qss=None):
    if qss is None:
        qss = online_mh_approximation(matrix, mu, Ne)
    nu = Ne - 1
    def phat(s):
        ep = score_seq(matrix,s)
        return (1 + exp(ep - mu))**(-nu)
    def rQ():
        return "".join([inverse_cdf_sample("ACGT",qs) for qs in qss])
    def dQ(s):
        ds = prod([qs["ACGT".index(sj)] # not quite right but...
                   for qs, sj in zip(qss,s)])
        return ds
    
    
    
    
def sample_site_bf(matrix, mu, Ne, ringer_site=None, verbose=False):
    """Sample site of length L from stationary fitness distribution under
    E(s) at effective population Ne, chemical potential mu.  (bf for
    brute force)
    """
    nu = Ne - 1
    L = len(matrix)
    if ringer_site is None:
        ringer_site = ringer_motif(matrix,1)[0]
    def phat(s):
        ep = score_seq(matrix,s)
        return (1 + exp(ep - mu))**(-nu)
    phat_max = phat(ringer_site)
    trials = 0
    while True:
        trials += 1
        site = random_site(L)
        ar = phat(site)/phat_max
        if random.random() < ar:
            if verbose:
                print trials, ar
            return site

def sample_motif_bf(matrix, mu, Ne, n, ringer_site=None, verbose=False):
    ringer_site = ringer_motif(matrix,1)[0]
    return [sample_site_bf(matrix, mu, Ne, ringer_site=ringer_site,verbose=verbose)
            for i in xrange(n)]
    

def sample_site_study(trials=1000):
    sigmas = np.linspace(0,10,100)
    Nes = np.linspace(2,10,100)
    mus = np.linspace(-10,10,100)
    L = 10
    results = {}
    for trial in trange(trials):
        sigma = random.choice(sigmas)
        Ne = random.choice(Nes)
        mu = random.choice(mus)
        matrix = sample_matrix(L,sigma)
        ringer_site = ringer_motif(matrix,1)[0]
        t0 = time.time()
        sites = [sample_site_bf(matrix, mu, Ne, ringer_site) for i in range(10)]
        t = time.time() - t0
        results[(sigma, Ne, mu)] = t
    return results

def spoof_motif_ar(motif, num_motifs=10, trials=1, sigma=None,Ne_tol=10**-4):
    n = len(motif)
    L = len(motif[0])
    copies = 10*n
    if sigma is None:
        sigma = sigma_from_matrix(pssm_from_motif(motif,pc=1))
    print "sigma:", sigma
    bio_ic = motif_ic(motif)
    matrix = sample_matrix(L, sigma)
    mu = approx_mu(matrix, copies=10*n, G=5*10**6)
    print "mu:", mu
    def f(Ne):
        motifs = [sample_motif_ar(matrix, mu, Ne, n)
                  for i in trange(trials)]
        return mean(map(motif_ic,motifs)) - bio_ic
    x0 = 2
    print "Ne guess:", x0
    Ne = bisect_interval_noisy(f,x0=x0,iterations=100,lb=1, verbose=False,w=0.5)
    print "Ne:",Ne
    return [sample_motif_ar(matrix, mu, Ne, n) for _ in trange(num_motifs)]

def sample_from_matrix(matrix, lamb):
    return "".join([inverse_cdf_sample("ACGT", [exp(-lamb*ep) for ep in row],
                                normalized=False)
             for row in matrix])

def sample_site_ar_spec(matrix, mu, Ne, lamb):
    nu = Ne - 1
    def rQ():
        return sample_from_matrix(matrix, lamb)
    log_Z = sum(log(sum(exp(-lamb*ep) for ep in col)) for col in matrix)
    def log_dQ(site):
        log_numer = -lamb*sum(row["ACGT".index(b)] for b,row in zip(site,matrix))
        return  log_numer - log_Z
    def log_f(site):
        ep = score_seq(matrix, site)
        return -nu*log(1+exp(ep-mu))
    site_min = "".join("ACGT"[argmax(row)] for row in matrix) # minimizes Q
    log_M = log_f(site_min) - log_dQ(site_min)
    print "log_M:", log_M
    while True:
        site = rQ()
        log_ar = log_f(site) - (log_M + log_dQ(site))
        print "log_f:",log_f(site)
        print "log_dQ:",log_dQ(site)
        print "log ar:", log_ar
        assert log_ar < 0
        if log(random.random()) < log_ar:
            return site
    
    
def sample_site_ar(matrix, mu, Ne, lamb=None, modulus=10**6, return_ar=False):
    nu = Ne - 1
    if nu == 0:
        return random_site(len(matrix))
    if lamb is None:
        lamb = nu/2.0
    L = len(matrix)
    def rQ():
        return sample_from_matrix(matrix, lamb)
    #log_Z = sum(log(sum(exp(-lamb*ep) for ep in col)) for col in matrix)
    def log_dQ(site):
        log_numer = -lamb*sum(row["ACGT".index(b)] for b,row in zip(site,matrix))
        return  log_numer# - log_Z
    def log_fit(site):
        return -nu*log(1+exp(score_seq(matrix,site)-mu))
    ep_max = sum(max(row) for row in matrix)
    ep_min = sum(min(row) for row in matrix)
    alpha = lamb/float(nu)
    def find_logM(ep):
        return nu*log((exp(alpha*ep)/(1+exp(ep-mu))))
    def log_M_p(ep):
        term1 = alpha*exp(alpha*ep)/(exp(ep-mu)+1)
        term2 = -exp(alpha*ep+ep-mu)/((exp(ep-mu)+1)**2)
        return term1 + term2
    if alpha != 1 and alpha/(1-alpha) > 0:
        ep_crit = log(alpha/(1-alpha)) + mu
    else:
        deriv = log_M_p(0)
        if deriv < 0:
            ep_crit = ep_min
        else:
            ep_crit = ep_max
    log_M = find_logM(ep_crit)
    # print "choosing from:",find_logM(ep_min), find_logM(ep_max)
    # print "log_M:",log_M
    trials = 0
    while True:
        trials += 1
        s = rQ()
        log_f = log_fit(s)
        log_prop = log_dQ(s) + log_M
        log_ar = log_f - log_prop
        log_r = log(random.random())
        accept = log_r < log_ar
        #print trials, s, "*" if accept else " ",log_r, log_ar, log_f, log_prop
        assert log_f < 0
        assert log_ar < 0
        if trials % modulus == 0:
            print trials, s, "*" if accept else " ",log_r, log_ar, log_f, log_prop
        if accept:
            if return_ar:
                return trials
            else:
                return s

def sample_motif_ar(matrix, mu, Ne, n, modulus=10**6):
    return [sample_site_ar(matrix, mu, Ne, modulus=modulus) for _ in xrange(n)]
    
def sample_motif_ar_param_study():
    """Examine dependence of IC on sigma, Ne"""
    sigmas = np.linspace(0.5,3,5)
    Nes = np.linspace(1,3,5)
    trials = 3
    n = 20
    L = 10
    def f(sigma, Ne):
        matrix = sample_matrix(L, sigma)
        mu = approx_mu(matrix, 10*n)
        return motif_ic(sample_motif_ar(matrix, mu, Ne, n, modulus=10**5))
    ics = [[(mean(f(sigma, Ne) for _ in range(trials)))
            for sigma in sigmas] for Ne in tqdm(Nes)]
    plt.contourf(sigmas, Nes,ics)
    plt.colorbar()
    # bio_motifs = [getattr(Escherichia_coli,tf) for tf in Escherichia_coli.tfs]
    # bio_sigmas = [sigma_from_matrix(pssm_from_motif(motif,pc=1))
    #               for motif in bio_motifs]
    # bio_ics = [motif_ic(motif) for motif in bio_motifs]
    # griddata((sigmas,Nes),ics)
    # interp = interp2d(sigmas,Nes,ics)
    # bio_Nes = [bisect_interval(lambda Ne:interp(show(bio_sigma),Ne)-bio_ic,0,20)
    #            for bio_sigma, bio_ic in zip(bio_sigmas,bio_ics)]
    # plt.scatter(sigm)
    
    
def sample_site_mh(matrix, mu, Ne, ringer_site, iterations=1000):
    nu = Ne - 1
    def phat(s):
        ep = score_seq(matrix, s)
        return (1 + exp(ep - mu))**(-nu)
    return mh(f=phat,proposal=mutate_site,x0=ringer_site, iterations=iterations)

def sample_motif_with_ic(n,L):
    matrix = sample_matrix(L,sigma=1)
    ringer_site = "".join(["ACGT"[argmin(col)] for col in matrix])
    mu = approximate_mu(matrix,10*n,G=5*10**6)
    Nes = range(2,10)
    trials = 10
    motifs = [[sample_motif_mh(matrix, mu, Ne, n)
               for t in range(trials)] for Ne in tqdm(Nes)]

def sample_motif_mh(matrix, mu, Ne, n, iterations=None):
    L = len(matrix)
    iterations = 20*L
    ringer_site = "".join(["ACGT"[argmin(col)] for col in matrix])
    return [sample_site_mh(matrix, mu, Ne, ringer_site, iterations=iterations)[-1]
            for i in xrange(n)]


def test():
    L = 10
    matrix = [[-1,0,0,0] for i in range(L)]
    ringer_site = "A"*L
    n = 10
    trials = 10
    mus = range(-9,1,1)
    Nes = range(2,11,1)
    ics = [[mean(motif_ic(sample_motif(matrix, show(mu), Ne, ringer_site, n))
                 for i in range(trials)) for mu in tqdm(mus)]
           for Ne in Nes]
    plt.contour(mus,Nes,ics)
    plt.colorbar(label="IC")
    plt.xlabel("Mu")
    plt.ylabel("Nes")

def site_sampling_methods_study(n=50, num_motifs=10, plot=True):
    """validate that the three proposed sampling methods:

    brute force
    rejection sampling
    metropolis hastings

    do in fact sample from the same distribution
    """

    L = 10
    sigma = 1
    matrix = sample_matrix(L, sigma)
    Ne = 5
    mu = -10
    print "bf"
    t0 = time.time()
    bf_motifs = [sample_motif_bf(matrix, mu, Ne, n,verbose=True)
                 for i in trange(num_motifs)]
    bf_time = time.time() - t0
    print "ar"
    t0 = time.time()
    ar_motifs = [sample_motif_ar(matrix, mu, Ne, n)
                 for i in range(num_motifs)]
    ar_time = time.time() - t0
    print "mh"
    t0 = time.time()
    mh_motifs = [sample_motif_mh(matrix, mu, Ne, n)
                 for i in range(num_motifs)]
    mh_time = time.time() - t0
    icss = mmap(motif_ic,[bf_motifs, ar_motifs, mh_motifs])
    print "ics:", map(mean_ci, icss)
    print "time per motif:", [t/num_motifs
                              for t in [bf_time, ar_time, mh_time]]
    if plot:
        plt.boxplot(icss)
    for xs, ys in choose2(icss):
        print mannwhitneyu(xs,ys)

def sample_site_cftp_dep(matrix, mu, Ne):
    L = len(matrix)
    def log_phat(s):
        ep = score_seq(matrix,s)
        nu = Ne - 1
        return -nu*log(1 + exp(ep - mu))
    first_site = "A"*L
    last_site = "T"*L
    best_site = "".join(["ACGT"[argmin(row)] for row in matrix])
    worst_site = "".join(["ACGT"[argmax(row)] for row in matrix])
    trajs = [[best_site],[random_site(L)],[random_site(L)],[random_site(L)], [worst_site]]
    def mutate_site(site,(ri,rb)):
        return subst(site,"ACGT"[rb],ri)
    iterations = 1
    rs = [(random.randrange(L),random.randrange(4),random.random())
          for i in range(iterations)]
    converged = False
    while not converged:
        for ri, rb, r in rs:
            for traj in trajs:
                x = traj[-1]
                xp = mutate_site(x,(ri, rb))
                if log(r) < log_phat(xp) - log_phat(x):
                    x = xp
                traj.append(x)
        if trajs[0][-1] == trajs[-1][-1]:
            converged = True
        iterations *= 2
        rs = [(random.randrange(L),random.randrange(4),random.random())
              for i in range(iterations)] + rs
    assert all(map(lambda traj:traj[-1] == trajs[0][-1],trajs))
    #return trajs[0][-1]
    return trajs

def sample_site_cftp(matrix, mu, Ne):
    L = len(matrix)
    f = seq_scorer(matrix)
    def log_phat(s):
        ep = f(s)
        nu = Ne - 1
        return -nu*log(1 + exp(ep - mu))
    first_site = "A"*L
    last_site = "T"*L
    best_site = "".join(["ACGT"[argmin(row)] for row in matrix])
    worst_site = "".join(["ACGT"[argmax(row)] for row in matrix])
    #middle_sites  = [[random_site(L)] for i in range(10)]
    #trajs = [[best_site]] + middle_sites + [[worst_site]]
    trajs = [[best_site],[worst_site]]
    ords = [rslice("ACGT",sorted_indices(row)) for row in matrix]
    def mutate_site(site,(ri,direction)):
        b = (site[ri])
        idx = ords[ri].index(b)
        idxp = min(max(idx + direction,0),3)
        bp = ords[ri][idxp]
        return subst(site,bp,ri)
    iterations = 1
    rs = [(random.randrange(L),random.choice([-1,1]),random.random())
          for i in range(iterations)]
    converged = False
    while not converged:
        for ri, rdir, r in rs:
            for traj in trajs:
                x = traj[-1]
                xp = mutate_site(x,(ri, rdir))
                if log(r) < log_phat(xp) - log_phat(x):
                    x = xp
                traj.append(x)
        if trajs[0][-1] == trajs[-1][-1]:
            converged = True
        iterations *= 2
        #print iterations,[traj[-1] for traj in trajs]
        rs = [(random.randrange(L),random.choice([-1,1]),random.random())
              for i in range(iterations)] + rs
    assert all(map(lambda traj:traj[-1] == trajs[0][-1],trajs))
    return trajs[0][-1]
    #return trajs

def pos(site):
    return sum(4**i * "ACGT".index(site[i]) for i,b in enumerate(site))

def sample_motif_cftp(matrix, mu, Ne, n,verbose=False):
    iterator = trange(n,desc="sampling cftp motif") if verbose else xrange(n)
    return [sample_site_cftp(matrix, mu, Ne)
            for i in iterator]
    
def spoof_motif_cftp(motif, num_motifs=10, trials=1, sigma=None,Ne_tol=10**-2):
    n = len(motif)
    L = len(motif[0])
    copies = 10*n
    if sigma is None:
        sigma = sigma_from_matrix(pssm_from_motif(motif,pc=1))
    print "sigma:", sigma
    bio_ic = motif_ic(motif)
    matrix = sample_matrix(L, sigma)
    mu = approx_mu(matrix, copies=10*n, G=5*10**6)
    print "mu:", mu
    def f(Ne):
        motifs = [sample_motif_cftp(matrix, mu, Ne, n)
                  for i in trange(trials)]
        return mean(map(motif_ic,motifs)) - bio_ic
    # lb = 1
    # ub = 10
    # while f(ub) < 0:
    #     ub *= 2
    #     print ub
    x0s = [2,10]#(lb + ub)/2.0
    # print "choosing starting seed for Ne"
    # fs = map(lambda x:abs(f(x)),x0s)
    # print "starting values:",x0s,fs
    # x0 = x0s[argmin(fs)]
    # print "chose:",x0
    # Ne = bisect_interval_noisy_ref(f,x0,lb=1,verbose=True)
    Ne = log_regress_spec(f,x0s,tol=Ne_tol)
    print "Ne:",Ne
    return [sample_motif_cftp(matrix, mu, Ne, n) for _ in trange(num_motifs)]

def log_regress(f,xs, tol=0.1):
    """find root f(x) = 0 using logistic regression, starting with xs"""
    last_log_xp = None
    log_xp = None
    print "initial seeding for log_regress"
    ys = map(f,xs)
    log_xs = map(log,xs)
    plotting = False
    while last_log_xp is None or abs(exp(log_xp) - exp(last_log_xp)) > tol:
        print "correlation:",pearsonr(log_xs,ys)
        #lin = poly1d(polyfit(log_xs,ys,1))
        m, b = (polyfit(log_xs,ys,1))
        if plotting:
            lin = poly1d([m,b])
            plt.scatter(log_xs,ys)
            plt.plot(*pl(lin,log_xs))
            plt.show()
        last_log_xp = log_xp or None
        log_xp = -b/m#secant_interval(lin,min(log_xs),max(log_xs))
        log_xs.append(log_xp)
        yxp = f(exp(log_xp))
        ys.append(yxp)
        print "x:",log_xp and exp(log_xp),"y:",ys[-1],\
               "x last:",last_log_xp and exp(last_log_xp),\
               "y last:",ys[-2] if len(ys) >= 2 else None
    #lin = poly1d(polyfit(log_xs,ys,1))
    m, b = (polyfit(log_xs,ys,1))
    log_xp = -b/m#secant_interval(lin,min(log_xs),max(log_xs))
    return exp(log_xp)

def log_regress_spec(f,xs, tol=0.1):
    """find root f(x) = 0 using logistic regression, starting with xs"""
    print "initial seeding for log_regress"
    ys = map(f,xs)
    log_xs = map(log,xs)
    plotting = False
    honest_guesses = []
    while len(honest_guesses) < 2 or abs(exp(honest_guesses[-1]) -
                                     exp(honest_guesses[-2])) > tol:
        #print "correlation:",pearsonr(log_xs,ys)
        #lin = poly1d(polyfit(log_xs,ys,1))
        m, b = (polyfit(log_xs,ys,1))
        if plotting:
            lin = poly1d([m,b])
            plt.scatter(log_xs,ys)
            plt.plot(*pl(lin,log_xs))
            plt.show()
        honest_guess = -b/m
        dx = -(honest_guesses[-1] - honest_guess) if honest_guesses else 0
        log_xp = honest_guess + dx
        log_xs.append(log_xp)
        yxp = f(exp(log_xp))
        ys.append(yxp)
        honest_guesses.append(honest_guess)
        diff = (abs(exp(honest_guesses[-1]) - exp(honest_guesses[-2]))
                if len(honest_guesses) > 1 else None)
        print "honest_guess:",exp(honest_guess),"xp:",exp(log_xp),\
            "y:",yxp, "diff:",diff
    #lin = poly1d(polyfit(log_xs,ys,1))
    m, b = (polyfit(log_xs,ys,1))
    log_xp = -b/m#secant_interval(lin,min(log_xs),max(log_xs))
    print "final guess: log_xp:",log_xp
    return exp(log_xp)

    
def sample_motif_cftp_param_study():
    """Examine dependence of IC on sigma, Ne"""
    grid_points = 10
    sigmas = np.linspace(0.5,10,grid_points)
    Nes = np.linspace(1,10,grid_points)
    trials = 3
    n = 20
    L = 10
    def f(sigma, Ne):
        matrix = sample_matrix(L, sigma)
        mu = approx_mu(matrix, 10*n)
        return motif_ic(sample_motif_cftp(matrix, mu, Ne, n))
    ics = [[(mean(f(sigma, Ne) for _ in range(trials)))
            for sigma in sigmas] for Ne in tqdm(Nes,desc="ic grid")]
    plt.contourf(sigmas, Nes,ics)
    plt.colorbar()
    #bio_motifs = [getattr(Escherichia_coli,tf) for tf in Escherichia_coli.tfs]
    bio_sigmas = [sigma_from_matrix(pssm_from_motif(motif,pc=1))
                  for motif in bio_motifs]
    bio_ics = [motif_ic(motif) for motif in bio_motifs]
    #griddata((sigmas,Nes),ics)
    interp = interp2d(sigmas,Nes,ics)
    bio_Nes = [bisect_interval(lambda Ne:interp(show(bio_sigma),Ne)-bio_ic,0,20)
               for bio_sigma, bio_ic in zip(bio_sigmas,bio_ics)]
    plt.scatter(sigm)

def mh_cftp_comparison():
    matrix = sample_matrix(10,1)
    mu = -10
    Ne = 5
    mh_ics = [motif_ic(sample_motif_mh(matrix, mu, Ne, 50)) for i in trange(100)]
    cftp_ics = [motif_ic(sample_motif_cftp(matrix, mu, Ne, 50)) for i in trange(100)]
