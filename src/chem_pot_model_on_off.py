from utils import choose, log_choose, inverse_cdf_sample, permute, secant_interval, show, normalize, kde_regress
from utils import maybesave, pl, bisect_interval, mean, log2, motif_ic, score_seq
from utils import bisect_interval_noisy, mmap, motif_gini, total_motif_mi, choose2
from formosa_utils import approx_mu as gle_approx_mu
from math import log, exp
from scipy.optimize import minimize
from scipy.stats import pearsonr
import random
import inspect
import numpy as np
from pwm_utils import site_sigma_from_matrix, psfm_from_motif, sigma_from_matrix
from pwm_utils import pssm_from_motif, matrix_from_motif
from matplotlib import pyplot as plt
from tqdm import *
from exact_evo_sim_sampling import log_regress_spec2
import itertools

G = 5*10**6

def prior(k,L):
    """prior probability of observing k mismatches in L sites"""
    return choose(L,k)*(1/4.0)**(L-k)*(3/4.0)**k

def fd(ep,mu):
    return 1/(1+exp(ep-mu))

def occ_from_mismatches(k,L,sigma,mu):
    matches = L-k
    ep = -sigma*matches
    return fd(ep,mu)

def ep_from_mismatch(k,L,sigma):
    return -sigma*(L-k)
    
def phat(k,sigma,mu,Ne,L):
    nu = Ne - 1
    return fd(-sigma*(L-k),mu)**nu*prior(k,L)

def mean_Zb_ref(sigma,L):
    return sum(exp(sigma*(L-k))*prior(k,L) for k in range(L+1))

def mean_Zb(sigma,L):
    return ((exp(sigma)+3)/4)**L
    
def test_mean_Zb(sigma,L,trials=10000):
    return mean(exp(sigma*sum(random.random() < 0.25 for i in range(L))) for j in trange(trials))
    
def phat_approx(k,sigma,mu,Ne,L):
    nu = Ne - 1
    return (1/(1+exp(-sigma*(L-k)-mu-log(nu))))*prior(k,L)

def copy_num_from(G,sigma,L,mu):
    return G*sum(prior(k,L)*occ_from_mismatches(k,L,sigma,mu) for k in range(L+1))

def mu_from(G,sigma,L,copy_num):
    f = lambda mu:copy_num_from(G,sigma,L,mu) - copy_num
    return bisect_interval(f,-500,500)

def approx_mu(G,sigma,L,copy_num):
    return log(copy_num) - log(G*mean_Zb(sigma,L))

def mu_approx_fig(filename=None):
    sigma = 1
    L = 10
    copy_range = np.linspace(1,10**5,100)
    plt.plot(*pl(lambda copy_num:mu_from(G,sigma,L,copy_num=copy_num),copy_range),label="Exact")
    plt.plot(*pl(lambda copy_num:approx_mu(G,sigma,L,copy_num=copy_num),copy_range),label="Approx")
    plt.xlabel("Copy number")
    plt.ylabel("$\mu$")
    plt.semilogx()
    plt.legend(loc='ul')
    plt.title("Exact vs. Approximate Chemical Potential")
    maybesave(filename)

def p(k,sigma,mu,Ne,L):
    return phat(k,sigma,mu,Ne,L)/sum(phat(kp,sigma,mu,Ne,L) for kp in range(L+1))

def p_approx(k,sigma,mu,Ne,L):
    return phat_approx(k,sigma,mu,Ne,L)/sum(phat_approx(kp,sigma,mu,Ne,L) for kp in range(L+1))

def p_from_copies(k,sigma,Ne,L,copies):
    mu = mu_from(G,sigma,L,copies)
    return p(k,sigma,mu,Ne,L)

def ps_from_copies(sigma,Ne,L,copies,approx=True):
    #print "ps from copies:", sigma, Ne, L, copies
    if approx:
        mu = approx_mu(G, sigma, L, copies)
    else:
        mu = mu_from(G,sigma,L,copies)
    return normalize([phat(k,sigma,mu,Ne,L) for k in range(L+1)])

def sample_site(sigma,mu,Ne,L):
    phats = [phat(k,sigma,mu,Ne,L) for k in range(L+1)]
    # Z = sum(phats)
    # ps = [ph/Z for ph in phats]
    k = inverse_cdf_sample(range(L+1), phats,normalized=False)
    return "".join(permute(["A" for _ in range(L-k)] + [random.choice("CGT") for _ in range(k)]))

def sample_site_from_copies(sigma,Ne,L,copies,ps=None):
    if ps is None:
        ps = ps_from_copies(sigma, Ne, L, copies)
    k = inverse_cdf_sample(range(L+1), ps,normalized=False)
    return "".join(permute(["A" for _ in range(L-k)] + [random.choice("CGT") for _ in range(k)]))
    
def sample_motif(sigma, Ne, L, copies, n, ps=None):
    if ps is None:
        ps = ps_from_copies(sigma, Ne, L, copies)
    return [sample_site_from_copies(sigma,Ne,L,copies,ps=ps) for _ in range(n)]

def sigma_Ne_contour_plot(filename=None):
    sigmas = np.linspace(0,5,20)
    Nes = np.linspace(1,20,20)
    L = 10
    n = 50
    copies = 10*n
    trials = 100
    motifss = [[[(sample_motif(sigma, Ne, L, copies, n))
               for i in range(trials)]
          for sigma in sigmas] for Ne in tqdm(Nes)]
    occ_M = [[expected_occupancy(sigma, Ne, L, copies)
          for sigma in sigmas] for Ne in tqdm(Nes)]
    print "ic_M"
    ic_M = mmap(lambda ms:mean(map(motif_ic,ms)),motifss)
    print "gini_M"
    gini_M = mmap(lambda ms:mean(map(motif_gini,ms)),motifss)
    print "mi_M"
    mi_M = mmap(lambda ms:mean(map(total_motif_mi,ms)),tqdm(motifss))
    plt.subplot(2,2,1)
    plt.contourf(sigmas,Nes,occ_M,cmap='jet')
    plt.colorbar()
    plt.subplot(2,2,2)
    plt.contourf(sigmas,Nes,ic_M,cmap='jet')
    plt.colorbar()
    plt.subplot(2,2,3)
    plt.contourf(sigmas,Nes,gini_M,cmap='jet')
    plt.colorbar()
    plt.subplot(2,2,4)
    plt.contourf(sigmas,Nes,mi_M,cmap='jet')
    plt.colorbar()
    maybesave(filename)
    
def spoof_motif_ref(motif, num_motifs=10, trials=10, sigma=None,Ne_tol=10**-4):
    n = len(motif)
    L = len(motif[0])
    copies = 10*n
    if sigma is None:
        sigma = sigma_from_matrix(pssm_from_motif(motif,pc=1))
    print "sigma:", sigma
    bio_ic = motif_ic(motif)
    def f(Ne):
        ps = ps_from_copies(sigma, Ne, L, copies)
        motifs = [sample_motif(sigma, Ne, L, copies, n,ps=ps)
                  for i in range(trials)]
        return mean(map(motif_ic,motifs)) - bio_ic
    lb = 1
    ub = 2
    while f(ub) < 0:
        ub *= 2
    ub *= 2 # once more for good measure
    x0 = (lb + ub)/2.0
    print "Ne guess:", x0
    Nes = [bisect_interval_noisy(f,x0=x0,tolerance=Ne_tol,lb=1) for i in range(3)]
    Ne = mean(Nes)
    print "Nes:",Nes,Ne
    return [sample_motif(sigma, Ne, L, copies, n) for _ in range(num_motifs)]

def spoof_motifs(motif, num_motifs=10, trials=1, sigma=None,Ne_tol=10**-4,double_sigma=True):
    N = len(motif)
    L = len(motif[0])
    copies = 10*N
    if sigma is None:
        sigma = sigma_from_matrix(pssm_from_motif(motif,pc=1))
    epsilon = (1+double_sigma)*sigma # 15 Jan 2016
    print "sigma:", sigma
    bio_ic = motif_ic(motif)
    def f(Ne):
        ps = ps_from_copies(sigma, Ne, L, copies)
        motifs = [sample_motif(epsilon, Ne, L, copies, n,ps=ps)
                  for i in range(trials)]
        return mean(map(motif_ic,motifs)) - bio_ic
    Ne = log_regress_spec2(f,[1,10],tol=10**-3)
    return [sample_motif(sigma, Ne, L, copies, n) for _ in range(num_motifs)]

def spoof_motifs_occ(motif, num_motifs=10, trials=1, sigma=None,Ne_tol=10**-4,double_sigma=True):
    N = len(motif)
    L = len(motif[0])
    copies = 10*N
    if sigma is None:
        sigma = sigma_from_matrix(pssm_from_motif(motif,pc=1))
    epsilon = (1+double_sigma)*sigma # 15 Jan 2016
    print "sigma:", sigma
    #bio_ic = motif_ic(motif)
    mat = matrix_from_motif(motif)
    eps = [score_seq(mat, site) for site in motif]
    mu = gle_approx_mu(mat, copies)
    bio_occ = mean([1/(1+exp(ep-mu)) for ep in eps])
    def f(Ne):
        return expected_occupancy(epsilon, Ne, L, copies) - bio_occ
    Ne = log_regress_spec2(f,[1,10],tol=10**-3)
    return [sample_motif(sigma, Ne, L, copies, N) for _ in range(num_motifs)]

    
def sample_ic(sigma,mu,Ne,L,n,trials=1000):
    return mean(motif_ic(sample_motif(sigma,mu,Ne,L,n)) for i in range(trials))

def expected_ic(sigma,Ne,L,copies):
    p = mismatch_probability(sigma,Ne,L,copies)
    q = 1-p
    return L*(2 - (-(q*log2(q) + 3*(p/3)*log2(p/3))))

def expected_mi_from_undersampling(sigma,Ne,L,copies,n):
    p = mismatch_probability(sigma,Ne,L,copies)
    q = 1-p
    Hx = -(q*log2(q) + p*log2(p/3))
    Hxy = -(q**2*log2(q**2) + 2*p*q*log2(p*q/3) + p**2*log2(p**2/9))
    return 

def expected_occupancy(sigma,Ne,L,copies):
    ps = ps_from_copies(sigma, Ne, L, copies)
    mu = mu_from(G, sigma, L, copies)
    return sum(occ_from_mismatches(k, L, sigma, mu)*p for (k, p) in enumerate(ps))

def expected_occ_spec(sigma,Ne,L,copies,epsilon=0.01):
    nu = Ne - 1
    w = lambda k:prior(k,L)
    mu = mu_from(G,sigma,L,copies)
    f = lambda k:occ_from_mismatches(k,L,sigma,mu)
    g = lambda nu:sum(w(k)*f(k)**(nu) for k in range(L+1))
    occ_check = g(nu+1)/g(nu)
    occ_check2 = g(nu+1)/(1.0/(nu+1)*diff(g,nu+1))
    print "occ check:",occ_check
    print "occ check2:",occ_check2
    occ_inv = 1.0/(nu+1) * (log(g(nu+1+epsilon)) - log(g(nu+1)))/epsilon
    return 1/occ_inv

def diff(f,x,epsilon=0.01):
    return (f(x+epsilon) - f(x))/epsilon

def mismatch_probability(sigma,Ne,L,copies):
    ps = ps_from_copies(sigma,Ne,L,copies)
    return sum(k*p for k,p in enumerate(ps))/float(L)

def sigma_scan(Ne,L,copies,trials=1,n=100,sigma_steps=100,max_sigma=10):
    sigma_range = np.linspace(1,max_sigma,sigma_steps)
    sigma = 1
    plt.subplot(1,4,1)
    obs_ics = map(lambda sigma:mean(motif_ic(sample_motif(sigma=sigma,Ne=Ne,L=L,n=n,copies=copies))
                                   for _ in range(trials)), sigma_range)
    pred_ics = map(lambda sigma:expected_ic(sigma,Ne,L,copies),sigma_range)
    occs = map(lambda sigma:expected_occupancy(sigma,Ne,L,copies),sigma_range)
    mismatches = map(lambda sigma:mismatch_probability(sigma,Ne,L,copies),sigma_range)
    mus = map(lambda sigma:mu_from(G,sigma,L,copies),sigma_range)
    approx_mus = map(lambda sigma:approx_mu(G,sigma,L,copies),sigma_range)
    mean_log_Zbs = map(lambda sigma:log(mean_Zb(sigma,L)),sigma_range)
    plt.plot(sigma_range,obs_ics)
    plt.plot(sigma_range,pred_ics)
    plt.plot(sigma_range,occs)
    plt.plot(sigma_range,mismatches)
    plt.plot(sigma_range,mus)
    plt.plot(sigma_range,approx_mus)
    plt.plot(sigma_range,mean_log_Zbs)
    plt.subplot(1,4,2)
    plt.plot(mismatches,pred_ics)
    plt.xlabel("Mismatches")
    plt.ylabel("IC")
    plt.subplot(1,4,3)
    plt.plot(pred_ics,occs)
    plt.xlabel("IC")
    plt.ylabel("Occupancy")
    plt.subplot(1,4,4)
    plt.plot(mismatches,occs)
    plt.xlabel("Mismatches")
    plt.ylabel("Occs")

def Ne_scan(sigma,L,copies,trials=1,n=100,max_Ne=10,Ne_steps=100):
    Ne_range = np.linspace(1,max_Ne,Ne_steps)
    sigma = 1
    plt.subplot(1,4,1)
    obs_ics = map(lambda Ne:mean(motif_ic(sample_motif(sigma=sigma,Ne=Ne,L=L,n=n,copies=copies))
                                   for _ in range(trials)), Ne_range)
    pred_ics = map(lambda Ne:expected_ic(sigma,Ne,L,copies),Ne_range)
    occs = map(lambda Ne:expected_occupancy(sigma,Ne,L,copies),Ne_range)
    mismatches = map(lambda Ne:mismatch_probability(sigma,Ne,L,copies),Ne_range)
    plt.plot(Ne_range,obs_ics)
    plt.plot(Ne_range,pred_ics)
    plt.plot(Ne_range,occs)
    plt.plot(Ne_range,mismatches)
    plt.subplot(1,4,2)
    plt.plot(mismatches,pred_ics)
    plt.xlabel("Mismatches")
    plt.ylabel("IC")
    plt.subplot(1,4,3)
    plt.plot(occs,pred_ics)
    plt.xlabel("Occupancy")
    plt.ylabel("IC")
    plt.subplot(1,4,4)
    plt.plot(mismatches,occs)
    plt.xlabel("Mismatches")
    plt.ylabel("Occs")


    
def hessian_experiment(trials=1000):
    sigma0 =   1
    mu0    = -10
    Ne0    =   5
    L0     =  10
    n0     =  50
    # n,L are fixed; vary sigma, mu, Ne.
    ic0 = sample_ic(sigma0,mu0,Ne0,L0,n0)
    epsilon = 0.01
    f = lambda x,y,z: show((sample_ic(sigma0+x,mu0+y,Ne0+z,L0,n0,trials=trials)-ic0)**2)
    hessian = compute_hessian(f,(sigma0,mu0,Ne0),epsilon=0.1)
    hessian2 = compute_hessian(f2,(sigma0,mu0,Ne0),epsilon=0.1)
    lambs, vs = np.linalg.eig(hessian)
    return lambs, vs

def hessian_experiment2(epsilon = 0.01):
    sigma =   1
    Ne    =   5
    L     =  10
    copies = 10
    # n,L are fixed; vary sigma, mu, Ne.
    ic0 = expected_ic(sigma,Ne,L,copies)
    f = lambda x,y,z: (expected_ic(sigma+x,Ne+y,L,copies+z)-ic0)**2
    tup = (0,0,0)
    hessian = compute_hessian(f,tup,epsilon)
    lambs, vs = np.linalg.eig(hessian)
    return lambs, vs

def fit_motif(motif):
    n = len(motif)
    L = len(motif[0])
    bio_ic = motif_ic(motif)
    def f((sigma,Ne,copies)):
        return expected_ic(sigma,Ne,L,copies)-bio_ic
    def fsq((sigma,Ne,copies)):
        return f((sigma,Ne,copies))**2
    x0 = (1,2,n)
    fit_params = minimize(fsq,x0,bounds=((0,10),(1,None),(1,None)),method='TNC').x
    return fit_params

def motif_ls_sq_surface_experiment():
    motif = Escherichia_coli.LexA
    L = len(motif[0])
    bio_ic = motif_ic(motif)
    sigma,Ne,copies = fit_motif(motif)
    fit_ic = expected_ic(sigma,Ne,L,copies)
    f = lambda (sigma,Ne,copies):expected_ic(sigma,Ne,L,copies)
    gr = compute_grad(f,(sigma,Ne,copies),epsilon=0.0001)
    hess = compute_hessian(f,(sigma,Ne,copies))
    lambs, vs = np.linalg.eig(hess)
    def orth_comp(y,z):
        a,b,c = gr
        x = -(b*y+c*z)/a
        return np.array([x,y,z])
    
def compute_hessian(f,tup,epsilon=0.01):
    """compute hessian of f at point tup"""
    N = len(tup)
    def H(i,j,epsilon=epsilon):
        """return ith element of hessian"""
        grid = [(1,1),(-1,1),(1,-1),(-1,-1)]
        tup1,tup2,tup3,tup4 = [tuple(arg + epsilon*(i_offset*(arg_idx==i) + j_offset*(arg_idx==j))
                                for arg_idx,arg in enumerate(tup))
                               for (i_offset,j_offset) in grid]
        return (f(tup1) - f(tup2) - f(tup3) + f(tup4))/(4.0*epsilon**2)
    return np.matrix([[H(i,j) for j in range(N)] for i in range(N)])

def compute_grad(f,tup,epsilon=0.01):
    """compute gradient of f at point tup"""
    f0 = f(tup)
    tups = [tuple(arg+epsilon*(arg_idx == i) for arg_idx,arg in enumerate(tup)) for i in range(len(tup))]
    return np.array([(f(t)-f0) for t in tups])/epsilon
    
def normalize_spectrum(lambs):
    lamb1 = max(lambs)
    log10(lambs/lamb1)

def expected_mi(sigma,Ne,L,copies):
    """How much MI should you expect due to ROR effect?"""
    ps = ps_from_copies(sigma,Ne,L,copies)
    misX = sum(k*p for k,p in enumerate(ps))/L
    matX = 1-misX
    misY = misX
    matY = matX
    L = float(L)
    matYmatX = sum(ps[k]*((L-k)/L)*(L-k-1)/(L-1) for k in range(int(L+1)))
    matYmisX = sum(ps[k]*((L-k)/L)*(k)/(L-1) for k in range(int(L+1)))
    misYmatX = matYmisX
    misYmisX = sum(ps[k]*(k/L)*(k-1)/(L-1) for k in range(int(L+1)))
    #print "joints sum to:", (matYmatX + matYmisX + misYmatX + misYmisX)
    HX = HY = -(matX*log2(matX) + 3*(misX/3)*log2(misX/3))
    #print "HX:",HX
    MI_ref = (misYmisX*log2(misYmisX/(misY*misX)) +
           matYmisX*log2(matYmisX/(matY*misX)) +
           misYmatX*log2(misYmatX/(misY*matX)) +
           matYmatX*log2(matYmatX/(matY*matX)))
    MI = (9*(misYmisX/9)*log2((misYmisX/9)/((misY/3)*(misX/3))) +
           3*(matYmisX/3)*log2((matYmisX/3)/(matY*(misX/3))) +
           3*(misYmatX/3)*log2((misYmatX/3)/((misY/3)*matX)) +
           matYmatX*log2(matYmatX/(matY*matX)))
    return (MI)*choose(int(L),2)

def test_expected_mi(trials=1000,n=1000):
    pred = []
    obs = []
    for i in trange(trials):
        sigma = random.random()*10
        Ne = random.random() *10 + 1
        L = random.randrange(10,20)
        copies = random.randrange(1,100)
        pred.append(expected_mi(sigma,Ne,L,copies))
        obs.append(total_motif_mi(sample_motif(sigma,Ne,L,copies,n=n)))
    print pearsonr(pred,obs)
    pred_obs(zip(pred,obs))

def check_miss_miss(k,L,trials=1000):
    combs = list(itertools.combinations(range(int(L)),k))
    def trial():
        hits = random.choice(combs)
        return (0 not in hits) and (1 not in hits)
    return mean(trial() for _ in range(trials))

def check_miss_hit(k,L,trials=1000):
    combs = list(itertools.combinations(range(int(L)),k))
    def trial():
        hits = random.choice(combs)
        return (0 in hits) and (1 not in hits)
    return mean(trial() for _ in range(trials))

def check_hit_hit(k,L,trials=1000):
    combs = list(itertools.combinations(range(int(L)),k))
    def trial():
        hits = random.choice(combs)
        return (0 in hits) and (1  in hits)
    return mean(trial() for _ in range(trials))

def mm_site(L,k):
    """return site consisting of L-k matches and k mismatches, uniformly at random"""
    mms = random.choice(list(itertools.combinations(range(L),k)))
    return "".join(["A" if not i in mms else random.choice("CGT") for i in range(L)])

def mm_motif(L,n,k):
    return [mm_site(L,k) for i in xrange(n)]

def expected_ic(L,k):
    L = float(L)
    q = k/L # prob mismatch
    p = 1 - q
    h_per_col = - ((p*log2(p) if p else 0) + (q*log2(q/3) if q else 0))
    ic_per_col = 2 - h_per_col
    return ic_per_col * L

def test_expected_ic(L,n):
    plt.plot([expected_ic(L,k) for k in range(L + 1)])
    plt.scatter(range(L+1),[motif_ic(mm_motif(L,n,k))
                            for k in trange(L + 1)])
    
    
def expected_mi(L,k):
    """expected MI between two columns, given site length L and k mismatches per site"""
    L = float(L)
    q = k/L # prob mismatch
    p = 1 - q
    match = lambda b:b=="A"
    mismatch = lambda b:b!="A"
    def joint(b1,b2):
        if match(b1):
             if match(b2):
                 return (L-k)/L * (L-k-1)/(L-1)
             else:
                 return (L - k)/L * k/(L - 1) / 3.0
        else:
            if match(b2):
                return (k)/L * (L-k)/(L-1) / 3.0
            else:
                return (k)/L * (k-1)/(L-1) / 9.0
    def marg(b):
        return p if match(b) else q/3
    return sum(joint(b1,b2)*log2(joint(b1,b2)/(marg(b1)*marg(b2))) if joint(b1,b2) else 0
               for b1,b2 in choose2("ACGT"))

def expected_mi2(L,k):
    possible_sites = choose(L,k)*(1**(L-k))*(3**k)
    total_IC = 2*L - log2(possible_sites)
    return total_IC - expected_ic(L,k)

def test_expected_mi(L,n):
    plt.plot([expected_mi(L,k)*choose(L,2) for k in range(L + 1)], label="Expected MI")
    plt.plot([expected_mi2(L,k) for k in range(L + 1)], label="Expected MI2")
    plt.scatter(range(L+1),[mean((total_motif_mi(mm_motif(L,n,k))) for i in range(10))
                            for k in trange(L + 1)])
        
    
