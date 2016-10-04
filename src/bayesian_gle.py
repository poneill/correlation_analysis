"""find GLE evo parameters through bayesian analysis"""

from utils import score_seq, mh, mean, maybesave, kmers, variance, random_site, sign, prod, random_motif
from utils import logsum, dnorm, secant_interval, zipWith, motif_ic, bisect_interval
from utils import scatter, h, mutate_site, argmin, choose, log_choose
from utils import normalize
from pwm_utils import psfm_from_matrix, sample_from_psfm, psfm_from_motif, sample_matrix, site_sigma_from_matrix
from pwm_utils import sigma_from_matrix
from math import log, exp, sqrt
import random
from evo_sampling import sample_motif_cftp
from tqdm import *
from matplotlib import pyplot as plt
from scipy.stats import norm
import numpy as np
from scipy import integrate
from formosa import spoof_maxent_motifs
from adjacent_pairwise_model import code_from_motif, sample_site

def log_fhat((matrix, mu, Ne), motif):
    assert type(motif) is list
    nu = Ne - 1
    eps = [score_seq(matrix, site) for site in motif]
    return -sum(nu*log(1+exp(ep-mu)) for ep in eps)

def log_ZS_analytic((matrix, mu, Ne)):
    """compute log_Z analytically"""
    acc = 0
    nu = Ne - 1
    L = len(matrix)
    for kmer in kmers(L):
        ep = score_seq(matrix, "".join(kmer))
        acc += (1/(1+exp(ep-mu)))**(Ne-1)
    return log(acc)

def log_ZM_analytic((matrix, mu, Ne), N):
    log_ZS = log_ZS_analytic((matrix, mu, Ne))
    return N * log_ZS
    
def log_Z_analytic((matrix, mu, Ne), N):
    """compute log_Z analytically"""
    acc = 0
    nu = Ne - 1
    L = len(matrix)
    for kmer in kmers(L):
        ep = score_seq(matrix, "".join(kmer))
        acc += (1/(1+exp(ep-mu)))**(Ne-1)
    return N * log(acc)

def log_ZS_naive((matrix, mu, Ne), trials=1000):
    acc = 0
    nu = Ne - 1
    L = len(matrix)
    for i in xrange(trials):
        ep = score_seq(matrix, random_site(L))
        acc += (1/(1+exp(ep-mu)))**(Ne-1)
    mean_Zs = acc / trials
    return L * log(4) + log(mean_Zs)

def log_ZM_naive((matrix, mu, Ne), N, trials=1000):
    return N * log_ZS_naive((matrix, mu, Ne), trials=1000)
    
def log_ZS_hack((matrix, mu, Ne), N):
    L = len(matrix)
    mat_mu = sum(map(mean,matrix))
    mat_sigma = sqrt(sum(map(lambda xs:variance(xs,correct=False), matrix)))
    log_perc_below_threshold = norm.logcdf(mu - log((Ne-1)), mat_mu, mat_sigma)
    log_Zs = L * log(4) + log_perc_below_threshold
    return log_Zs

def log_ZM_hack((matrix, mu, Ne), N):
    log_ZS = log_ZS_hack((matrix, mu, Ne), N)
    return N * log_ZS

def log_Z_hack((matrix, mu, Ne), N):
    L = len(matrix)
    mat_mu = sum(map(mean,matrix))
    mat_sigma = sqrt(sum(map(lambda xs:variance(xs,correct=False), matrix)))
    log_perc_below_threshold = norm.logcdf(mu - log((Ne-1)), mat_mu, mat_sigma)
    log_Zs = L * log(4) + log_perc_below_threshold
    ans_ref = ((N*L * log(4)) +  log_perc_below_threshold)
    ans = N * log_Zs
    return ans

def log_ZS_importance((matrix, mu, Ne), trials=1000):
    L = len(matrix)
    psfm = psfm_from_matrix(matrix)
    log_psfm = [[log(p) for p in row] for row in psfm]
    log_psfm_prob = lambda site:score_seq(log_psfm, site)
    sites = [sample_from_psfm(psfm) for _ in xrange(trials)]
    mean_ZS = mean(exp(log_fhat((matrix, mu, Ne), [site]) + log(1.0/4**L) - log_psfm_prob(site))
                  for site in sites)
    ZS = 4**L * mean_ZS
    return log(ZS)

def log_ZS_importance_ref((matrix, mu, Ne), trials=1000):
    L = len(matrix)
    psfm = [[0.25]*4 for _ in range(L)]
    log_psfm = [[log(p) for p in row] for row in psfm]
    log_psfm_prob = lambda site:score_seq(log_psfm, site)
    sites = [sample_from_psfm(psfm) for _ in xrange(trials)]
    mean_ZS = mean(exp(log_fhat((matrix, mu, Ne), [site]) + log(1.0/4**L) - log_psfm_prob(site))
                  for site in sites)
    ZS = 4**L * mean_ZS
    return log(ZS)

def log_ZS_importance2((matrix, mu, Ne), trials=1000):
    y = mu - log(Ne)
    def expectation(lamb):
        psfm = [normalize([exp(-lamb*ep) for ep in row]) for row in matrix]
        return sum(ep*p for row, ps in zip(matrix, psfm) for ep,p in zip(row,ps))
    lamb = secant_interval(lambda x:expectation(x)-y,-10,10)
        
        
def log_ZM_importance((matrix, mu, Ne), N, trials=1000):
    log_ZS = log_ZS_importance((matrix, mu, Ne), trials=trials)
    return N * log_ZS
    
def log_ZS_empirical((matrix, mu, Ne), trials=1000):
    L = len(matrix)
    acc = 0
    for i in xrange(trials):
        ep = score_seq(matrix, random_site(L))
        acc += 1.0/(1+exp(ep-mu))**(Ne-1)
    est_mean = acc / trials
    log_Zs = L*log(4) + log(est_mean)
    return log_Zs

def log_ZS_empirical_ref((matrix, mu, Ne), trials=1000):
    L = len(matrix)
    sites = [random_site(L) for _ in xrange(trials)]
    mean_ZS = mean(exp(log_fhat((matrix, mu, Ne), [site])) for site in sites)
    log_ZS = L*log(4) + log(mean_ZS)
    return log_ZS

def log_ZM_empirical((matrix, mu, Ne), N, trials=1000):
    log_ZS = log_ZS_empirical((matrix, mu, Ne),  trials=trials)
    return N * log_ZS

def log_ZM_empirical_ref((matrix, mu, Ne), N, trials=1000):
    L = len(matrix)
    acc = 0
    for i in xrange(trials):
        eps = [score_seq(matrix, random_site(L)) for _ in range(N)]
        acc += prod(1.0/(1+exp(ep-mu))**(Ne-1) for ep in eps)
    est_mean = acc / trials
    log_Zs = N*L*log(4) + log(est_mean)
    return log_Zs

def log_ZM_empirical_ref2(theta, N, trials=1000):
    L = len(theta[0])
    lfhs = [log_fhat(theta, random_motif(L, N)) for _ in xrange(trials)]
    return N*L * log(4) + logsum(lfhs) - log(trials)
    
def log_ZM_empirical_ref3(theta, N,trials=1000):
    L = len(theta[0])
    lfhs = [log_fhat(theta, random_motif(L, 1)) for _ in xrange(trials)]
    log_avg = logsum(lfhs) - log(trials)
    log_ZS = L*log(4) + log_avg
    log_ZM = N * log_ZS
    return log_ZM

def log_Z_empirical((matrix, mu, Ne), N, trials=1000):
    L = len(matrix)
    acc = 0
    for i in xrange(trials):
        ep = score_seq(matrix, random_site(L))
        acc += 1.0/(1+exp(ep-mu))**(Ne-1)
    est_mean = acc / trials
    log_Zs = L*log(4) + log(est_mean)
    ans_ref = (N * L * log(4) + log(est_mean))
    ans = N * log_Zs
    #assert ans == ans_ref
    return ans

    
def occs((matrix, mu, Ne), motif):
    eps = [score_seq(matrix, site) for site in motif]
    return [1/(1+exp(ep-mu)) for ep in eps]
    
def prop((matrix, mu, Ne)):
    min_Ne = 1.1
    sigma = 0.1
    matrix_p = [[w+random.gauss(0,sigma) for w in row] for row in matrix]
    mu_p = mu + random.gauss(0,sigma)
    Ne_p = max(Ne + random.gauss(0,sigma), min_Ne)
    return (matrix_p, mu_p, Ne_p)

def prop2((matrix, mu, Ne), sigma=1):
    min_Ne = 1.1
    r = random.random()
    L = len(matrix)
    params = float(L * 4 + 2)
    if r < (L*4)/params:
        matrix_p = [row[:] for row in matrix]
        i = random.randrange(len(matrix))
        j = random.randrange(4)
        matrix_p[i][j] += random.gauss(0, sigma)
        return matrix_p, mu, Ne
    elif r < (L*4 + 1)/params :
        return matrix, mu + random.gauss(0, sigma), Ne
    else:
        return matrix, mu, max(Ne + random.gauss(0, sigma), min_Ne)

def prop_theta((sigma, mu, Ne)):
    min_Ne = 1.1
    min_sigma = 0.01
    sigma_p = max(min_sigma, sigma + random.gauss(0, 1))
    mu_p = mu + random.gauss(0,1)
    Ne_p = max(Ne + random.gauss(0,sigma), min_Ne)
    return (sigma_p, mu_p, Ne_p)
    
def posterior_chain(motif, iterations=50000, theta0=None, sigma=1, num_spoof_sites='N', verbose=False):
    """do MH with doubly intractable MCMC one-point estimator"""
    L = len(motif[0])
    N = len(motif)
    if num_spoof_sites == 'N':
        num_spoof_sites = N  # should this be N or 1?
    if theta0 is None:
        matrix0 = [[0,0,0,0] for i in range(L)]
        mu0 = -10
        Ne0 = 3
        theta = (matrix0, mu0, Ne0)
    else:
        theta = theta0
    log_f_theta = log_fhat(theta, motif)
    chain = []
    acceptances = 0
    for it in trange(iterations):
        theta_p = prop2(theta, sigma)
        log_f_theta_p = log_fhat(theta_p, motif)
        matrix_p, mu_p, Ne_p = theta_p
        xp = sample_motif_cftp(matrix_p, mu_p, Ne_p, num_spoof_sites)
        log_Z = log_fhat(theta, xp)
        log_Z_p = log_fhat(theta_p, xp)
        log_ar = log_f_theta_p - log_f_theta + N/num_spoof_sites * (log_Z - log_Z_p)
        if log(random.random()) < log_ar:
            theta = theta_p
            log_f_theta = log_f_theta_p
            log_Z = log_Z_p
            acceptances += 1
        chain.append(theta)
        if verbose:
            print "log(f), log_Z:", log_f_theta, log_Z
            print "mean_ep:", mean(score_seq(theta[0],site) for site in motif)
            print "mean_occ:", mean(occs(theta, motif))
            print "mu, Ne:", theta[1], theta[2]
    print "acceptances:", acceptances/float(it+1)
    return chain

def posterior_chain2(motif, iterations=50000, theta0=None, sigma=1, num_spoof_sites="N", verbose=False,
                     integration='hack'):
    """do MH, estimating ratio of partition functions empirically"""
    L = len(motif[0])
    N = len(motif)
    if num_spoof_sites == "N":
        num_spoof_sites = N  # should this be N or 1?
    if theta0 is None:
        matrix0 = [[random.gauss(0,1) for _ in range(4)] for i in range(L)]
        mu0 = -10
        Ne0 = 2
        theta = (matrix0, mu0, Ne0)
    else:
        theta = theta0
    log_f_theta = log_fhat(theta, motif)
    #log_Z = log_ZM_gaussian(theta, N, integration=integration)
    log_Z = log_ZM_sophisticated(theta, N)
    chain = []
    acceptances = 0
    def log_prior((matrix, mu, Ne)):
        log_matrix_prior = sum([log(dnorm(ep,0,1)) for row in matrix for ep in row])
        log_mu_prior = log(dnorm(mu,0,10))
        log_Ne_prior = log(exp(-Ne))
        return log_matrix_prior + log_mu_prior + log_Ne_prior
        
    for it in trange(iterations):
        #print "Ne:", theta[2]
        theta_p = prop2(theta, sigma)
        log_f_theta_p = log_fhat(theta_p, motif)
        matrix_p, mu_p, Ne_p = theta_p
        #log_Z = log_ZM_gaussian(theta, N, trials=100, integration='quad')
        #log_Z_p = log_ZM_gaussian(theta_p, N, trials=100, integration='hack')
        #log_Z = log_ZM_gaussian(theta, N, trials=100, integration='quad')
        log_Z_p = log_ZM_sophisticated(theta_p, N)
        #log_Z_p = log_ZM_importance(theta_p, N, trials=100)
        log_ar = log_f_theta_p - log_f_theta  + (log_Z - log_Z_p) + log_prior(theta_p) - log_prior(theta)
        if log(random.random()) < log_ar:
            theta = theta_p
            log_f_theta = log_f_theta_p
            log_Z = log_Z_p
            acceptances += 1
        chain.append(theta)
        if verbose:
            print "log(f), log_Z:", log_f_theta, log_Z
            print "mean_ep:", mean(score_seq(theta[0],site) for site in motif)
            print "mean_occ:", mean(occs(theta, motif))
            print "mu, Ne:", theta[1], theta[2]
    print "acceptances:", acceptances/float(it+1)
    return chain

def motif_from_theta((matrix, mu, Ne), N):
    return sample_motif_cftp(matrix, mu, Ne, N)

def logmod(x):
    return sign(x)*log(abs(x) + 1)
    
def interpret_chain(chain, motif, filename=None):
    N = len(motif)
    log_fhats = [log_fhat(theta,motif) for theta in chain]
    log_Zs = [log_ZM_hack(theta,N) for theta in chain]
    log_ps = [lf - log_Z for (lf, log_Z) in zip(log_fhats, log_Zs)]
    plt.plot(map(logmod, [mean(score_seq(x[0],site) for site in motif) for x in chain]),
             label="Mean Site Energy (kBT)")
    plt.plot(map(logmod, [x[1] for x in chain]),label="$\mu$ (kBT)")
    plt.plot(map(logmod, [x[2] for x in chain]),label="$Ne$")
    plt.plot(map(logmod, log_fhats),label="log fhat")
    plt.plot(map(logmod, log_Zs),label="log_ZM")
    plt.plot(map(logmod, log_ps),label="log p")
    plt.plot(map(logmod, [mean(occs(x, motif)) for x in chain]),label="Mean Occupancy")
    plt.legend(loc='right',fontsize='large')
    plt.xlabel("Iteration",fontsize='large')
    maybesave(filename)

def rejection_sample_site((matrix, mu, Ne)):
    psfm = psfm_from_matrix(matrix)
    log_psfm = [[log(p) for p in row] for row in psfm]
    log_psfm_prob = lambda site:score_seq(log_psfm, site)
    log_M = -sum(map(max,psfm))
    sites = [sample_from_psfm(psfm) for _ in xrange(trials)]
    log_fs = [log_fhat((matrix, mu, Ne), [site]) for site in sites]
    log_qs = [log_psfm_prob(site) for site in sites]
    ars = [exp(log_f - (log_q + log_M)) for log_f, log_q in zip(log_fs, log_qs)]

def sample_uniform_energy(matrix):
    mu = sum(map(mean, matrix))
    sigma = sqrt(sum(map(lambda x:variance(x,correct=False), matrix)))
    ep_min = sum(map(min, matrix))
    ep_max = sum(map(max, matrix))
    M_min = 1/norm.pdf(ep_min, mu, sigma)
    M_max = 1/norm.pdf(ep_max, mu, sigma)
    M = max(M_min, M_max)
    trials = 0
    while True:
        trials += 1
        if trials % 10000 == 0:
            print trials
        site = random_site(L)
        ep = score_seq(matrix, site)
        ar = 1/(M*norm.pdf(ep, mu, sigma))
        if random.random() < ar:
            return site

def log_ZS_gaussian((matrix, mu, Ne), trials=1000, integration='quad'):
    nu = Ne - 1
    L = len(matrix)
    mat_mu = sum(map(mean, matrix))
    mat_sigma = sqrt(sum(map(lambda x:variance(x,correct=False), matrix)))
    ep_min = sum(map(min, matrix))
    ep_max = sum(map(max, matrix))
    p = lambda x:norm.pdf(x, mat_mu, mat_sigma)
    f = lambda x: (1+exp(x-mu))**-nu
    integrand = lambda ep:dnorm(ep, mat_mu, mat_sigma) * (1+exp(ep-mu))**-nu
    log_integrand = lambda ep:log(dnorm(ep, mat_mu, mat_sigma)) + -nu*log(1+exp(ep-mu))
    if integration == 'quad':
        try:
            mean_ZS, err = integrate.quad(integrand, ep_min, ep_max,epsabs=10**-15)
        except:
            print (matrix, mue, Ne)
            raise Exception
    elif integration == 'mc':
        mean_ZS = mean(f(random.gauss(mat_mu, mat_sigma)) for _ in xrange(trials))
    elif integration == 'uniform':
        dx = (ep_max - ep_min)/trials
        mean_ZS = sum([p(x)*f(x) for x in np.linspace(ep_min, ep_max,trials)]) * dx
    elif integration == 'hack':
        mean_ZS = norm.cdf(mu - log(nu), mat_mu, mat_sigma)
    else:
        raise Exception("invalid integration method")
    if mean_ZS == 0:
        print (matrix, mu, Ne)
        raise Exception
    return L * log(4) + log(mean_ZS)

def log_ZM_gaussian(theta, N, trials=1000, integration='quad'):
    return N * log_ZS_gaussian(theta, trials=trials, integration=integration)


def test_log_ZS_gaussian(L, sigma = 1):
    """test wrt analytic, importance methods"""
    matrix = sample_matrix(L,sigma)
    mu = random.random() * 20 - 10
    Ne = random.random() * 2 + 1
    ans_analytic = log_ZS_analytic((matrix, mu, Ne))
    #ans_importance = log_ZS_importance((matrix, mu, Ne))
    ans_gaussian = log_ZS_gaussian((matrix, mu, Ne))
    #return ans_analytic, ans_importance, ans_gaussian
    return ans_analytic, ans_gaussian
    
def log_f((matrix, mu, Ne), motif):
    N = len(motif)
    nu = Ne - 1
    eps = [score_seq(matrix, site) for site in motif]
    lf = -sum(nu*log(1+exp(ep-mu)) for ep in eps)
    log_ZM = log_ZM_gaussian((matrix, mu, Ne), N)
    return lf - log_ZM

def test_log_ZM_gaussian2(L):
    """test wrt unbiased CFTP estimator"""
    pass
    
def main_experiment(generate_data=False):
    if generate_data:
        iterations = 10000
        prok_chains = [posterior_chain2(motif,iterations=iterations) for motif in tqdm(prok_motifs)]
        prok_bayes_spoofs = [[motif_from_theta(theta, len(motif)) for theta in tqdm(chain[iterations/2::500])]
                       for chain, motif in tqdm(zip(prok_chains, prok_motifs))]
        prok_psfms = [psfm_from_motif(motif, pc=1/4.0) for motif in prok_motifs]
        prok_psfm_spoofs = [[[sample_from_psfm(psfm) for _ in range(len(motif))] for _ in range(10)]
                            for psfm, motif in zip(prok_psfms, prok_motifs)]
        prok_maxent_spoofs = [spoof_maxent_motifs(motif, 10) for motif in tqdm(prok_motifs)]
        prok_apws = map(lambda m:code_from_motif(m, pc=1/16.0),tqdm(prok_motifs))
        prok_apw_spoofs = [[[sample_site(apw) for _ in range(len(motif))] for __ in range(10)]
                             for apw, motif in tqdm(zip(prok_apws,prok_motifs))]
        euk_submotifs = map(subsample, euk_motifs)
        euk_chains = [posterior_chain2(motif,iterations=iterations) for motif in tqdm(euk_submotifs)]
        euk_bayes_spoofs = [[motif_from_theta(theta, len(motif)) for theta in tqdm(chain[iterations/2::500])]
                            for chain, motif in tqdm(zip(euk_chains, euk_submotifs))]
        euk_psfms = [psfm_from_motif(motif, pc=1/4.0) for motif in euk_submotifs]
        euk_psfm_spoofs = [[[sample_from_psfm(psfm) for _ in range(len(motif))] for _ in range(10)]
                           for psfm, motif in zip(euk_psfms, euk_submotifs)]
        euk_maxent_spoofs = [spoof_maxent_motifs(motif, 10) for motif in tqdm(euk_submotifs)]
        euk_apws = map(lambda m:code_from_motif(m, pc=1/16.0),tqdm(euk_submotifs))
        euk_apw_spoofs = [[[sample_site(apw) for _ in range(len(motif))] for __ in range(10)]
                          for apw, motif in tqdm(zip(euk_apws,euk_submotifs))]
        with open("prok_chains.pkl",'w') as f:
            cPickle.dump(prok_chains,f)
        with open("prok_bayes_spoofs.pkl",'w') as f:
            cPickle.dump(prok_bayes_spoofs,f)
        with open("prok_maxent_spoofs.pkl",'w') as f:
            cPickle.dump(prok_maxent_spoofs,f)
        with open("prok_psfm_spoofs.pkl",'w') as f:
            cPickle.dump(prok_psfm_spoofs,f)
        with open("prok_apw_spoofs.pkl",'w') as f:
            cPickle.dump(prok_apw_spoofs,f)

        with open("euk_submotifs.pkl",'w') as f:
            cPickle.dump(euk_submotifs,f)
        with open("euk_chains.pkl",'w') as f:
            cPickle.dump(euk_chains,f)
        with open("euk_bayes_spoofs.pkl",'w') as f:
            cPickle.dump(euk_bayes_spoofs,f)
        with open("euk_maxent_spoofs.pkl",'w') as f:
            cPickle.dump(euk_maxent_spoofs,f)
        with open("euk_psfm_spoofs.pkl",'w') as f:
            cPickle.dump(euk_psfm_spoofs,f)
        with open("euk_apw_spoofs.pkl",'w') as f:
            cPickle.dump(euk_apw_spoofs,f)

    else:
        with open("prok_chains.pkl") as f:
            prok_chains = cPickle.load(f)
        with open("prok_bayes_spoofs.pkl") as f:
            prok_bayes_spoofs = cPickle.load(f)
        with open("prok_maxent_spoofs.pkl") as f:
            prok_maxent_spoofs = cPickle.load(f)
        with open("prok_psfm_spoofs.pkl") as f:
            prok_psfm_spoofs = cPickle.load(f)
        with open("prok_apw_spoofs.pkl") as f:
            prok_apw_spoofs = cPickle.load(f)

        with open("euk_submotifs.pkl") as f:
            euk_submotifs = cPickle.load(f)
        with open("euk_chains.pkl") as f:
            euk_chains = cPickle.load(f)
        with open("euk_bayes_spoofs.pkl") as f:
            euk_bayes_spoofs = cPickle.load(f)
        with open("euk_maxent_spoofs.pkl") as f:
            euk_maxent_spoofs = cPickle.load(f)
        with open("euk_apw_spoofs.pkl") as f:
            euk_apw_spoofs = cPickle.load(f)
        with open("euk_psfm_spoofs.pkl") as f:
            euk_psfm_spoofs = cPickle.load(f)

    #--------
    prok_ics = map(motif_ic, prok_motifs)
    prok_mis = map(mi_per_col, prok_motifs)
    prok_maxent_ics = [mean(map(motif_ic,xs)) for xs in prok_maxent_spoofs]
    prok_maxent_mis = [mean(map(mi_per_col,xs)) for xs in prok_maxent_spoofs]
    prok_psfm_ics = [mean(map(motif_ic,xs)) for xs in prok_psfm_spoofs]
    prok_psfm_mis = [mean(map(mi_per_col,xs)) for xs in tqdm(prok_psfm_spoofs)]
    prok_bayes_ics = [mean(map(motif_ic,xs)) for xs in prok_bayes_spoofs]
    prok_bayes_mis = [mean(map(mi_per_col,xs)) for xs in tqdm(prok_bayes_spoofs)]
    prok_apw_ics = [mean(map(motif_ic,xs)) for xs in prok_apw_spoofs]
    prok_apw_mis = [mean(map(mi_per_col,xs)) for xs in prok_apw_spoofs]

    prok_ics_pp = map(motif_ic_per_col, prok_motifs)
    prok_maxent_ics_pp = [mean(map(motif_ic_per_col,xs)) for xs in prok_maxent_spoofs]
    prok_psfm_ics_pp = [mean(map(motif_ic_per_col,xs)) for xs in prok_psfm_spoofs]
    prok_bayes_ics_pp = [mean(map(motif_ic_per_col,xs)) for xs in prok_bayes_spoofs]
    prok_apw_ics_pp = [mean(map(motif_ic_per_col,xs)) for xs in prok_apw_spoofs]
    

    #--------
    euk_ics = map(motif_ic, tqdm(euk_submotifs))
    euk_mis = map(mi_per_col, tqdm(euk_submotifs))
    euk_maxent_ics = [mean(map(motif_ic,xs)) for xs in tqdm(euk_maxent_spoofs)]
    euk_maxent_mis = [mean(map(mi_per_col,xs)) for xs in tqdm(euk_maxent_spoofs)]
    euk_psfm_ics = [mean(map(motif_ic,xs)) for xs in tqdm(euk_psfm_spoofs)]
    euk_psfm_mis = [mean(map(mi_per_col,xs)) for xs in tqdm(euk_psfm_spoofs)]
    euk_bayes_ics = [mean(map(motif_ic,xs)) for xs in tqdm(euk_bayes_spoofs)]
    euk_bayes_mis = [mean(map(mi_per_col,xs)) for xs in tqdm(euk_bayes_spoofs)]
    euk_apw_ics = [mean(map(motif_ic,xs)) for xs in tqdm(euk_apw_spoofs)]
    euk_apw_mis = [mean(map(mi_per_col,xs)) for xs in tqdm(euk_apw_spoofs)]

    euk_ics_pp = map(motif_ic_per_col, euk_motifs)
    euk_maxent_ics_pp = [mean(map(motif_ic_per_col,xs)) for xs in euk_maxent_spoofs]
    euk_psfm_ics_pp = [mean(map(motif_ic_per_col,xs)) for xs in euk_psfm_spoofs]
    euk_bayes_ics_pp = [mean(map(motif_ic_per_col,xs)) for xs in euk_bayes_spoofs]
    euk_apw_ics_pp = [mean(map(motif_ic_per_col,xs)) for xs in euk_apw_spoofs]



    #ic_min, ic_max, mi_min, mi_max = 4.5, 25, -0.1, 0.7
    ic_min, ic_max, mi_min, mi_max = -.1, 2.6, -0.05, 1
    #ic_xtext, ic_ytext, mi_xtext, mi_ytext = 5, 20, -0.05, 0.5
    ic_xtext, ic_ytext, mi_xtext, mi_ytext = -0.05, 2.2, -0.05, 0.85
    mi_xticks = [0, 0.25, 0.5, 0.75, 1]
    ic_yticks = [0, 0.5, 1, 1.5, 2]
    revscatter = lambda xs, ys:scatter(ys, xs)
    sns.set_style('dark')
    plt.subplot(4,4,1)
    plt.xticks([])
    #plt.yticks([])
    plt.yticks(ic_yticks, ic_yticks)
    r, p = revscatter(prok_ics_pp, prok_maxent_ics_pp)
    rmsd = sqrt(mean(zipWith(lambda x,y:(x-y)**2, prok_ics_pp, prok_maxent_ics_pp)))
    plt.xlim(ic_min, ic_max)
    plt.ylim(ic_min, ic_max)
    plt.text(ic_xtext, ic_ytext,s='$r^2$ = %1.3f' % (r**2))
    plt.text(ic_xtext, ic_ytext*0.8,s='$RMSD$ = %1.3f' % rmsd)
    plt.ylabel("MaxEnt",fontsize='large')
    plt.subplot(4,4,3)
    plt.xticks([])
    plt.yticks(mi_xticks, mi_xticks)
    plt.xlim(mi_min, mi_max)
    plt.ylim(mi_min, mi_max)
    r, p = revscatter(prok_mis, prok_maxent_mis)
    rmsd = sqrt(mean(zipWith(lambda x,y:(x-y)**2, prok_mis, prok_maxent_mis)))
    plt.text(mi_xtext, mi_ytext, s='$r^2$ = %1.3f' % (r**2))
    plt.text(mi_xtext, mi_ytext*0.8,s='$RMSD$ = %1.3f' % rmsd)
    plt.subplot(4,4,5)
    plt.xticks([])
    #plt.yticks([])
    plt.yticks(ic_yticks, ic_yticks)
    plt.xlim(ic_min, ic_max)
    plt.ylim(ic_min, ic_max)
    r, p = revscatter(prok_ics_pp, prok_psfm_ics_pp)
    rmsd = sqrt(mean(zipWith(lambda x,y:(x-y)**2, prok_ics_pp, prok_psfm_ics_pp)))
    plt.text(ic_xtext, ic_ytext,s='$r^2$ = %1.3f' % (r**2))
    plt.text(ic_xtext, ic_ytext*0.8,s='$RMSD$ = %1.3f' % rmsd)
    plt.ylabel("PSFM",fontsize='large')
    plt.subplot(4,4,7)
    plt.xticks([])
    plt.yticks(mi_xticks, mi_xticks)
    plt.xlim(mi_min, mi_max)
    plt.ylim(mi_min, mi_max)
    r, p = revscatter(prok_mis, prok_psfm_mis)
    rmsd = sqrt(mean(zipWith(lambda x,y:(x-y)**2, prok_mis, prok_psfm_mis)))
    plt.text(mi_xtext, mi_ytext, s='$r^2$ = %1.3f' % (r**2))
    plt.text(mi_xtext, mi_ytext*0.8,s='$RMSD$ = %1.3f' % rmsd)
    plt.subplot(4,4,9)
    plt.xticks([])
    #plt.yticks([])
    plt.yticks(ic_yticks, ic_yticks)
    plt.xlim(ic_min, ic_max)
    plt.ylim(ic_min, ic_max)
    r, p = revscatter(prok_ics_pp, prok_apw_ics_pp)
    rmsd = sqrt(mean(zipWith(lambda x,y:(x-y)**2, prok_ics_pp, prok_apw_ics_pp)))
    plt.text(ic_xtext, ic_ytext,s='$r^2$ = %1.3f' % (r**2))
    plt.text(ic_xtext, ic_ytext*0.8,s='$RMSD$ = %1.3f' % rmsd)
    plt.ylabel("APW",fontsize='large')
    #plt.xlabel("IC (bits)",fontsize='large')
    plt.subplot(4,4,11)
    plt.xticks([])
    plt.yticks(mi_xticks, mi_xticks)
    plt.xlim(mi_min, mi_max)
    plt.ylim(mi_min, mi_max)
    r, p = revscatter(prok_mis, prok_apw_mis)
    rmsd = sqrt(mean(zipWith(lambda x,y:(x-y)**2, prok_mis, prok_apw_mis)))
    plt.text(mi_xtext, mi_ytext, s='$r^2$ = %1.3f' % (r**2))
    plt.text(mi_xtext, mi_ytext*0.8,s='$RMSD$ = %1.3f' % rmsd)
    plt.subplot(4,4,13)
    #plt.xticks([])
    plt.yticks(ic_yticks, ic_yticks)
    plt.xticks(ic_yticks, ic_yticks)
    plt.xlim(ic_min, ic_max)
    plt.ylim(ic_min, ic_max)
    r, p = revscatter(prok_ics_pp, prok_bayes_ics_pp)
    rmsd = sqrt(mean(zipWith(lambda x,y:(x-y)**2, prok_ics_pp, prok_bayes_ics_pp)))
    plt.text(ic_xtext, ic_ytext,s='$r^2$ = %1.3f' % (r**2))
    plt.text(ic_xtext, ic_ytext*0.8,s='$RMSD$ = %1.3f' % rmsd)
    plt.xlabel("Prok IC",fontsize='large')
    plt.ylabel("Bayes",fontsize='large')
    plt.subplot(4,4,15)
    #plt.xticks([])
    plt.xticks(mi_xticks, mi_xticks)
    plt.yticks(mi_xticks, mi_xticks)
    plt.xlim(mi_min, mi_max)
    plt.ylim(mi_min, mi_max)
    r, p = revscatter(prok_mis, prok_bayes_mis)
    rmsd = sqrt(mean(zipWith(lambda x,y:(x-y)**2, prok_mis, prok_bayes_mis)))
    plt.text(mi_xtext, mi_ytext, s='$r^2$ = %1.3f' % (r**2))
    plt.text(mi_xtext, mi_ytext*0.8,s='$RMSD$ = %1.3f' % rmsd)
    plt.xlabel("Prok MI",fontsize='large')

    #--- euk plots ---#
    plt.subplot(4,4,2)
    plt.xticks([])
    plt.yticks([])
    r, p = revscatter(euk_ics_pp, euk_maxent_ics_pp)
    rmsd = sqrt(mean(zipWith(lambda x,y:(x-y)**2, euk_ics_pp, euk_maxent_ics_pp)))
    plt.xlim(ic_min, ic_max)
    plt.ylim(ic_min, ic_max)
    plt.text(ic_xtext, ic_ytext,s='$r^2$ = %1.3f' % (r**2))
    plt.text(ic_xtext, ic_ytext*0.8,s='$RMSD$ = %1.3f' % rmsd)
    #plt.ylabel("MaxEnt",fontsize='large')
    plt.subplot(4,4,4)
    plt.xticks([])
    plt.yticks([])
    plt.xlim(mi_min, mi_max)
    plt.ylim(mi_min, mi_max)
    r, p = revscatter(euk_mis, euk_maxent_mis)
    rmsd = sqrt(mean(zipWith(lambda x,y:(x-y)**2, euk_mis, euk_maxent_mis)))
    plt.text(mi_xtext, mi_ytext, s='$r^2$ = %1.3f' % (r**2))
    plt.text(mi_xtext, mi_ytext*0.8,s='$RMSD$ = %1.3f' % rmsd)
    plt.subplot(4,4,6)
    plt.xticks([])
    plt.yticks([])
    plt.xlim(ic_min, ic_max)
    plt.ylim(ic_min, ic_max)
    r, p = revscatter(euk_ics_pp, euk_psfm_ics_pp)
    rmsd = sqrt(mean(zipWith(lambda x,y:(x-y)**2, euk_ics_pp, euk_psfm_ics_pp)))
    plt.text(ic_xtext, ic_ytext,s='$r^2$ = %1.3f' % (r**2))
    plt.text(ic_xtext, ic_ytext*0.8,s='$RMSD$ = %1.3f' % rmsd)
    #plt.ylabel("PSFM",fontsize='large')
    plt.subplot(4,4,8)
    plt.xticks([])
    plt.yticks([])
    plt.xlim(mi_min, mi_max)
    plt.ylim(mi_min, mi_max)
    r, p = revscatter(euk_mis, euk_psfm_mis)
    rmsd = sqrt(mean(zipWith(lambda x,y:(x-y)**2, euk_mis, euk_psfm_mis)))
    plt.text(mi_xtext, mi_ytext, s='$r^2$ = %1.3f' % (r**2))
    plt.text(mi_xtext, mi_ytext*0.8,s='$RMSD$ = %1.3f' % rmsd)
    plt.subplot(4,4,10)
    plt.xticks([])
    plt.yticks([])
    plt.xlim(ic_min, ic_max)
    plt.ylim(ic_min, ic_max)
    r, p = revscatter(euk_ics_pp, euk_apw_ics_pp)
    rmsd = sqrt(mean(zipWith(lambda x,y:(x-y)**2, euk_ics_pp, euk_apw_ics_pp)))
    plt.text(ic_xtext, ic_ytext,s='$r^2$ = %1.3f' % (r**2))
    plt.text(ic_xtext, ic_ytext*0.8,s='$RMSD$ = %1.3f' % rmsd)
    #plt.ylabel("APW",fontsize='large')
    #plt.xlabel("IC (bits)",fontsize='large')
    plt.subplot(4,4,12)
    plt.xticks([])
    plt.yticks([])
    plt.xlim(mi_min, mi_max)
    plt.ylim(mi_min, mi_max)
    r, p = revscatter(euk_mis, euk_apw_mis)
    rmsd = sqrt(mean(zipWith(lambda x,y:(x-y)**2, euk_mis, euk_apw_mis)))
    plt.text(mi_xtext, mi_ytext, s='$r^2$ = %1.3f' % (r**2))
    plt.text(mi_xtext, mi_ytext*0.8,s='$RMSD$ = %1.3f' % rmsd)
    plt.subplot(4,4,14)
    #plt.xticks([])
    #
    plt.yticks([])
    plt.xlim(ic_min, ic_max)
    plt.ylim(ic_min, ic_max)
    r, p = revscatter(euk_ics_pp, euk_bayes_ics_pp)
    rmsd = sqrt(mean(zipWith(lambda x,y:(x-y)**2, euk_ics_pp, euk_bayes_ics_pp)))
    plt.text(ic_xtext, ic_ytext,s='$r^2$ = %1.3f' % (r**2))
    plt.text(ic_xtext, ic_ytext*0.8,s='$RMSD$ = %1.3f' % rmsd)
    #plt.ylabel("Bayes",fontsize='large')
    plt.xlabel("Euk IC",fontsize='large')
    plt.subplot(4,4,16)
    #plt.xticks([])
    plt.xticks(mi_xticks, mi_xticks)
    plt.yticks([])
    plt.xlim(mi_min, mi_max)
    plt.ylim(mi_min, mi_max)
    r, p = revscatter(euk_mis, euk_bayes_mis)
    rmsd = sqrt(mean(zipWith(lambda x,y:(x-y)**2, euk_mis, euk_bayes_mis)))
    plt.text(mi_xtext, mi_ytext, s='$r^2$ = %1.3f' % (r**2))
    plt.text(mi_xtext, mi_ytext*0.8,s='$RMSD$ = %1.3f' % rmsd)
    #plt.axis('off')
    #plt.xlabel("MI (bits/column pair)",fontsize='large')
    plt.xlabel("Euk MI",fontsize='large')
    plt.tight_layout()
    maybesave("spoof-statistics-rmsd.pdf")
    
def motif_ic_per_col(motif):
    return motif_ic(motif)/len(motif[0])
    
def log_ZS_sophisticated((matrix, mu, Ne)):
    L = len(matrix)
    nu = Ne - 1
    mat_mu = sum(map(mean,matrix))
    mat_sigma = sqrt(sum(map(lambda xs:variance(xs,correct=False), matrix)))
    dfde = lambda ep: -nu*exp(ep-mu)/(1+exp(ep-mu)) - (ep-mat_mu)/mat_sigma**2
    ep_min = sum(map(min, matrix))
    ep_max = sum(map(max, matrix))
    try:
        mode = secant_interval(dfde,ep_min - 20, ep_max + 20)
    except:
        print (matrix, mu, Ne)
        raise Exception
    kappa = -nu*(exp(mu-mode)/(1+exp(mu-mode))**2) - 1/mat_sigma**2
    sigma_approx = sqrt(-1/kappa)
    integrand = lambda ep:dnorm(ep, mat_mu, mat_sigma) * (1+exp(ep-mu))**-nu
    gauss_max = dnorm(mode, mode, sigma_approx)
    integrand_max = integrand(mode)
    mean_ZS = integrand_max / gauss_max
    return L * log(4) + log(mean_ZS)

def log_ZM_sophisticated(theta, N):
    return N * log_ZS_sophisticated(theta)

def evo_ic_sample_motif(N, L, des_ic, beta=1, theta=None, iterations=10000, verbose=False):
    """Do MH over evo param space with likelihood function proportional to IC mismatch"""
    matrix0 = [[random.gauss(0,1) for _ in range(4)] for i in range(L)]
    mu0 = -10
    Ne0 = 2
    if theta is None:
        theta = (matrix0, mu0, Ne0)
    def f(theta):
        matrix, mu, Ne = theta
        motif = sample_motif_cftp(matrix, mu, Ne, N)
        return exp(-beta*(motif_ic(motif) - des_ic)**2)
    chain = mh(f, prop2, theta, iterations=iterations, verbose=verbose, cache=False)
    return chain

def avg_ic_from_theta(theta, N, L, trials=3):
    sigma, mu, Ne = theta
    matrices = [sample_matrix(L, sigma) for i in xrange(trials)]
    motifs = [sample_motif_cftp(matrix, mu, Ne, N) for matrix in matrices]
    ics = map(motif_ic,motifs)
    mean_ic = mean(ics)
    return mean_ic
        
def evo_ic_sample_motif2(N, L, des_ic, beta=1, theta=None, iterations=10000, prop_sigma=1, trials=1, verbose=False):
    """Do MH over evo param space with likelihood function proportional to IC mismatch"""
    if theta is None:
        sigma0 = 1
        mu0 = -10
        Ne0 = 2
        theta = (sigma0, mu0, Ne0)
    def f(theta):
        sigma, mu, Ne = theta
        matrices = [sample_matrix(L, sigma) for i in xrange(trials)]
        motifs = [sample_motif_cftp(matrix, mu, Ne, N) for matrix in matrices]
        ics = map(motif_ic,motifs)
        ic = mean(ics)
        print "sigma, mu, Ne:", sigma, mu, Ne
        print "mean IC:", ic
        return exp(-beta*(ic - des_ic)**2)
    def prop(theta):
        #print "propping:", theta
        thetap = (max(0.01,theta[0] + random.gauss(0,prop_sigma)),
                  theta[1] + random.gauss(0,prop_sigma),
                  max(1,theta[2] + random.gauss(0,prop_sigma)))
        #print "thetap:", thetap
        return thetap
    chain = mh(f, prop, theta, iterations=iterations, verbose=verbose, cache=False)
    return chain

def sample_motifs_evo_ic(motif, iterations=1000, verbose=False, theta=None):
    N = len(motif)
    L = len(motif[0])
    des_ic = motif_ic(motif)
    chain = evo_ic_sample_motif2(N, L, des_ic, iterations=iterations, verbose=False, theta=theta)
    motifs = [sample_motif_cftp(sample_matrix(L, sigma), mu, Ne, N) for (sigma, mu, Ne) in tqdm(chain)]
    return chain, motifs

def predict_ic_from_theta(theta, L, num_matrices=3):
    sigma, mu, Ne = theta
    return mean(predict_ic(sample_matrix(L, sigma), mu, Ne, N=100) for _ in range(num_matrices))

def observe_ic_from_theta(theta, L, num_matrices=3):
    sigma, mu, Ne = theta
    return mean((motif_ic(sample_motif_cftp(sample_matrix(L, sigma), mu, Ne, n=100))
                         for _ in range(num_matrices)))
    
def predict_ic(matrix, mu, Ne, N=100):
    nu = Ne - 1
    ep_min, ep_max, L = sum(map(min,matrix)), sum(map(max,matrix)), len(matrix)
    site_sigma = site_sigma_from_matrix(matrix)
    density = lambda ep:(1/(1+exp(ep-mu)))**(Ne-1) * dnorm(ep,0,site_sigma)*(ep_min <= ep <= ep_max)
    d_density = lambda ep:ep/site_sigma**2 + nu/(1+exp(mu-ep))
    mode = bisect_interval(d_density, -100, 100)
    if mode < ep_min:
        mode = ep_min
    dmode = density(mode)
    # calculate mean epsilon via rejection sampling
    eps = []
    while len(eps) < N:
        ep = random.random() * (ep_max - ep_min) + ep_min
        if random.random() < density(ep)/dmode:
            eps.append(ep)
    #return eps
    des_mean_ep = mean(eps)
    des_mean_ep_analytic = integrate.quad(lambda ep:ep*density(ep), ep_min, ep_max)
    # print "des_means:", des_mean_ep, des_mean_ep_analytic
    # print "min ep: %s max_ep: %s des_mean_ep: %s" % (ep_min, ep_max, des_mean_ep)
    def mean_ep(lamb):
        try:
            psfm = psfm_from_matrix(matrix, lamb=lamb)
            return sum([ep * p for (mat_row, psfm_row) in zip(matrix, psfm)
                        for (ep, p) in zip(mat_row, psfm_row)])
        except:
            print matrix, lamb
            raise Exception
    try:
        lamb = bisect_interval(lambda l:mean_ep(l) - des_mean_ep, -20, 20)
    except:
        print matrix, mu, Ne
        raise Exception
    tilted_psfm = psfm_from_matrix(matrix, lamb)
    return sum([2 - h(col) for col in tilted_psfm])
    #psfm = [normalize(map(exp, row)) for row in matrix]
    #sites = []
    #while len(sites) < N:
    # for i in range(10000):
    #     site = random_site(L)
    #     ep = score_seq(matrix, site)
    #     if random.random() < density(ep)/dmode:
    #         sites.append(site)
    #         print i, len(sites)
    # return sites

def predict_ic_from_theta(theta, L):
    sigma, mu, Ne = theta
    nu = Ne - 1
    ep_star = mu - log(Ne - 1)
    matrix = sample_matrix(L, sigma)
    ep_min = sum(map(min, matrix))
    des_ep = max(ep_star, ep_min + 1)
    def f(lamb):
        psfm = psfm_from_matrix(matrix, lamb)
        return sum([sum(ep*p for ep,p in zip(eps, ps)) for eps, ps in zip(matrix, psfm)]) - des_ep
    log_psfm = [[log(p) for p in ps] for ps in psfm]
    lamb = bisect_interval(f,-20,20)
    sites = ([sample_from_psfm(psfm) for i in range(100)])
    log_ps = [-nu*log(1+exp(score_seq(matrix, site) - mu)) for site in sites]
    log_qs = [score_seq(log_psfm, site) for site in sites]
    
def sample_motif_ar(matrix, mu, Ne, N):
    nu = Ne - 1
    L = len(matrix)
    ep_min, ep_max, L = sum(map(min,matrix)), sum(map(max,matrix)), len(matrix)
    site_sigma = site_sigma_from_matrix(matrix)
    density = lambda ep:(1/(1+exp(ep-mu)))**(Ne-1) * dnorm(ep,0,site_sigma)*(ep_min <= ep <= ep_max)
    d_density = lambda ep:ep/site_sigma**2 + nu/(1+exp(mu-ep))
    phat = lambda ep:(1/(1+exp(ep-mu)))**(Ne-1)
    mode = bisect_interval(d_density, -100, 100)
    if mode < ep_min:
        mode = ep_min + 1 # don't want mode right on the nose of ep_min for sampling purposes, so offset it a bit
    pmode = phat(mode)
    # calculate mean epsilon via rejection sampling
    motif = []
    def mean_ep(lamb):
        psfm = psfm_from_matrix(matrix, lamb=lamb)
        return sum([ep * p for (mat_row, psfm_row) in zip(matrix, psfm)
                    for (ep, p) in zip(mat_row, psfm_row)])
    lamb = bisect_interval(lambda l:mean_ep(l) - mode, -20, 20)
    while len(motif) < N:
        site = random_site(L)
        ep = score_seq(matrix, site)
        if random.random() < phat(ep)/pmode:
            motif.append(site)    
    return motif


def sample_motif_ar_tilted(matrix, mu, Ne, N):
    nu = Ne - 1
    L = len(matrix)
    ep_min, ep_max, L = sum(map(min,matrix)), sum(map(max,matrix)), len(matrix)
    site_sigma = site_sigma_from_matrix(matrix)
    density = lambda ep:(1/(1+exp(ep-mu)))**(Ne-1) * dnorm(ep,0,site_sigma)*(ep_min <= ep <= ep_max)
    d_density = lambda ep:ep/site_sigma**2 + nu/(1+exp(mu-ep))
    phat = lambda ep:(1/(1+exp(ep-mu)))**(Ne-1)
    mode = bisect_interval(d_density, -100, 100)
    if mode < ep_min:
        mode = ep_min + 1 # don't want mode right on the nose of ep_min for sampling purposes, so offset it a bit
    dmode = density(mode)
    # calculate mean epsilon via rejection sampling
    motif = []
    def mean_ep(lamb):
        psfm = psfm_from_matrix(matrix, lamb=lamb)
        return sum([ep * p for (mat_row, psfm_row) in zip(matrix, psfm)
                    for (ep, p) in zip(mat_row, psfm_row)])
    lamb = bisect_interval(lambda l:mean_ep(l) - mode, -20, 20)
    tilted_psfm = psfm_from_matrix(matrix, lamb=lamb)
    log_tilted_psfm = [map(log,row) for row in tilted_psfm]
    while len(motif) < N:
        site = random_site(L)
        ep = score_seq(matrix, site)
        if random.random() < phat(ep)/pmode:
            motif.append(site)    
    return motif

def sample_motif_mh(matrix, mu, Ne, N, iterations=None):
    nu = Ne - 1
    L = len(matrix)
    if iterations is None:
        iterations = 10*L
    phat = lambda ep:(1/(1+exp(ep-mu)))**(nu)
    best_site = "".join(["ACGT"[argmin(col)] for col in matrix])
    def f(site):
        ep = score_seq(matrix, site)
        return phat(ep)
    def sample_site():
        return mh(f, mutate_site, best_site, iterations=10*L, verbose=0)[-1]
    return [sample_site() for i in range(N)]

def sample_site_imh(matrix, mu, Ne, lamb, iterations=None):
    nu = Ne - 1
    L = len(matrix)
    if iterations is None:
        iterations = 10*L
    log_phat = lambda site:-nu*log(1+exp(score_seq(matrix,site)-mu))
    tilted_psfm = psfm_from_matrix(matrix, lamb=lamb)
    log_tilted_psfm = [map(log,row) for row in tilted_psfm]
    def prop(_):
        return sample_from_psfm(tilted_psfm)
    def log_dprop(xp, _):
        return score_seq(log_tilted_psfm, xp)
    return mh(log_phat, proposal=prop, dprop=log_dprop, x0=prop(None), use_log=True)[-1]
    
    
def sanity_check(trials = 1000):
    L = 10
    matrix = [[-2,0,0,0] for i in range(L)]
    mu = -10
    Ne = 2
    nu = Ne - 1
    log_match_phats = [-nu * log(1+exp(-2*k - mu)) + log_choose(L,k) + k * log(1/4.0) + (L-k) * log(3/4.0)
                       for k in range(L+1)]
    match_ps = normalize(map(exp, log_match_phats))
    mh_motif = sample_motif_mh(matrix, mu, Ne, trials)
    mh_match_counts = Counter([site.count('A') for site in mh_motif])
    mh_match_ps = [mh_match_counts[k]/float(trials) for k in range(L+1)]
    cftp_motif = sample_motif_cftp(matrix, mu, Ne, trials)
    cftp_match_counts = Counter([site.count('A') for site in cftp_motif])
    cftp_match_ps = [cftp_match_counts[k]/float(trials) for k in range(L+1)]
    plt.plot(match_ps, label="Analytic")
    plt.plot(mh_match_ps, label="MH")
    plt.plot(cftp_match_ps, label="CFTP")
    plt.xlabel("Matches")
    plt.ylabel("Frequency")

    
def test_predict_ic(trials=100):
    pred_ics = []
    obs_ics = []
    for trial in trange(trials):
        sigma = random.random() * 5 + 0.1
        L = random.randrange(5, 15)
        matrix = sample_matrix(L, sigma)
        mu = random.random() * (-20)
        Ne = random.random() * 5 + 1
        pred_ic = predict_ic(matrix, mu, Ne)
        obs_ic = motif_ic(sample_motif_cftp(matrix, mu, Ne, n=100))
        pred_ics.append(pred_ic)
        obs_ics.append(obs_ic)
    r, p = scatter(pred_ics, obs_ics)
    print r, p

def test_predict_ic_from_theta(trials=100, num_matrices=10):
    pred_ics = []
    obs_ics = []
    for trial in trange(trials):
        sigma = random.random() * 5 + 0.1
        L = random.randrange(5, 15)
        mu = random.random() * (-20)
        Ne = random.random() * 5 + 1
        theta = sigma, mu, Ne
        pred_ic = predict_ic_from_theta(theta, L, num_matrices=num_matrices)
        obs_ic = observe_ic_from_theta(theta, L, num_matrices=num_matrices)
        pred_ics.append(pred_ic)
        obs_ics.append(obs_ic)
        print len(pred_ics), len(obs_ics)
    r, p = scatter(pred_ics, obs_ics)
    print r, p
    
    
def degrade(site):
    L = len(site)
    if all(b=='T' for b in site):
        return None
    dna = "ACGT"
    while True:
        i = random.randrange(L)
        b = site[i]
        if b == 'T':
            continue
        else:
            return subst(site,dna[dna.index(b)+1],i)
            
def eps_from_theta(theta, L, N=100):
    matrix = sample_matrix(L, sigma)
    motif = sample_motif_cftp(matrix, mu, Ne, N)
    eps = [score_seq(matrix, site) for site in motif]
    return eps
    
def resample_from_post_chain(chain, N):
    """given chain of the form [(mat, mu, Ne)], perform reduction:
    mat -> sigma -> mat' -> motif'

    Conclusion: heavily underestimates IC.
    """
    L = len(chain[0][0])
    sigmas = [sigma_from_matrix(mat) for (mat, mu, Ne) in chain]
    matrices = [sample_matrix(L, sigma) for sigma in sigmas]
    motifs = [sample_motif_cftp(matrix, mu, Ne, N) for matrix, (_, mu, Ne) in tqdm(zip(matrices, chain))]
    return motifs
