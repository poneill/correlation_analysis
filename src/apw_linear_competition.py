"""Which models fare better under mutation, linear or adjacent pairwise?"""
from utils import score_seq, mutate_site, random_site, scatter, mean, mh
from math import log, exp
from adjacent_pairwise_model import score, sample_code, linearize
from pwm_utils import sample_matrix
import numpy as np
from tqdm import *

def experiment1_():
    L = 10
    sigma = 1
    code = sample_code(L, 1)
    mu = -10
    Ne = 2
    pssm = linearize(code)
    def apw_phat(site):
        ep = score(code, site)
        return 1/(1+exp(ep-mu))**(Ne-1)
    def linear_phat(site):
        ep = score_seq(pssm, site)
        return 1/(1+exp(ep-mu))**(Ne-1)
    def sample_apw_site():
        return mh(apw_phat, proposal=mutate_site, x0=random_site(L))
    apw_chain = mh(apw_phat, proposal=mutate_site, x0=random_site(L))
    linear_chain = mh(linear_phat, proposal=mutate_site, x0=random_site(L))
    apw_fits = map(apw_phat, apw_chain)
    linear_fits = map(linear_phat, linear_chain)
    return apw_fits, linear_fits
    
def experiment1():
    """APW models win, presumably because they have larger sigmas"""
    scatter(*transpose([map(lambda xs:mean(map(log,xs)),experiment1_()) for i in trange(100)]))

def experiment2_():
    L = 10
    sigma = 1
    code = sample_code(L, 1)
    mu = -10
    Ne = 2
    sites = [random_site(L) for i in xrange(10000)]
    apw_eps = [score(code, site) for site in sites]
    site_sigma = sd(apw_eps)
    pssm = sample_matrix(L, sqrt(site_sigma**2/L))
    #linear_eps = [score_seq(pssm, site) for site in sites]
    def apw_phat(site):
        ep = score(code, site)
        return 1/(1+exp(ep-mu))**(Ne-1)
    def linear_phat(site):
        ep = score_seq(pssm, site)
        return 1/(1+exp(ep-mu))**(Ne-1)
    def sample_apw_site():
        return mh(apw_phat, proposal=mutate_site, x0=random_site(L))
    apw_chain = mh(apw_phat, proposal=mutate_site, x0=random_site(L))
    linear_chain = mh(linear_phat, proposal=mutate_site, x0=random_site(L))
    apw_fits = map(apw_phat, apw_chain)
    linear_fits = map(linear_phat, linear_chain)
    return apw_fits, linear_fits

def experiment2(trials=100):
    """APW models win, presumably because they have larger sigmas"""
    scatter(*transpose([map(lambda xs:mean(map(log,xs)),experiment2_()) for i in trange(trials)]))

log10 = lambda x:log(x,10)

def experiment3(trials=10):
    mu = -10
    Ne = 5
    L = 10
    sigma = 1
    codes = [sample_code(L, sigma) for i in range(trials)]
    pssms = [sample_matrix(L, sigma) for i in range(trials)]
    sites = [random_site(L) for i in xrange(10000)]
    apw_site_sigmas = [sd([score(code,site) for site in sites]) for code in codes]
    linear_site_sigmas = [sd([score_seq(pssm,site) for site in sites]) for pssm in pssms]
    def apw_phat(code, site):
        ep = score(code, site)
        return 1/(1+exp(ep-mu))**(Ne-1)
    def apw_occ(code, site):
        ep = score(code, site)
        return 1/(1+exp(ep-mu))
    def linear_phat(pssm, site):
        ep = score_seq(pssm, site)
        return 1/(1+exp(ep-mu))**(Ne-1)
    def linear_occ(pssm, site):
        ep = score_seq(pssm, site)
        return 1/(1+exp(ep-mu))
    apw_mean_fits = [exp(mean(map(log10, mh(lambda s:apw_phat(code, s), proposal=mutate_site, x0=random_site(L),
                                          capture_state = lambda s:apw_occ(code, s))[1:])))
                         for code in tqdm(codes)]
    linear_mean_fits = [exp(mean(map(log10, mh(lambda s:linear_phat(pssm, s), proposal=mutate_site, x0=random_site(L),
                                             capture_state = lambda s:linear_occ(pssm, s))[1:])))
                        for pssm in tqdm(pssms)]
    plt.scatter(apw_site_sigmas, apw_mean_fits, label='apw')
    plt.scatter(linear_site_sigmas, linear_mean_fits, color='g',label='linear')
    plt.semilogy()
    plt.legend(loc='lower right')

def experiment4(L=10):
    """do grid search to determine whether linear or apw models fair better under different regimes"""
    def apw_fit(sigma, mu, Ne):
        code = sample_code(L, sigma)
        def apw_phat(site):
            ep = score(code, site)
            return 1/(1+exp(ep-mu))**(Ne-1)
        chain = mh(lambda s:apw_phat(s), proposal=mutate_site, x0=random_site(L),
                   capture_state = lambda s:apw_occ(code, mu, s))[25000:]
        return mean(chain)
    def linear_fit(sigma, mu, Ne):
        pssm = sample_matrix(L, sigma)
        def linear_phat(site):
            ep = score_seq(pssm, site)
            return 1/(1+exp(ep-mu))**(Ne-1)
        chain = mh(lambda s:linear_phat(s), proposal=mutate_site, x0=random_site(L),
                   capture_state = lambda s:linear_occ(pssm, mu, s))[25000:]
        return mean(chain)
    def apw_occ(code, mu, site):
        ep = score(code, site)
        return 1/(1+exp(ep-mu))
    def linear_occ(pssm, mu, site):
        ep = score_seq(pssm, site)
        return 1/(1+exp(ep-mu))
    sigmas = np.linspace(0,5,5)
    mus = np.linspace(-10,10,5)
    Nes = np.linspace(0,5,5)
    apws = [apw_fit(sigma, mu, Ne) for sigma in tqdm(sigmas) for mu in mus for Ne in Nes]
    linears = [linear_fit(sigma, mu, Ne) for sigma in tqdm(sigmas) for mu in mus for Ne in Nes]
    
        
