"""
Define induced Boltzmann density on simplex from normally
distributed energy levels 
"""

from utils import simplex_sample,mean,transpose,fac
from math import exp,log,sqrt,pi
import random
from scipy.stats import norm,pearsonr
from tqdm import *
import numpy as np

def bltz(eps):
    ws = [exp(-ep) for ep in eps]
    Z = sum(ws)
    return [w/Z for w in ws]

def canonical_energies(ps):
    """undo boltzmann mapping"""
    return [-log(p) for p in ps]

def sample_epsilons(n,sigma):
    return [random.gauss(0,sigma) for i in range(n)]

def sample(n,sigma):
    return bltz(sample_epsilons(n,sigma))

def gaussian_integrate(f,n,sigma,trials):
    return mean(f(sample_epsilons(n,sigma)) for i in range(trials))

def simplex_integrate_ref(f,n,sigma,trials=1000):
    return mean(f(sample(n,sigma)) for i in xrange(trials))

def simplex_integrate(f,n,sigma,trials=1000):
    integrand = lambda ps:f(ps)*logit_measure(ps,sigma)/fac(n-1)
    return mean(integrand(simplex_sample(n)) for i in xrange(trials))

def sample_alt(n,sigma):
    """sample according to logit-normal distribution article"""
    ys = [random.gauss(0,sigma) for i in range(n-1)]
    Z = 1 + sum(exp(y) for y in ys)
    return [exp(y)/Z for y in ys] + [1/Z]

def entropy(ps):
    return -sum(p*log(p) for p in ps)

def prod(xs):
    return reduce(lambda x,y:x*y,xs)

def log_measure_ref(xs,sigma):
    """Compute probability mass assigned to preimage under boltzmann
    mapping"""
    N = float(len(xs))
    ys = canonical_energies(eps)
    A = sum(ys)**2
    B = sum(y**2 for y in ys)
    log_prefactor = log(sigma*sqrt(2*pi/N)*(1/(sigma*sqrt(2*pi))**N))
    log_weight = (-B + A/N)/(2*sigma**2)
    return log_prefactor + log_weight

def log_measure(xs,sigma):
    """Compute probability mass assigned to preimage under boltzmann
    mapping"""
    N = float(len(xs))
    ys = canonical_energies(xs)
    A = sum(ys)
    B = sum(y**2 for y in ys)
    log_prefactor = log(1/(sqrt(N)*(sigma*sqrt(2*pi))**(N-1)))
    log_weight = (A**2/N-B)/(2*sigma**2)
    return log_prefactor + log_weight

def measure(xs,sigma):
    return exp(log_measure(xs,sigma))

def measure2(xs,sigma):
    eps = [-log(x) for x in xs]
    return 1/det_jacobian(eps)*measure(xs,sigma)

def measure3(xs,sigma):
    #n = len(xs)
    return (1/prod(xs))*prod(norm.pdf(-log(x),0,sigma) for x in xs)

def norm_by_last(eps):
    last = float(eps[-1])
    return [ep - last for ep in eps]

def logit(eps):
    _eps = norm_by_last(eps)
    ws = [exp(_ep) for _ep in _eps]
    Z = sum(ws)
    return [w/Z for w in ws]

def logit_np1(eps):
    return logit(eps + [0])

def sample_logit(n,sigma):
    factor = 1#1/sqrt(2)
    eps = sample_epsilons(n,factor*sigma)
    return logit(eps)

def sample_logit_ref(n,sigma):
    x = random.gauss(0,sigma)
    eps = [x + ep for ep in sample_epsilons(n-1,sigma)] + [0]
    return logit(eps)

def logit_normal_pdf(xs,mu=0,Sigma=None):
    n = len(xs)
    if Sigma == None:
        Sigma = np.eye(n-1)
    xD = xs[-1]
    _xs = np.array([log(x/xD) for x in xs[:-1]])
    prefactor = 1/(sqrt(np.linalg.det(2*pi*Sigma)))#*(1/fac(n-1))
    weight = (1.0/prod(xs))*exp(-1/2.0*(_xs-mu).dot(np.linalg.inv(Sigma)).dot(_xs-mu))
    return prefactor*weight

def sample_multivariate_normal(mus,Sigma):
    return np.random.multivariate_normal(mus,Sigma)

def logit_measure(xs,sigma):
    n = len(xs)
    Sigma = logit_cov_matrix(n-1,sigma)
    return logit_normal_pdf(xs,mu=0,Sigma=Sigma)

def test_commutativity():
    """test commutativity of boltzmann mapping with additive shift"""
    pass

def cov_matrix_dep(n,sigma):
    def a(i,j):
        if i == j:
            return 1
        elif i == n-1 or j == n-1:
            return 1/sqrt(2)
        else:
            return 1/2.0
    Sigma = sigma**2*np.matrix([[a(i,j) for i in range(n)]
                                for j in range(n)])
    return Sigma

def logit_cov_matrix(n,sigma):
    Sigma = sigma**2*(np.eye(n) + np.ones((n,n)))
    return Sigma
    
def cov(xs,ys):
    mu_x = mean(xs)
    mu_y = mean(ys)
    return mean((x-mu_x)*(y-mu_y) for (x,y) in zip(xs,ys))

def get_cov_matrix(xss):
    cols = transpose(xss)
    k = len(cols)
    return [[cov(cols[i],cols[j]) for i in range(k)] for j in range(k)]

def measure_sanity(xs,sigma,trials=1000):
    """Monte Carlo estimate of measure by weighted sampling"""
    ys = canonical_energies(xs)
    # sigma of sampling points; need not equal sigma but most efficient
    test_sigma = sigma 
    f = lambda s:prod([norm.pdf(s,y,sigma) for y in ys])
    ss = [random.gauss(0,test_sigma) for i in range(trials)]
    return mean(f(s)/norm.pdf(s,0,test_sigma) for s in ss)
    
def test_measure_sanity(sigma=1,N=3,test_trials=100,trials=1000):
    ms = []
    ms_sanity = []
    for i in trange(test_trials):
        xs = simplex_sample(N)
        ms.append(measure(xs,sigma))
        ms_sanity.append(measure_sanity(xs,sigma,trials))
    plt.scatter(ms,ms_sanity)
    plt.plot([0,1],[0,1])
    print pearsonr(ms,ms_sanity)
    plt.show()
    
def sample_measure(n,sigma):
    while True:
        xs = simplex_sample(n)
        p = exp(log_measure(xs,sigma))
        if p > 1:
            raise Exception("p > 1")
        if random.random() < p:
            return xs

def sample_measure2(n,sigma):
    f = lambda xs:measure2(xs,sigma)
    x0 = simplex_sample(n)
    prop=lambda xs:simplex_sample(n)
    chain = mh(f,prop,x0)
    return chain

def sample_measure_sanity(n,sigma):
    while True:
        xs = simplex_sample(n)
        p = exp(log_measure(xs,sigma))
        if p > 1:
            raise Exception("p > 1")
        if random.random() < p/2:
            return xs

def measure_test3_integrate_to_one(sigma=1,N=3,trials=1000):
    """check to see that measure integrates to 1"""
    return mean(measure(simplex_sample(N),1) for i in trange(trials))

def measure_test2(N=3,sigma=1,trials=1000000):
    lms = [log_measure(sample(N,sigma),sigma) for i in trange(trials)]
    return lms

def measure_test():
    N = 3
    trials = 10
    max_sigma = 100
    replicates = 1000
    sigmas = [random.random()*max_sigma for i in range(trials)]
    xss = [[sample(N,sigma) for r in range(replicates)]
           for sigma in sigmas]
    for sigma,xs in zip(sigmas,xss):
        lls = [sum(log_measure(rep,s) for rep in xs) for s in sigmas]
        print max(range(trials),key=lambda i:lls[i])

def kde(xs,sigma):
    return lambda y:mean(norm.pdf(y,x,sigma) for x in xs)

def empirical_measure(samples,sigma):
    components = transpose(samples)
    fs = [kde(comp,sigma) for comp in components]
    return lambda xs:prod(f(x) for f,x in zip(fs,xs))

def is_original_idea_correct():
    """should see striations if integral is one-dimensional"""
    xs = np.linspace(-10,10,1000)
    ys = np.linspace(-10,10,1000)
    M = np.matrix([[exp(-x)/(exp(-x)+exp(-y)) for x in xs] for y in ys])
    plt.imshow(M,interpolation='none')
    plt.show()

def det_jacobian(eps):
    """return determinant of jacobian dpi/depj"""
    # f(ep) = exp(-ep)
    # dfdep = -exp(-ep)
    # d2fdep2 = exp(-ep)
    # 1/det(J) = 1/prod(exp(-ep)) = 1/exp(-sum(eps))
    return exp(-sum(eps))
