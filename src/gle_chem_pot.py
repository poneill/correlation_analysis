"""Implement Gaussian Linear Model with chemical potential"""

from linear_gaussian_ensemble import sample_matrix
from utils import score_seq, dnorm

# lower mu means less protein.

def log_fitness(matrix, motif, mu):
    eps = [score_seq(matrix,site) for site in motif]
    return sum(-log(1+exp(ep-mu)) for ep in eps)
    
def sella_hirsch_mh(Ne=5, n=16, L=16, sigma=1, mu=0, init="random", 
                                             matrix=None, x0=None, iterations=50000, p=None):
    print "p:", p
    if matrix is None:
        matrix = sample_matrix(L, sigma)
    if x0 is None:
        if init == "random":
            x0 = random_motif(L, n)
        elif init == "ringer":
            x0 = ringer_motif(matrix, n)
        elif init == "anti_ringer":
            x0 = anti_ringer_motif(matrix, n)
        else:
            x0 = init
    if p is None:
        p = 1.0/(n*L)
    nu = Ne - 1
    def log_f(motif):
        return nu * log_fitness(matrix, motif, mu)
    def prop(motif):
        motif_p = mutate_motif_p(motif, p) # probability of mutation per basepair
        return motif_p
    chain = mh(log_f, prop, x0, use_log=True, iterations=iterations)
    return matrix, chain

def sample_site(matrix,n,Ne):
    """sample site from distribution P(s) ~= exp(nu*log(f)) = (1+exp(E(s)-mu))^(-v)/Z"""
################################################################################
# Mu penalization ideas
################################################################################

def log_fitness_penalize_mu(matrix, mut, mu, alpha, G):
    """Assume a penalty for tf production in form  of fitness penalty for mu"""
    log_f = log_fitness(matrix, motif, mu, G)
    # given log(a), log(b), log(a+b) = log(a(1+b/a)) = log(a) + log(1+exp(log(b)-log(a)))
    return log_f - alpha*mu

def sella_hirsch_mh_penalize_mu(Ne=5, n=16, L=16, G=5*10**6, sigma=1, alpha=0.01, init="random", 
                                             matrix=None, x0=None, iterations=50000, p=None):
    print "p:", p
    if matrix is None:
        matrix = sample_matrix(L, sigma)
    if x0 is None:
        if init == "random":
            x0 = (random_motif(L, n),random.gauss(0,1))
        elif init == "ringer":
            x0 = (ringer_motif(matrix, n),random.gauss(0,1))
        elif init == "anti_ringer":
            x0 = (anti_ringer_motif(matrix, n), random.gauss(0,1))
        else:
            x0 = init
    if p is None:
        p = 1.0/(n*L)
    nu = Ne - 1
    def log_f((motif, mu)):
        return nu * log_fitness_penalize_mu(matrix, motif, mu, alpha, G)
    def prop((motif, mu)):
        motif_p = mutate_motif_p(motif, p) # probability of mutation per basepair
        mu_p = mu + random.gauss(0,0.01)
        return motif_p, mu_p
    chain = mh(log_f, prop, x0, use_log=True, iterations=iterations)
    return matrix, chain

def predict_modal_energy(site_sigma,mu,Ne):
    nu = Ne - 1
    dlogPe_de = lambda ep:-nu*exp(ep-mu)/(1+exp(ep-mu)) - ep/site_sigma**2
    return secant_interval(dlogPe_de,-50,50)

def modal_energy_curvature(site_sigma,mu,Ne):
    nu = Ne - 1
    ep_star = predict_modal_energy(site_sigma,nu,Ne)
    dlogPe2_de2 = lambda ep:-(1/site_sigma**2) - nu*(exp(ep-mu)/(exp(ep-mu)+1) + exp(2*(ep-mu))/(exp(ep-mu)+1)**2)
    curvature = dlogPe2_de2(ep_star)
    return curvature
    
                                          
def gaussian_hist(xs,sigma=1):
    def f(xp):
        return mean(dnorm(xp,mu=x,sigma=sigma) for x in xs)
    return f
