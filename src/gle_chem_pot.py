"""Implement Gaussian Linear Model with chemical potential"""

from linear_gaussian_ensemble import  ringer_motif
from linear_gaussian_ensemble_gini_analysis import mutate_motif_k_times
from pwm_utils import matrix_from_motif, sample_matrix, sigma_from_matrix, sample_matrix, Zb_from_matrix
from utils import score_seq, dnorm, mean, random_motif, mh, mutate_motif_p, motif_ic, variance, mmap
from utils import sorted_indices, rslice, secant_interval, pairs, sd, random_site, mean, motif_gini, total_motif_mi
from utils import mean_ci, bisect_interval, concat, pred_obs, maybesave
from math import exp, log, sqrt, pi
from scipy.stats import norm, pearsonr
from scipy import polyfit, poly1d
from scipy.optimize import leastsq
import numpy as np
from tqdm import *
from matplotlib import pyplot as plt
import random
from motifs import Escherichia_coli
#from parse_tfbs_data import tfdf
import seaborn as sns
# lower mu means less protein.

G = 5 * 10**6

def log_fitness(matrix, motif, mu):
    eps = [score_seq(matrix,site) for site in motif]
    return -sum(log(1+exp(ep-mu)) for ep in eps)
    
def sella_hirsch_mh(Ne=5, n=16, L=16, sigma=1, mu=0, init="random", 
                                             matrix=None, x0=None, iterations=50000, p=None):
    print "p:", p
    if matrix is None:
        matrix = sample_matrix(L, sigma)
    else:
        L = len(matrix)
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

################################################################################
# Mu penalization ideas
################################################################################

def log_fitness_penalize_mu(matrix, motif, mu, alpha):
    """Assume a penalty for tf production in form  of fitness penalty for mu"""
    log_f = log_fitness(matrix, motif, mu)
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
        return nu * log_fitness_penalize_mu(matrix, motif, mu, alpha)
    def prop((motif, mu)):
        motif_p = mutate_motif_p(motif, p) # probability of mutation per basepair
        mu_p = mu + random.gauss(0,0.1)
        return motif_p, mu_p
    chain = mh(log_f, prop, x0, use_log=True, iterations=iterations)
    return matrix, chain

def sample_motif_from_energies(matrix,n,mu,Ne):
    nu = Ne - 1
    site_sigma = site_sigma_from_matrix
    
    
    
def predict_modal_energy(site_mu, site_sigma,mu,Ne):
    nu = Ne - 1
    dlogPe_de = lambda ep:-nu*exp(ep-mu)/(1+exp(ep-mu)) - (ep-site_mu)/site_sigma**2
    return secant_interval(dlogPe_de,-50,50)

def modal_energy_gaussian_params(site_mu, site_sigma, mu, Ne):
    """"""
    nu = Ne - 1
    ep_star = predict_modal_energy(site_mu, site_sigma,mu,Ne)
    dlogPe2_de2 = lambda ep:-(1/site_sigma**2) - nu*exp(mu-ep)/(exp(mu-ep)+1)**2
    curvature = dlogPe2_de2(ep_star)
    # gaussian curvature at mean = -1/(sqrt(2*pi)*sigma**3)
    sigma = sqrt(-1/curvature)
    return ep_star,sigma

def gaussian_params2(site_mu, site_sigma, mu, Ne):
    m = mean_epsilon(site_mu, site_sigma, mu, Ne)
    s = sqrt(var_epsilon(site_mu, site_sigma, mu, Ne))
    return m, s
    
def Pe(ep,site_mu,site_sigma,mu,Ne):
    nu = Ne - 1
    Z = norm.cdf(mu - log(nu),site_mu,site_sigma)
    return 1/Z*(1/(1+exp(ep-mu))**nu)*dnorm(ep,site_mu,site_sigma)

def mean_epsilon(site_mu,site_sigma,mu,Ne):
    nu = Ne - 1
    return norm.cdf(mu - log(nu),site_mu,site_sigma)

def var_epsilon(site_mu,site_sigma,mu,Ne):
    nu = Ne - 1
    E_ep_sq = norm.cdf(mu - log(2*nu),site_mu,site_sigma)
    E_ep = norm.cdf(mu - log(nu),site_mu,site_sigma)
    return E_ep_sq - E_ep**2
    
def log_Pe(ep,site_mu,site_sigma,mu,Ne):
    """return log probability of binding energy up to constant"""
    nu = Ne - 1
    # Pe(ep) =
    # N(e;m,s^2)*(1/(1+exp(ep-mu)))^nu/Z(site_mu,site_sigma,mu,nu), so
    # normalization depends on site_mu, site_sigma, mu, nu
    Z = norm.cdf(mu - log(nu),site_mu,site_sigma)
    return -log(Z) - nu*log(1+exp(ep-mu)) - (log(site_sigma) + log(sqrt(2*pi))) - (ep-site_mu)**2/(2*site_sigma**2)

def site_mh(matrix,mu,Ne,iterations=50000):
    site_mu, site_sigma = site_mu_from_matrix(matrix), site_sigma_from_matrix(matrix)
    L = len(matrix)
    nu = Ne - 1
    log_f = lambda site:log_Pe(score_seq(matrix,site),site_mu,site_sigma,mu,Ne)
    #prop = lambda site:random_site(L)
    prop = lambda site:mutate_site(site)
    return mh(log_f,prop,x0=random_site(L),use_log=True,iterations=iterations)

def kde(xs,sigma=1):
    def f(xp):
        return mean(dnorm(xp,mu=x,sigma=sigma) for x in xs)
    return f

def kde_interpolate(xs,ys,sigma=1):
    def interp(xstar):
        numer = sum(y*dnorm(xstar,x,sigma) for x,y in zip(xs,ys))
        denom = sum(dnorm(xstar,x,sigma) for x in xs)
        return numer/denom
    return interp

def linear_interpolate(xs,ys):
    js = sorted_indices(xs)
    xs = sorted(xs)
    ys = rslice(ys,js)

def bisect_interval_noisy(f,epsilon=0.01,sigma=None,debug=False):
    """find zero of stochastic function f using linear regression"""
    print "in bisect"
    xmin = 1
    xmax = 2
    xs = [xmin,xmax]
    print xmin,xmax
    print f(1)
    ys = map(f,xs)
    print ys
    print "ys[-1]:",ys[-1]
    while ys[-1] < 0:
        xmax += 1
        xs.append(xmax)
        y = f(xmax)
        ys.append(y)
    xs2 = [x + xs[-1] for x in xs]
    ys2 = map(f,xs2)
    xs = xs + xs2
    ys = ys + ys2
    #xs = list(np.linspace(lb,ub,10))
    #ys = map(f,xs)
    print "xs,ys:",xs,ys
    i = 1
    while sd(xs[-3:]) > epsilon:
        print "starting round",i
        i += 1
        ### select xp
        # m = (y2-y1)/float(x2-x1)
        # xp = -y1/m + x1
        # yp = f(xp)
        if sigma is None:
            print "interpolating on:",xs,ys
            r = kde_interpolate(xs,ys,sigma=sd(xs)/3.0)
        else:
            r = kde_interpolate(xs,ys,sigma=sigma)
        try:
            xp = bisect_interval(r,min(xs),max(xs))
            print "selected xp:",xp
        except:
            "secant regression failed!"
            Exception()
        if debug:
            plt.scatter(xs,ys)
            plt.plot(*pl(r,np.linspace(min(xs),max(xs),1000)))
            plt.plot([xp,xp],[-10,10])
            plt.plot([min(xs),max(xs)],[0,0])
            plt.show()
        yp = f(xp)
        ### end select xp
        print "xp,yp:",xp,yp
        xs.append(xp)
        ys.append(yp)
        #js = sorted_indices(xs)
        #xs = rslice(xs,js)
        #ys = rslice(ys,js)
        #assert xs == sorted(xs)
    return xp,(xs,ys)

def robbins_monro(f,x0,trials=10,debug=False):
    """use Robbins-Monro iteration to find root of stochastic function f"""
    x = x0
    for i in range(1,trials + 1):
        ai = 1.0/i
        x = x - ai*f(x)
        if debug:
            print i,x
    return x
    
def stoch_lin_reg(f,lb=1,ub=50,stop_sd=0.1):
    xs = [lb,ub]
    ys = map(f,xs)
    while True:
        m, b = polyfit(xs,ys,1)
        x = -b/float(m)
        if x < 1:
            x = random.random()*(max(xs)-min(xs)) + min(xs)
        y = f(x)
        xs.append(x)
        ys.append(y)
        if sd(xs[-3:]) < stop_sd:
            break
    # end with one more round of interpolation
    m, b = polyfit(xs,ys,1)
    x = -b/float(m)
    y = f(x)
    return x,(xs,ys)

def stoch_logistic_reg(f,xmin,xmax,ymax,desired_ic,stop_sd=0.01,debug=False):
    ymin = 0
    def sigmoid(params,x):
        x0,k,ymax =params
        y = ymax / (1 + np.exp(-k*(x-x0))) + ymin
        return y
    def residuals(params,x,y):
        return y - sigmoid(params,x)
    xs = list(np.linspace(xmin,xmax,10))
    ys = map(f,xs)
    params = (2,1,desired_ic*2)
    while True:
        p_guess = params
        params, cov, infodict, mesg, ier = leastsq(
            residuals,p_guess,args=(xs,ys),full_output=1)
        try:
            x = secant_interval(lambda x:sigmoid(params,x)-desired_ic,xmin,xmax)
        except:
            print "failed secant interval"
            print params, xs,ys
            raise Exception()
        y = f(x)
        xs.append(x)
        ys.append(y)
        if sd(xs[-3:]) < stop_sd:
            break
        if debug:
            plt.scatter(xs,ys)
            plt.plot(*pl(lambda x:sigmoid(params,x),np.linspace(xmin,xmax,1000)))
            plt.plot(*pl(lambda x:desired_ic,np.linspace(xmin,xmax,1000)))
            plt.plot([x,x],[0,2*L])
            plt.show()
    # end with one more round of interpolation
    params, cov, infodict, mesg, ier = leastsq(
            residuals,p_guess,args=(xs,ys),full_output=1)
    x = secant_interval(lambda x:sigmoid(params,x)-desired_ic,xmin,xmax)
    return x,(xs,ys)
    
    
def site_mu_from_matrix(matrix):
    return sum(map(mean,matrix))
    
def site_sigma_from_matrix(matrix):
    """return sd of site energies from matrix"""
    return sqrt(sum(map(lambda xs:variance(xs,correct=False),matrix)))

def sigma_from_matrix(matrix):
    """given a GLE matrix, estimate standard deviation of cell weights,
    correcting for bias of sd estimate.  See:
    https://en.wikipedia.org/wiki/Unbiased_estimation_of_standard_deviation
    """
    c = 2*sqrt(2/(3*pi))
    return mean(map(lambda x:sd(x,correct=True),matrix))/c
    
def spoof_motif(motif,Ne=None,iterations=10000):
    matrix = matrix_from_motif(motif)
    L = len(motif[0])
    n = len(motif)
    sigma = sigma_from_matrix(matrix)
    spoof_matrix = sample_matrix(L,sigma)
    bio_ic = motif_ic(motif)
    # this method of reading site_mu, site_sigma off of motif is slightly suspect...
    site_mu = site_mu_from_matrix(matrix_from_motif(motif))
    site_sigma = site_sigma_from_matrix(matrix_from_motif(motif))
    # now need to find mu, nu
    n = len(motif)
    assumed_copies = 10*n
    mu = approximate_mu(matrix,assumed_copies,G)
    spoof_mu = approximate_mu(spoof_matrix,assumed_copies,G)
    if Ne is None:
        Ne = estimate_Ne(spoof_matrix,spoof_mu,n,bio_ic)
        print "chose Ne:",Ne
    spoof_matrix, chain = sella_hirsch_mh(Ne=Ne,matrix=spoof_matrix,mu=mu,n=n)
    return spoof_matrix, chain, Ne

def estimate_Ne(matrix,mu,n,desired_ic,iterations=10000):
    def f(Ne):
        print "calling f with Ne:",Ne
        #_,chain = sella_hirsch_mh(Ne=Ne,mu=mu,n=n,matrix=spoof_matrix,init='ringer',iterations=iterations)
        _,chain = sella_hirsch_mh(Ne=Ne,mu=mu,n=1,matrix=matrix,init='ringer',iterations=iterations)
        return mean(map(motif_ic,[[random.choice(chain[iterations/2:])[0] for i in range(n)]
                                  for j in range(100)])) - desired_ic
    Ne, (xs,ys) = bisect_interval_noisy(f)
    return Ne
    
def mean_truncated_normal(m,s,a,b):
    s = float(s)
    phi = norm.pdf
    Phi = norm.cdf
    alpha = (a - m)/s
    beta = (b - m)/s
    return m - s * ((phi(beta,0,1) - phi(alpha,0,1))/(Phi(beta,0,1) - Phi(alpha,0,1)))

def variance_truncated_normal(m, s, a, b):
    s = float(s)
    phi = norm.pdf
    Phi = norm.cdf
    alpha = (a - m)/s
    beta = (b - m)/s
    return s**2 * (1 - (beta*phi(beta,0,1) - alpha*phi(alpha,0,1))/(Phi(beta,0,1) - Phi(alpha,0,1))
                   - ((phi(beta,0,1) - phi(alpha,0,1))/(Phi(beta,0,1) - Phi(alpha,0,1)))**2)
    
def test_mean_truncated_normal(m,s,a,b,trials=10000):
    xs = filter(lambda x:a<x<b,[random.gauss(m,s) for i in trange(trials)])
    if not xs:
        return None
    else:
        return mean(xs)

def L_sigma_plot(mu=-10):
    def occupancy(matrix):
        site = ringer_motif(matrix,1)[0]
        ep = score_seq(matrix,site)
        return 1/(1+exp(ep-mu))
    Ls = range(1,30)
    sigmas = np.linspace(0,20,100)
    occ_matrix = [[mean(occupancy(sample_matrix(L,sigma)) for i in range(10))
                   for L in Ls]
                  for sigma in tqdm(sigmas)]
    pred_matrix = [[1/(1+exp(-L*sigma-mu)) for L in Ls] for sigma in sigmas]
    plt.subplot(1,2,1)
    plt.imshow(occ_matrix,interpolation='none',aspect='auto')
    plt.subplot(1,2,2)
    plt.imshow(pred_matrix,interpolation='none',aspect='auto')
    plt.colorbar()
        
def penalize_mu_optimal_params(site_mu, site_sigma,alpha,n,Ne):
    if n/float(alpha) - 1 <= 0:
        return None,None
    nu = Ne - 1
    ep_star = -nu*alpha*site_sigma**2/float(n)
    mu_star = log(n/float(alpha) - 1) + ep_star
    return ep_star, mu_star

def critical_alpha(site_sigma,n,Ne):
    """return value of alpha that zeros d(mu_star)/d(alpha)"""
    nu = Ne - 1
    sqt = sqrt(1+4/float(nu*site_sigma**2))
    return n*(1+sqt)/2.0,n*(1-sqt)/2.0

def epsilon_u_shaped_experiment(trials = 1000):
    site_mu = 0
    site_sigma = 1
    lb, ub = -10*site_sigma,10*site_sigma
    space = np.linspace(lb,ub,trials)
    nu = 1
    def P(ep,mu,alpha):
        return (1/(1+exp(ep-mu))*exp(-alpha*mu))**nu*dnorm(ep,site_mu,site_sigma)
    eps = space
    mus = space
    eps_mus = [(ep,mu) for ep in eps for mu in mus]
    ps = [P(ep,mu,alpha) for (ep,mu) in eps_mus]
    print "sum(ps):",sum(ps)
    def mean_ep(alpha):
        return sum(ep*p for (ep,mu),p in zip(eps_mus,ps))/sum(ps)
    def mean_mu(alpha):
        return sum(mu*p for (ep,mu),p in zip(eps_mus,ps))/sum(ps)
    alphas = np.linspace(0,1,10)
    plt.plot(*pl(mean_ep,alphas))
    plt.plot(*pl(mean_mu,alphas))
        
        
def normalize_matrix(xss):
    Z = float(sum(map(sum,xss)))
    return mmap(lambda x:x/Z,xss)

def approximate_mu(matrix,n,G):
    Zb = Zb_from_matrix(matrix,G)
    return log(n) - log(Zb)

def select_sites_by_occupancy(matrix,mu,n):
    L = len(matrix)
    motif = []
    while len(motif) < n:
        site = random_site(L)
        if random.random() < 1/(1+exp(score_seq(matrix,site)-mu)):
            motif.append(site)
            print len(motif)
    return motif

def motif_degradation_experiment():
    """what is the effect of repeatedly inferring a motif from selected sites?"""
    from motifs import Escherichia_coli
    motif = Escherichia_coli.LexA
    n = len(motif)
    matrix = matrix_from_motif(motif)
    assumed_copies = 10*n
    mu = approximate_mu(matrix,assumed_copies,G)
    for i in range(10):
        print i,"motif ic:",motif_ic(motif)
        motif = select_sites_by_occupancy(matrix,mu,n)
        matrix = matrix_from_motif(motif)

def occupancies(matrix,motif,mu):
    eps = [score_seq(matrix,site) for site in motif]
    return [1/(1+exp(ep-mu)) for ep in eps]

def main_experiment(samples=30,iterations=10000,delta_ic = 0.1):
    results_dict = {}
    for tf_idx,tf in enumerate(tfdf.tfs):
        print "starting on:",tf
        motif = getattr(tfdf,tf)
        if motif_ic(motif) < 5:
            print "excluding",tf,"for low IC"
            continue
        bio_ic = motif_ic(motif)
        n = len(motif)
        L = len(motif[0])
        matrix = matrix_from_motif(motif)
        sigma = sigma_from_matrix(matrix)
        mu = approximate_mu(matrix,n,G)
        Ne = estimate_Ne(matrix,mu,n,bio_ic)
        spoofs = []
        ar = 0
        spoof_trials = 0.0
        while len(spoofs) < samples:
            spoof_trials += 1
            matrix, chain = sella_hirsch_mh(Ne=Ne,mu=mu,n=1,matrix=sample_matrix(L,sigma),
                                         init='ringer',iterations=iterations)
            spoof_motif = concat([random.choice(chain[iterations/2:]) for i in range(n)])
            if abs(motif_ic(spoof_motif) - bio_ic) < delta_ic:
                spoofs.append(spoof_motif)
                ar += 1
            print "spoof acceptance rate:",ar/spoof_trials,len(spoofs),samples,spoof_trials
        #spoofs = [chain[-1] for (spoof_matrix,chain,Ne) in [spoof_motif(motif,Ne) for i in range(samples)]]
        results_dict[tf] = {fname:map(eval(fname),spoofs)
                            for fname in "motif_ic motif_gini total_motif_mi".split()}
        print "finished:",tf,"(%s/%s)" % (tf_idx,len(tfdf.tfs))
        print bio_ic,mean_ci(results_dict[tf]['motif_ic'])
    return results_dict

def interpret_results_dict(results_dict,filename=None,annotate=False):
    ic_in_range = 0
    ic_lower = 0
    ic_upper = 0
    fnames = "motif_ic motif_gini total_motif_mi".split()
    rel_tfs = [tf for tf in tfdf.tfs if motif_ic(getattr(tfdf,tf)) > 5]
    for tf in rel_tfs:
        motif = getattr(tfdf,tf)
        print tf
        for fname_idx,fname in enumerate(fnames):
            f = eval(fname)
            bio_stat = f(motif)
            lb, ub = mean_ci(results_dict[tf][fname])
            in_range = (lb <= bio_stat <= ub)
            if fname == 'motif_ic':
                ic_in_range += (lb < bio_stat < ub)
                ic_lower += (bio_stat < lb)
                ic_upper += (ub < bio_stat)
            print fname, bio_stat, "(%1.2f, %1.2f)" % (lb,ub), in_range
    print "motif ic in range:",ic_in_range/float(len(rel_tfs))
    print "motif ic lower:",ic_lower/float(len(rel_tfs))
    print "motif ic higher:",ic_upper/float(len(rel_tfs))
    for fname_idx,fname in enumerate(fnames):
        f = eval(fname)
        plt.subplot(1,len(fnames),fname_idx+1)
        plt.title(fname)
        bio_stats = [f(getattr(tfdf,tf)) for tf in rel_tfs]
        sim_stats = [mean(results_dict[tf][fname]) for tf in rel_tfs]
        pred_obs(zip(bio_stats,sim_stats),show=False)
        if annotate:
            for s,xy in zip(rel_tfs,zip(bio_stats,sim_stats)):
                plt.annotate(s=s, xy=xy)
        plt.xlabel("Biological Value")
        plt.ylabel("Simulated Value")
        r, p = pearsonr(bio_stats,sim_stats)
        print fname,r, r**2, p
    plt.tight_layout()
    maybesave(filename)
