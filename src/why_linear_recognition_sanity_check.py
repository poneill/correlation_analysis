from tqdm import *
from utils import score_seq,score_genome,random_site,mean,sd,variance,dnorm,maybesave,pl,log2,make_pssm,motif_ic
from utils import argmin,choose
from motifs import Escherichia_coli
import random
from math import exp,log,sqrt,pi
import numpy as np
import inspect
from matplotlib import pyplot as plt
import pandas as pd


G = 5*10**6

def prod(xs):
    return reduce(lambda x,y:x*y,xs)

def ln_mean(mu,sigma_sq):
    return exp(mu + sigma_sq/2.0)

def ln_median(mu,sigma_sq):
    return exp(mu)
    
def var_Zb(sigma,L,G):
    """compute variance of Zb"""
    return sqrt(G*(exp(L*sigma**2)-1)*exp(L*sigma**2))

def sample_matrix(L,sigma):
    return [[random.gauss(0,sigma) for j in range(4)] for i in range(L)]
    
def occ_ref(sigma,L,G=5*10**6):
    matrix = sample_matrix(L,sigma)
    ef = sum(min(row) for row in matrix)
    eps = [sum(random.choice(row) for row in matrix) for i in xrange(G)]
    Zb = sum(exp(-ep) for ep in eps)
    actual_occ = exp(-ef)/(exp(-ef)+Zb)
    # predicted_Zb = exp(L*sigma**2/2.0 + log(G))
    # predicted_occ = exp(-ef)/(exp(-ef)+predicted_Zb)
    #print "predicted Zb: %1.3e actual: %1.3e" % (predicted_Zb,Zb)
    #print "predicted occ: %1.3e actual occ: %1.3e" % (predicted_occ,actual_occ)
    return actual_occ

def exact_occ(matrix,G):
    fg = exp(-sum(min(row) for row in matrix))
    Zb = G*exact_mean_Zb(matrix)
    return fg/(fg+G*Zb)

def rbinom(n,p):
    return sum(random.random() < p for i in range(n))
    
def on_off_occ_ref(sigma,L,G=5*10**6):
    ef = -sigma*L
    p = 1/4.0
    eps = [-rbinom(L,p) for i in trange(G)]
    log_Zb = log_Zb_from_eps(eps)
    ###
    log_actual_occ = - log(1 + exp(log_Zb + ef))
    return exp(log_actual_occ)

def on_off_occ(sigma,L,G=5*10**6):
    ef = -sigma*L
    p = 1/4.0
    #eps = [-np.random.binomial(L,p) for i in xrange(G)]
    # log_Zb = log_Zb_from_eps(eps)
    eps = -np.random.binomial(L,p,G)
    log_Zb = log_Zb_from_eps_np(eps)
    ###
    log_actual_occ = - log(1 + exp(log_Zb + ef))
    return exp(log_actual_occ)

def on_off_occ2(sigma,L,G=5*10**6):
    ef = -sigma*L
    p = 1/4.0
    log_Zb = log(G) + log(sum(choose(L,i)*p**i*(1-p)**(L-i)*exp(sigma*i) for i in range(L+1)))
    log_actual_occ = - log(1 + exp(log_Zb + ef))
    return exp(log_actual_occ)
    
def occ(sigma,L,G=5*10**6):
    matrix = sample_matrix(L,sigma)
    ef = sum(min(row) for row in matrix)
    eps = [sum(random.choice(row) for row in matrix) for i in xrange(G)]
    #Zb = sum(exp(-ep) for ep in eps); next two lines do this more carefully
    log_Zb = log_Zb_from_eps(eps)
    ###
    log_actual_occ = - log(1 + exp(log_Zb + ef))
    return exp(log_actual_occ)
    # predicted_Zb = exp(L*sigma**2/2.0 + log(G))
    # predicted_occ = exp(-ef)/(exp(-ef)+predicted_Zb)
    #print "predicted Zb: %1.3e actual: %1.3e" % (predicted_Zb,Zb)
    #print "predicted occ: %1.3e actual occ: %1.3e" % (predicted_occ,actual_occ)
    #return actual_occ

def occ_gaussian(sigma,L,G=5*10**6):
    print sigma,L
    if sigma == 0:
        return 1/float(G+1)
    eps = -np.random.normal(0,L*sigma,G)
    ef = -L*sigma
    log_Zb = log_Zb_from_eps_np(eps)
    if log_Zb + ef < 700:
        log_actual_occ = - log(1 + exp(log_Zb + ef))
        return exp(log_actual_occ)
    else:
        return 0
    
def log_Zb_from_eps(eps):
    """compute log(sum(exp(-ep)) for ep in eps)"""
    min_ep = min(eps)
    log_Zb = -min_ep + log(sum(exp(-ep + min_ep) for ep in eps))
    return log_Zb

def log_Zb_from_eps_np(eps):
    """compute log(sum(exp(-ep)) for ep in eps)"""
    min_ep = np.min(eps)
    log_Zb = -min_ep + log(np.sum(np.exp(-eps + min_ep)))
    return log_Zb

def occ2(sigma,L,G=5*10**5,approx_fg=False):
    matrix = sample_matrix(L,sigma)
    if approx_fg == True:
        approx_fg = (L,sigma)
    return exact_occ(matrix,G,approx_fg)

def mutation_occ(sigma,L,G=5*10**5):
    matrix = sample_matrix(L,sigma)
    return exact_mutation_occ(matrix,G)
    
def logadd_ref(a,b):
    """given a,b: return log(a + b)"""
    return log(a+b)

def logadd(a,b):
    """given a,b: return log(a + b)"""
    return log(a) + log(1+exp(log(b)-log(a)))

def exact_occ_ref(matrix,G):
    #fg = exp(-sum(min(row) for row in matrix))
    log_fg = (-sum(min(row) for row in matrix))
    #Zb = G*exact_mean_Zb(matrix)
    log_Zb = log(G) + log_exact_mean_Zb(matrix)
    print log_fg,log_Zb,exp(-log(1+exp(log_Zb-log_fg)))
    return log_fg,log_Zb

def exact_occ(matrix,G,approx_fg=False):
    if type(approx_fg) is tuple:
        L,sigma = approx_fg
        log_fg = L*sigma
    else:
        log_fg = (-sum(min(row) for row in matrix))
    log_Zb = log(G) + log_exact_mean_Zb(matrix)
    # log(a + b) = log(a*(1+b/a)) = log(a) + log(1+b/a)
    #            = log(a) + log(1+b/a)
    # b/a = exp(log(b))/exp(log(a))
    # = exp(log(b) - log(a))
    # => log(a) + log(1+exp(log(b)-log(a)))
    log_denom = log_fg + log(1+exp(log_Zb - log_fg))
    log_occ = log_fg - log_denom
    return exp(log_occ)

def exact_mutation_occ(matrix,G):
    i = random.randrange(len(matrix))
    log_fg = (-sum(min(row) if j != i else random.choice(([r for r in (row) if not r == min(row)]))
                   for j,row in enumerate(matrix)))
    log_Zb = log(G) + log_exact_mean_Zb(matrix)
    # log(a + b) = log(a*(1+b/a)) = log(a) + log(1+b/a)
    #            = log(a) + log(1+b/a)
    # b/a = exp(log(b))/exp(log(a))
    # = exp(log(b) - log(a))
    # => log(a) + log(1+exp(log(b)-log(a)))
    log_denom = log_fg + log(1+exp(log_Zb - log_fg))
    log_occ = log_fg - log_denom
    return exp(log_occ)
    
def analyze_foreground_approximation():
    Ls = range(1,31)
    sigmas = np.linspace(0,50,100)
    exact_fgs = [[mean(-sum(min(row) for row in sample_matrix(L,sigma)) for i in range(3))
                  for L in Ls] for sigma in tqdm(sigmas)]
    approx_fgs = [[L*sigma for L in Ls] for sigma in sigmas]
    plt.subplot(1,3,1)
    plt.title("exact")
    plt.imshow(exact_fgs,interpolation='none',aspect='auto')
    plt.colorbar()
    plt.subplot(1,3,2)
    plt.title("approx")
    plt.imshow(approx_fgs,interpolation='none',aspect='auto')
    plt.colorbar()
    plt.subplot(1,3,3)
    plt.title("diff")
    plt.imshow((np.matrix(exact_fgs) - np.matrix(approx_fgs))/np.matrix(exact_fgs),
               interpolation='none',aspect='auto')
    plt.colorbar()
    plt.show()
    
def exact_occ_debug(matrix,G):
    log_fg = (-sum(min(row) for row in matrix))
    log_Zb = log(G) + log_exact_mean_Zb(matrix)
    # log(a + b) = log(a*(1+b/a)) = log(a) + log(1+b/a)
    #            = log(a) + log(1+b/a)
    # b/a = exp(log(b))/exp(log(a))
    # = exp(log(b) - log(a))
    # => log(a) + log(1+exp(log(b)-log(a)))
    log_denom = log_fg + log(1+exp(log_Zb - log_fg))
    log_occ = log_fg - log_denom
    return exp(log_fg),exp(log_Zb)

def occ_spec(sigma,L,G):
    log_K = sigma*L + log(G/float(4**L))
    log_prod = L*log(1.02*sigma + 0.13)
    Kprod = exp(log_K + log_prod)
    print "Kprod:",Kprod, ("WARNING:%s,%s" % (sigma,L)) * (Kprod < 100)
    return exp(-(log_K + log_prod))

    
def exact_mean_Zb(matrix):
    """return Zb up to constant G"""
    L = len(matrix)
    Zb = prod(sum(exp(-ep) for ep in row) for row in matrix)/(4**L)
    return Zb

def log_exact_mean_Zb(matrix):
    """return Zb up to constant G"""
    L = len(matrix)
    log_Zb = sum(log(sum(exp(-ep) for ep in row)) for row in matrix) - L*log(4)
    return log_Zb

def sample_mean_Zb(L,sigma):
    return exact_mean_Zb(sample_matrix(L,sigma))

def predict_mean_Zb(L,sigma):
    m,s = fw_method(0,sigma,4)
    Zb_mean = 1.0/(4**L)*ln_mean(L*m,sqrt(L)*s)
    # Y_i = sum(exp(-eps_bi) for i in range(4))
    expect_Y = ln_mean(m,s)
    expect_Y2 = ln_variance(m,s) + expect_Y**2
    Zb_variance = 1.0/(4**(2*L))*((expect_Y2**L) - expect_Y**(2*L))
    return Zb_mean,sqrt(Zb_variance)

def predict_log_Zb2(L,sigma):
    m,s = fw_method(0,sigma,4)
    c = 1.0/(4**L)
    log_Zb_mean = L*m + log(c)
    log_Zb_sd = sqrt(L)*s
    return log_Zb_mean,log_Zb_sd

    
def test_sample_mean_Zb(L,sigma,trials=1000):
    Zbs = [sample_mean_Zb(L,sigma) for i in trange(trials)]
    pred_mean,pred_sd = predict_mean_Zb(L,sigma)
    obs_mean,obs_sd = mean(Zbs),sd(Zbs)
    print "pred:",pred_mean,pred_sd
    print "obs:",obs_mean,obs_sd

def occ_crooks(sigma,L,G=5000000):
    m,s = predict_log_Zb2(L,sigma)
    log_K = log(G) + (-L*sigma)
    return mean_lgn(log_K + m,s)
    
def ln_mean(mu,sigma):
    return exp(mu+sigma**2/2.0)

def ln_variance(mu,sigma):
    return (exp(sigma**2)-1)*exp(2*mu+sigma**2)

def sample_Zb(G,L,sigma):
    eps = sample_eps(G,L,sigma)
    Zb = sum(exp(-ep) for ep in eps)
    return Zb
    
def sample_eps(G,L,sigma):
    matrix = [[random.gauss(0,sigma) for j in range(4)] for i in range(L)]
    eps = [sum(random.choice(row) for row in matrix) for i in range(G)]
    return eps

def sample_eps(G,L,sigma):
    matrix = [[random.gauss(0,sigma) for j in range(4)] for i in range(L)]
    eps = [sum(random.choice(row) for row in matrix) for i in range(G)]
    return eps

def predict_Zb(G,L,sigma):
    site_sigma_sq = 3/4.0 * L * sigma**2
    return G*exp(0+site_sigma_sq/2.0)

def predict_Zb2(G,L,sigma):
    site_mu = 0
    site_sigma_sq = 3/4.0 * L * sigma**2 # changed this from sigma
    expect = G*exp(site_sigma_sq/2.0)
    var = G*(exp(site_sigma_sq)-1)*exp(site_sigma_sq)
    return expect,sqrt(var)

def occ_final(G,L,sigma):
    site_mu = 0
    site_sigma_sq = 3/4.0 * L * sigma**2 # changed this from sigma
    K = G*exp(-L*sigma)
    return mean_lgn(site_mu,site_sigma_sq*K)
    
def compare_Zb2(G,L,sigma,trials=100):
    Zbs = [sample_Zb(G,L,sigma) for trial in trange(trials)]
    Zb_mu,Zb_sigma = predict_Zb2(G,L,sigma)
    print "expected:",Zb_mu,Zb_sigma
    print "observed:",mean(Zbs),sd(Zbs)

def predict_mean_Zb_from_matrix_deprecated(matrix,G):
    score_mu = sum(mean(row) for row in matrix)
    score_sigma_sq = sum(variance(row,correct=False) for row in matrix)
    predicted_Zb = exp(-score_mu + score_sigma_sq/2 + log(G)) # prediction given matrix
    return predicted_Zb

def predict_mean_Zb_from_matrix(matrix,G):
    L = len(matrix)
    expect_eps = reduce(lambda x,y:x*y,[sum(map(lambda x:exp(-x),row)) for row in matrix])/(4**L)
    Zb = G*expect_eps
    expect_eps_sq = reduce(lambda x,y:x*y,[sum(map(lambda x:exp(-2*x),row)) for row in matrix])/(4**L)
    Zb_sq = G*expect_eps_sq + (G**2-G)*expect_eps**2
    var = Zb_sq - Zb**2
    return Zb,sqrt(var)

def test_predict_mean_Zb_from_matrix(matrix,G,trials=100):
    # works.
    Zbs = [sample_Zb_from_matrix(matrix,G) for i in trange(trials)]
    m, s = predict_mean_Zb_from_matrix(matrix,G)
    print "expected:",m,s
    print "observed:",mean(Zbs),sd(Zbs)
    
def predict_median_Zb_from_matrix(matrix,G):
    score_mu = sum(mean(row) for row in matrix)
    score_sigma_sq = sum(variance(row,correct=False) for row in matrix)
    predicted_Zb = exp(-score_mu + log(G)) # prediction given matrix
    return predicted_Zb
    
def sample_Zb_from_matrix(matrix,G):
    eps = [sum(random.choice(row) for row in matrix) for i in range(G)]
    Zb = sum(exp(-ep) for ep in eps)
    return Zb
    
def predict_log_Zb(G,L,sigma):
    expect = log(G) + (sigma**2/2.0)
    var = log(G) + log(exp(sigma**2)-1) + (sigma**2)
    return expect,var
    
def mean_occ(sigma,L,G=5*10**6):
    ef = -L*sigma
    Zb = predict_Zb(G,L,sigma)
    return exp(-ef)/(exp(-ef) + Zb)

def mean_occ2(sigma,L,G=5*10**6):
    site_sigma_sq = 3/4.0*L*sigma**2
    ef = -L*sigma
    Zb = exp(site_sigma_sq/2.0 + log(G))
    return exp(-ef)/(exp(-ef) + Zb)

def mean_occ3(sigma,L,G=5*10**6,terms=2):
    if terms == 0:
        return 1/(1+G*exp(-L*sigma))
    a = exp(L*sigma)
    term0 = a/(a + G)
    term2 = sigma**2*L*(a*G/4*(a*G/4 - (a + G))/((a + G)**3))
    Zb = G
    dZb = -G/4.0
    d2Zb = G/4.0
    term2_ref2 = (sigma**2 * L * (2*a*dZb**2)/((Zb + a)**3) - a*d2Zb/((Zb + a)**2))/2.0
    term2_ref3 = (L*sigma**2*G*exp(L*sigma))/(2*(exp(L*sigma)+G)**2)*(G/(2*(exp(L*sigma) + G)**2) - 1)
    #term2 = 1/2*4*L*sigma**2*(2*exp(L*sigma)*G**2/16.0)/(exp(L*sigma)+G)**3 - (exp(L*sigma)*G/4.0)/(exp(L*sigma)+G)**2
    #print L,sigma,term0,term2, term0 + term2,abs(term2 - term2_ref3)
    if terms == 2:
        return term0 + term2
    elif terms == 0:
        return term0

def med_occ(sigma,L,G=5*10**6):
    "Not good in sigma -> infty limit"
    ef = -L*sigma
    Zb = exp(0 + log(G))
    return exp(-ef)/(exp(-ef) + Zb)

def meanify(f,trials=10):
    return lambda *args:mean([f(*args) for i in xrange(trials)])
    
def mode_occ(sigma,L,G=5*10**6):
    ef = -L*sigma
    Zb = exp(0 - (sigma**2)/2.0 + log(G))
    return exp(-ef)/(exp(-ef) + Zb)

def occ_sigma_0(sigma,L,G=5*10**6):
    return 1/(1+G*exp(L*sigma*(sigma/2.0 - 1)))
    
def occ_matrix(G=10**3,occ_f=occ,Ls = range(1,31),sigmas = np.linspace(0,10,100),plot=True):
    M = np.matrix([[occ_f(sigma,L,G) for L in Ls] for sigma in tqdm(sigmas)])
    if plot:
        plt.imshow(M,aspect='auto',interpolation='none')
        plt.xticks(Ls)
        plt.yticks(range(len(sigmas)),sigmas)
    return M
    

def plot_matrix(matrix,colorbar=True,show=True):
    plt.imshow(matrix,aspect='auto',interpolation='none')
    #plt.xticks(Ls)
    #plt.yticks(range(len(sigmas)),sigmas)


def make_ecoli_df():
    Ls = []
    Ls_adj = []
    ns = []
    sigmas = []
    labels = []
    motif_ics = []
    motif_ics_per_base = []
    for tf in Escherichia_coli.tfs:
        sites = getattr(Escherichia_coli,tf)
        L = len(sites[0])
        n = len(sites)
        ns.append(n)
        L_adj = len(sites[0])+log2(n)
        sigma = mean((map(sd,make_pssm(sites))))
        Ls.append(L)
        Ls_adj.append(L_adj)
        motif_ics.append(motif_ic(sites))
        motif_ics_per_base.append(motif_ic(sites)/float(L))
        sigmas.append(sigma)
    df = pd.DataFrame({"L":Ls,"n":ns,"sigma":sigmas,"motif_ic":motif_ics,"info_density":motif_ics_per_base},index=Escherichia_coli.tfs)
    return df
    
def make_ecoli_sigma_L_plot():
    Ls = []
    Ls_adj = []
    ns = []
    sigmas = []
    labels = []
    motif_ics = []
    motif_ics_per_base = []
    for tf in Escherichia_coli.tfs:
        sites = getattr(Escherichia_coli,tf)
        L = len(sites[0])
        n = len(sites)
        ns.append(n)
        L_adj = len(sites[0])+log2(n)
        sigma = mean((map(sd,make_pssm(sites))))
        Ls.append(L)
        Ls_adj.append(L_adj)
        motif_ics.append(motif_ic(sites))
        motif_ics_per_base.append(motif_ic(sites)/float(L))
        sigmas.append(sigma)
        labels.append(tf)
    sigma_space = np.linspace(0.1,3,10)
    crit_lambs_actual = map(lambda sigma:critical_lamb_actual(sigma,G=4.5*10**6,trials=100),tqdm(sigma_space))
    plt.subplot(1,6,1)
    plt.scatter(sigmas,Ls)
    for L,sigma,label in zip(Ls,sigmas,labels):
        plt.annotate(label,xy=(sigma,L))
    plt.plot(*pl(lambda sigma:critical_lamb(sigma,G=5*10**6),sigma_space))
    plt.plot(*pl(lambda sigma:critical_lamb(sigma,G=4.5*10**6),sigma_space))
    plt.plot(sigma_space,crit_lambs_actual)
    plt.subplot(1,6,2)
    plt.scatter(sigmas,Ls_adj)
    for L_adj,sigma,label in zip(Ls_adj,sigmas,labels):
        plt.annotate(label,xy=(sigma,L_adj))
    plt.plot(*pl(lambda sigma:critical_lamb(sigma,G=5*10**6),sigma_space))
    plt.plot(*pl(lambda sigma:critical_lamb(sigma,G=4.5*10**6),sigma_space))
    plt.plot(sigma_space,crit_lambs_actual)
    preds = [critical_lamb(sigma,G=4.5*10**6) for sigma in tqdm(sigmas)]
    preds_actual = [critical_lamb_actual(sigma,G=4.5*10**6,trials=100) for sigma in tqdm(sigmas)]
    plt.subplot(1,6,3)
    plt.scatter(preds,Ls)
    plt.xlabel("Predicted Length")
    plt.ylabel("Observed Length")
    plt.title("Preds vs Ls")
    print "Preds vs Ls",pearsonr(preds,Ls)
    plt.plot([0,30],[0,30])
    plt.subplot(1,6,4)
    plt.scatter(preds,Ls_adj)
    plt.xlabel("Predicted Length")
    plt.ylabel("Observed Length")
    plt.plot([0,30],[0,30])
    plt.title("Preds vs Ls_adj")
    print "Preds vs Ls_adj",pearsonr(preds,Ls_adj)
    plt.subplot(1,6,5)
    plt.scatter(preds_actual,Ls)
    plt.xlabel("Predicted Length")
    plt.ylabel("Observed Length")
    plt.plot([0,30],[0,30])
    plt.title("Preds_actual vs Ls")
    print "Preds_actual vs Ls",pearsonr(preds_actual,Ls)
    plt.subplot(1,6,6)
    plt.scatter(preds_actual,Ls_adj)
    plt.xlabel("Predicted Length")
    plt.ylabel("Observed Length")
    plt.plot([0,30],[0,30])
    plt.title("Preds_actual vs Ls_adj")
    print "Preds_actual vs Ls_adj",pearsonr(preds_actual,Ls_adj)
    return Ls,sigmas
        
def every(n,xs):
    return [x for i,x in enumerate(xs) if i % n == 0]

def fmt(x):
    return "%1.2f" % x
    
def plot_matrix(matrix,colorbar=True,show=True,vmin=None,vmax=None,Ls=range(1,31),sigmas=np.linspace(0,10,100),xlabel="Binding Site Length",ylabel="Standard Deviation of Weight Matrix"):
    plt.imshow(matrix,aspect='auto',interpolation='none',vmin=vmin,vmax=vmax)
    plt.xticks(every(5,Ls))
    plt.yticks(every(5,range(len(sigmas))),map(fmt,every(5,sigmas)))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if colorbar:
        plt.colorbar(label="Occupancy")
    if show:
        plt.show()

def make_occupancy_figure():
    occs2 = occ_matrix(G=5*10**6,occ_f=lambda *args:mean(occ2(*args) for i in range(1000)),
                       sigmas=np.linspace(0,10,100))
    plot_matrix(occs2,show=False)
    plt.title("Mean Occupancy of Random Gaussian Ensembles of PWMs")
    maybesave("basic_occupancy1000.png")

def make_sigma_infty_asymptote_figure():
    Ls = range(1,20)
    sigma = 100
    plt.plot(*pl(lambda L:mean(occ2(sigma,L,G=5*10**6) for i in range(100)),Ls),label='Occupancy')
    plt.ylabel("Occupancy")
    plt.xlabel("Length")
    plt.plot([11.12,11.12],[0,1],linestyle='--',label='Predicted Critical Length')
    plt.plot(Ls,[0.5]*len(Ls),linestyle='--',label="occ = 1/2")
    plt.legend(loc='upper left')
    plt.title("Mean Occupancy for sigma = 100")
    maybesave("sigma_infty_asymptote.png")

def make_sigma_0_figure(sigma=0.1,fname="sigma_0.png"):
    G = 5*10**6
    def critical_L(sigma):
        return log(G)/(sigma*(1-sigma/2.0))
    Lstar = critical_L(sigma)
    print "Lstar:",Lstar
    Ls = range(1,int(2*Lstar))
    plt.plot(*pl(lambda L:mean(occ2(sigma,L,G=5*10**6) for i in range(100)),Ls),label='Occupancy')
    plt.ylabel("Occupancy")
    plt.xlabel("Length")
    plt.plot([Lstar,Lstar],[0,1],linestyle='--',label='Predicted Critical Length')
    plt.plot(Ls,[0.5]*len(Ls),linestyle='--',label="occ = 1/2")
    plt.legend(loc='upper left')
    plt.title("Mean Occupancy for sigma = %s" % sigma)
    maybesave(fname)

def occ_sigma_0(G,L,sigma):
    if L*sigma*(sigma/2.0 -1) > 500:
        return 0
    Zb = G*exp(L*sigma*(sigma/2.0 -1))
    return 1/(1+Zb)

def occ_sigma_inf(G,L,sigma):
    pseudos = exp((log(G) - L*log(4)))
    return 1/(1+pseudos)
    
def occ_approx(sigma,L,G=5*10**6):
    Lstar = log(G)/log(4)
    if L < Lstar:
        return 0
    else:
        crit_val = log(G)/(L-Lstar)
        return int(sigma > crit_val)

def critical_lamb(sigma,G=5*10**6):
    return log(G)*(1/sigma + 1/log(4))

def critical_lamb_actual(sigma,G=5*10**6,trials=10):
    Ls = range(1,30)
    occs = [meanify(occ2,trials)(sigma,L,G) for L in Ls]
    idx = argmin([(x-0.5)**2 for x in occs])
    return Ls[idx]

def occ_approx2(sigma,L,G=5*10**6):
    lamb = critical_lamb(sigma,L,G)
    #return int(L > lamb)
    return 1/(1+exp(lamb-L))

def make_occ_approx2_fig():
    pred_occs = occ_matrix(occ_f=occ_approx2,G=5*10**6)
    obs_occs = occ_matrix(occ_f=meanify(occ2,trials=10),G=5*10**6)
    plot_matrices(pred_occs,obs_occs,obs_occs-pred_occs,labels=["Predicted","Observed","Difference"])
    
def plot_matrices(*args,**kwargs):
    print kwargs
    n = len(args)
    for i,arg in enumerate(args):
        plt.subplot(1,n,i+1)
        if "labels" in kwargs:
            plt.title(kwargs["labels"][i])
        plt.imshow(arg,interpolation='none')
        #plot_matrix(arg,show=False,xlabel=None,ylabel=None)
        plt.xlabel("Binding Site Length")
        plt.ylabel("Standard Deviation of Weight Matrix")
    plt.colorbar(label='Occupancy')
    fname = kwargs.get('fname',None)
    maybesave(fname)

def convolve(M,d=0.1):
    R,C = M.shape
    out = np.zeros((R,C))
    for i in xrange(R):
        for j in xrange(C):
            count = 0
            for ip in (-1,1):
                for jp in (-1,1):
                    if 0 <= i + ip < R and 0 <= j + jp < C:
                        out[i+ip,j+jp] += M[i,j] * d
                        count += 1
            out[i,j] += M[i,j] * (1-d*count)
    return out
    
def test_Zb_approx(trials=10,G=5*10**6,L=10):
    predicted_Zb = exp(L*sigma**2/2.0 + log(G)) # a priori prediction
    matrix = [[random.gauss(0,sigma) for j in range(4)] for i in range(L)]
    score_mu = sum(mean(row) for row in matrix)
    score_sigma_sq = sum(variance(row,correct=False) for row in matrix)
    predicted_Zb2 = exp(score_mu + score_sigma_sq/2 + log(G)) # prediction given matrix
    Zbs = []
    for trial in trange(trials):
        eps = [sum(random.choice(row) for row in matrix) for i in range(G)]
        Zb = sum(exp(-ep) for ep in eps)
        Zbs.append(Zb)
    print "Predicted: %1.3e +/- %1.3e" % (predicted_Zb,sqrt(var_Zb(sigma,L,G)))
    print "Predicted2: %1.3e" % (predicted_Zb2)
    print "Actual: %1.3e +/- %1.3e" % (mean(Zbs),sd(Zbs))

def sample_integrate_multivariate_normal(f,trials=1000):
    num_args = len(inspect.getargspec(f).args)
    return mean(f(*([random.gauss(0,1) for i in range(num_args)])) for i in range(trials))

def predict_integrate_multivariate_normal(f,fpp):
    """given f and fpp, a list containing diagonal elements of hessian
    matrix Hii evaluated at 0, estimate integral by expectation of taylor expansion"""
    n = len(fpp)
    return f(*[0 for i in range(n)]) + sum(fpp)/2.0

def test_integrate_multivariate_normal(trials=1000):
    f = lambda x,y,z:x**2 + y**2 + z**2
    fpp = [2,2,2]
    pred = predict_integrate_multivariate_normal(f,fpp)
    obs = sample_integrate_multivariate_normal(f,trials=trials)
    print "pred:",pred
    print "obs:",obs
    
def dZdi_ref(matrix,i,delta=0.01):
    """given an epsilon matrix and an index i, give dZ/deps_i numerically"""
    dmatrix = [row[:] for row in matrix]
    dmatrix[i/4][i%4] += delta
    Zb = exact_mean_Zb(matrix)
    dZb = exact_mean_Zb(dmatrix)
    return (dZb - Zb)/delta

def dZdi(matrix,i):
    row_idx = i//4
    col_idx = i%4
    _matrix = [row for i,row in enumerate(matrix) if not i == row_idx]
    ep = matrix[row_idx][col_idx]
    return -exp(-ep)*exact_mean_Zb(_matrix)/4

def dZdii_ref(matrix,i,delta=0.01):
    """given an epsilon matrix and an index i, give dZ/deps_i numerically"""
    dmatrix_b = [row[:] for row in matrix]
    dmatrix_f = [row[:] for row in matrix]
    dmatrix_b[i/4][i%4] -= delta
    dmatrix_f[i/4][i%4] += delta
    Zb = exact_mean_Zb(matrix)
    dZb_b = exact_mean_Zb(dmatrix_b)
    dZb_f = exact_mean_Zb(dmatrix_f)
    return (dZb_b - 2* Zb + dZb_f)/delta**2

def dZdii(matrix,i):
    row_idx = i//4
    col_idx = i%4
    _matrix = [row for i,row in enumerate(matrix) if not i == row_idx]
    ep = matrix[row_idx][col_idx]
    return exp(-ep)*exact_mean_Zb(_matrix)/4

def dZdij_ref(matrix,i,j,delta=0.01):
    """given an epsilon matrix and an index i, give dZ/deps_i numerically"""
    dmatrix_ff = [row[:] for row in matrix]
    dmatrix_fb = [row[:] for row in matrix]
    dmatrix_bf = [row[:] for row in matrix]
    dmatrix_bb = [row[:] for row in matrix]
    dmatrix_ff[i/4][i%4] += delta
    dmatrix_ff[j/4][j%4] += delta
    
    dmatrix_fb[i/4][i%4] += delta
    dmatrix_fb[j/4][j%4] -= delta
    
    dmatrix_bf[i/4][i%4] -= delta
    dmatrix_bf[j/4][j%4] += delta
    
    dmatrix_bb[i/4][i%4] -= delta
    dmatrix_bb[j/4][j%4] -= delta
    dZb_ff = exact_mean_Zb(dmatrix_ff)
    dZb_fb = exact_mean_Zb(dmatrix_fb)
    dZb_bf = exact_mean_Zb(dmatrix_bf)
    dZb_bb = exact_mean_Zb(dmatrix_bb)
    return (dZb_ff - dZb_fb - dZb_bf + dZb_bb)/(4*(delta**2))

def dZdij(matrix,i,j):
    if i // 4 == j//4: # if i, j in same row
        return 0
    else:
        row_idx = i//4
        col_idx = i%4
        row_jdx = j//4
        col_jdx = j%4
        _matrix = [row for i,row in enumerate(matrix)
                   if not (i == row_idx or i == row_jdx)]
        epi = matrix[row_idx][col_idx]
        epj = matrix[row_jdx][col_jdx]
        return exp(-(epi + epj))/16*exact_mean_Zb(_matrix)

def dZdkk(matrix,i,j):
    if i == j:
        return dZdii(matrix,i)
    else:
        return dZdij(matrix,i,j)
    
def Z_hessian(matrix):
    L = len(matrix)
    return np.matrix([[dZdkk(matrix,i,j) for i in range(4*L)]
                      for j in range(4*L)])

def rlgn(mu,sigma):
    return 1/(1+exp(-random.gauss(mu,sigma)))

def mean_lgn(mu,sigma):
    gamma = sqrt(1+pi*sigma**2/8)
    return 1/(1+exp(-mu*gamma))

def diffuse_array(xs,d=0.1):
    n = len(xs)
    ys = [0]*n
    for i in range(n):
        if i == 0:
            ys[i] = (1-d)*xs[i] + d*xs[i+1]
        elif i == n-1:
            ys[i] = (1-d)*xs[i] + d*xs[i-1]
        else:
            ys[i] = (1-2*d)*xs[i] + d*(xs[i-1] + xs[i+1])
    return ys

def diffusion_experiment():
    mus = np.linspace(-10,10,100)
    sigmas = np.linspace(0.001,10,100)
    obs = [[mean(rlgn(mu,sigma) for i in xrange(100)) for mu in mus] for sigma in sigmas]
    xs = obs[0]
    scaling_factor = 12
    def make_pred(scaling_factor):
        return [row for i,row in enumerate(iterate_list(lambda xs:diffuse_array(xs,d=0.5),xs,scaling_factor*len(sigmas))) if i % scaling_factor == 0]
    def diff(pred):
        return (sum((x-y)**2 for (ob_row,pred_row) in zip(obs,pred) for (x,y) in transpose([ob_row,pred_row])))

    
def iterate_list(f,x,n):
    xs = [x]
    for i in range(1,n+1):
        xs.append(f(xs[-1]))
    return xs

def occ3(sigma,L,G=5*10**5):
    """estimate occupancy with 2nd order taylor expansion"""
    zeroth_term = G*exact_mean_Zb([[0 for i in range(4)] for j in range(L)])


def infinite_sigma_limit(G):
    """return minimum length necessary to occupy inf sigma limit"""
    # expected number of pseudo sites = G/(4**L)
    # occ = 1/(1+G/(4**L))
    # G < 4**L
    # log G < L * log(4)
    # L > log(G)/log(4)
    return log(G)/log(4)

def fw_method(mu,sigma,N):
    """given sum_i=1^N exp(N(mu,sigma^2)), assume sum is log-normal and
    find first two moments through fenton-wilkinson moment matching"""
    A = N*exp(mu+(sigma**2)/2.0)
    B = N*(exp(sigma**2)-1)*exp(2*mu+sigma**2)
    s = sqrt((log(B/(A**2) + 1)))
    m = log(A) - (s**2)/2
    s_simp = sqrt(log((exp(sigma**2) - 1)/N + 1))
    if sigma**2 > 10*log(N-1): # >> log(N-1)
        s_approx = sqrt(sigma**2-log(N))
        m_approx = log(A) - (s_approx**2)/2
    else:
        s_approx = 0
        m_approx = 0
    #s2 = sqrt(log(N)-(log(exp(sigma**2)-1) - log(N) + 1))
    #m2 = mu + sigma**2/2.0 + log(N) - s2**2/2.0
    # print "pred M,V,log(V/(M**2)):",M,V,log(V/(M**2))
    # print "m,s:",m,s
    # print "m_approx,s_approx:",m_approx,s_approx
    return m,s

def occ_fw():
    pass
    
def fw_method2(mu,sigma,N):
    A = N*exp(mu+(sigma**2)/2.0)
    B = N*(exp(sigma**2)-1)*exp(2*mu+sigma**2)
    s_sq = log((exp(sigma**2)-1)/N + 1)
    m = log(N) + mu + sigma**2/2.0 - s_sq/2.0
    return m,sqrt(s_sq)
    
def rfw(mu,sigma,N):
    return log(sum(exp(random.gauss(mu,sigma)) for i in range(N)))
    
def test_fw_method(reps=10):
    mus = np.linspace(-10,10,100)
    sigmas = np.linspace(0,10,100)
    obs = np.matrix([[mean(rfw(mu,sigma,4) for i in xrange(reps)) for mu in mus] for sigma in tqdm(sigmas)])
    pred = np.matrix([[fw_method(mu,sigma,4)[0] + sigma for mu in mus] for sigma in sigmas]) # + sigma [why?]
    diff = pred - obs
    print "max diff:",np.matrix.max(np.abs(diff))
    vmin = min(np.matrix.min(obs),np.matrix.min(pred),np.matrix.min(diff))
    vmax = max(np.matrix.max(obs),np.matrix.max(pred),np.matrix.max(diff))
    plt.subplot(1,3,1)
    plot_matrix(pred,show=False,vmin=vmin,vmax=vmax,colorbar=False)
    plt.subplot(1,3,2)
    plot_matrix(obs,show=False,vmin=vmin,vmax=vmax,colorbar=False)
    plt.subplot(1,3,3)
    plot_matrix(pred-obs,show=False,vmin=vmin,vmax=vmax)
    return pred,obs
    
def sample_power_sum(mu,sigma,N):
    return sum(exp(random.gauss(mu,sigma)) for i in range(N))

def sample_log_power_sum(mu,sigma,N):
    return log(sample_power_sum(mu,sigma,N))
    
def occ_fw(sigma,L,G=10**6):
    log_K = sigma*L + log(G) - L*log(4)
    m,s = fw_method(0,sigma,4)
    log_occ_m = -m - log_K
    log_occ_s = s
    return exp(log_occ_m + (log_occ_s**2)/2)

def log_1px(x,terms=2):
    return log(1) + sum((-1)**(i+1)*(x**i)/float(i) for i in range(1,terms+1))
    
def test_fw_method2(mu,sigma,N,trials=10000):
    xs = [sum(exp(random.gauss(mu,sigma)) for i in range(N)) for j in xrange(trials)]
    M,V = mean(xs),variance(xs)
    print "obs M,V,log(V/(M**2)):",M,V,log(V/(M**2))
    ys = map(log,xs)
    m_obs,s_obs = mean(ys),sd(ys)
    m,s = fw_method(mu,sigma,N)
    print "pred:",m,s
    print "obs:",m_obs,s_obs

def rvar(mu,sigma):
    return log(1+exp(random.gauss(mu,sigma)))

def dvar(x,mu,sigma):
    return dnorm(log(exp(x)-1),mu,sigma)*exp(x)/(exp(x)-1)

def rlgn(mu,sigma):
    return 1/(1+exp(-random.gauss(mu,sigma)))
    
def dlgn(x,mu,sigma):
    """return density of 1/(1+exp(N(mu,sigma**2)))"""
    return dnorm(log(1/x-1),mu,sigma)*1/(x*(1-x))
    
def rln(mu,sigma):
    return exp(random.gauss(mu,sigma))

def mean_lgn(mu,sigma):
    """approximate expectation of logit normal rv via Maragakis-Crooks approximation"""
    #print "mean lgn on:",mu,sigma
    gamma = sqrt(1+pi*sigma**2/8.0)
    if -300 < gamma*mu < 300:
        return 1/(1+exp(-gamma*mu))
    else:
        return 1 if gamma*mu > 0 else 0

def dphidmu(x,mu,sigma):
    return (x-mu)/float(sigma**2)*dnorm(x,mu,sigma)

def test_dphidmu():
    x = random.random()
    mu = random.random()
    sigma = random.random()
    pred = dphidmu(x,mu,sigma)
    obs = diff(lambda mu:dnorm(x,mu,sigma),mu,10**-10)
    return pred,obs

def test_dphidmu_plot():
    "correct"
    plt.scatter(*transpose([test_dphidmu() for i in range(1000)]))
    plt.show()

def dphidsigma(x,mu,sigma):
    return dnorm(x,mu,sigma) * ((x-mu)**2/(sigma**3) - 1/sigma**2)

def test_dphidsigma():
    x = random.random()
    mu = random.random()
    sigma = random.random()
    pred = dphidsigma(x,mu,sigma)
    obs = diff(lambda sigma:dnorm(x,mu,sigma),sigma,10**-10)
    return pred,obs

def test_dphidsigma_plot():
    "correct"
    plt.scatter(*transpose([test_dphidmu() for i in range(1000)]))
    plt.show()

def dmeanlgn_dmu(mu,sigma):
    pass
    
def diff(f,x,dx=0.01):
    return (f(x+dx) - f(x))/float(dx)
    
def test_mean_lgn(trials = 100):
    mus = np.linspace(-10,10,100)
    sigmas = np.linspace(0.01,10,100)
    pred = np.matrix([[mean_lgn(mu,sigma) for mu in mus] for sigma in sigmas])
    obs = np.matrix([[mean(rlgn(mu,sigma) for i in xrange(trials)) for mu in mus] for sigma in tqdm(sigmas)])
    plt.subplot(1,3,1)
    plt.title("Predicted")
    plt.imshow(pred,interpolation='none')
    plt.colorbar()
    plt.subplot(1,3,2)
    plt.title("Observed")
    plt.imshow(obs,interpolation='none')
    plt.colorbar()
    plt.subplot(1,3,3)
    plt.title("Predicted - Observed")
    plt.imshow(pred-obs,interpolation='none')
    plt.colorbar()
    plt.show()

def log1pex(x):
    """approximate log(1+e^x)"""
    # log(x+1) = 0 + x - x^2/2 + x^3/3 + ...
    log(2) + x/2.0 + x**2/80 -x**4/192.0 + x**6/2880.0

def main():
    pred_occs = occ_matrix(occ_f=occ2)
    obs_occs = occ_matrix(occ_f=occ)
    

