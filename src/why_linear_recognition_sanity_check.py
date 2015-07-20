from tqdm import *
from utils import score_seq,score_genome,random_site,mean,sd,variance
import random
from math import exp,log,sqrt
import numpy as np
import inspect
from matplotlib import pyplot as plt

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
    
def occ(sigma,L,G=5*10**6):
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

def exact_mean_Zb(matrix):
    """return Zb up to constant G"""
    L = len(matrix)
    Zb = prod(sum(exp(-ep) for ep in row) for row in matrix)/(4**L)
    return Zb

def sample_Zb(G,L,sigma):
    eps = sample_eps(G,L,sigma)
    Zb = sum(exp(-ep) for ep in eps)
    return Zb

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
    ef = -L*sigma
    Zb = exp(0 + log(G))
    return exp(-ef)/(exp(-ef) + Zb)

def mode_occ(sigma,L,G=5*10**6):
    ef = -L*sigma
    Zb = exp(0 - (sigma**2)/2.0 + log(G))
    return exp(-ef)/(exp(-ef) + Zb)

    
def occ_matrix(G=10**3,occ_f=occ,plot=True):
    Ls = range(1,31)
    sigmas = np.linspace(0,5,100)
    M = np.matrix([[occ_f(sigma,L,G=G) for L in Ls] for sigma in tqdm(sigmas)])
    if plot:
        plt.imshow(M,aspect='auto',interpolation='none')
        plt.xticks(Ls)
        plt.yticks(range(len(sigmas)),sigmas)
    return M
    

def plot_matrix(matrix,colorbar=True,show=True):
    plt.imshow(matrix,aspect='auto',interpolation='none')
    #plt.xticks(Ls)
    #plt.yticks(range(len(sigmas)),sigmas)
    if colorbar:
        plt.colorbar()
    if show:
        plt.show()
    
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

