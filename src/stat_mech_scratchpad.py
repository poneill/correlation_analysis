from math import *
from utils import iota, pl, prod, normalize
from pwm_utils import Zb_from_matrix
import numpy as np

eps = np.arange(10)

def Z(lamb):
    return sum(np.exp(-lamb*eps))

def expect(f,lamb):
    return sum([f(ep)*exp(-lamb*ep) for ep in eps])/Z(lamb)

def diff_expect(f,lamb):
    return - (expect(lambda ep:ep*f(ep),lamb) - expect(iota, lamb)*expect(f,lamb))

def test():
    f = lambda x:x
    plt.plot(*pl(lambda lamb:expect(f,lamb), np.linspace(-10,10,1000)))
    plt.plot(*pl(lambda lamb:diff_expect(f,lamb), np.linspace(-10,10,1000)))
    plt.show()
    
def Dkl(matrix, nu, lamb):
    p_matrix = [normalize([exp(-lamb*ep) for ep in eps]) for eps in matrix]
    term0 = nu*prod(sum(exp(ep)*p for ep,p in zip(eps,p_eps))
                    for eps,p_eps in zip(matrix,p_matrix))
    term1 = lamb*prod(sum(ep*p for ep,p in zip(eps,p_eps))
                    for eps,p_eps in zip(matrix,p_matrix))
    L = len(matrix)
    term2 = -log(Zb_from_matrix(matrix,4**L))
    return term0 + term1 + term2

def fhat(matrix, site, nu, mu):
    ep = score_seq(matrix, site)
    return fhat_ep(ep, nu, mu)

def fhat_ep(ep, mu, nu):
    return (1/(1+exp(ep-mu)))**nu

def fhat_ep_approx(ep, mu, nu):
    return (1/(1+exp(ep-mu+log(nu))))

def log_fhat_ep(ep, mu, nu):
    return -nu*log(1+exp(ep-mu))

def log_fhat_approx(ep,mu,nu):
    if ep <= mu:
        return -nu*exp(ep-mu)
    else:
        return -nu*(ep-mu)

