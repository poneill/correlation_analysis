import random
from math import exp, log
from utils import mean
from scipy.interpolate import interp1d

def expect_Z(sigma):
    """return <Z> for a single column of (4) gaussian energies ~ N(0,sigma^2)"""
    return 4 * exp(sigma**2/2.0)

def expect_Z2(sigma):
    """return <Z^2> for a single column of (4) gaussian energies ~ N(0,sigma^2)"""
    return 4*exp(4*sigma**2/2) + 12*exp(sigma**2)

def approx_logZ(sigma,terms=2):
    Zhat = expect_Z(sigma)
    #return log(Zhat) + log(1+dZ_sq(sigma)/(Zhat**2))
    return log(Zhat) - (1/2.0*dZ_sq(sigma)/(Zhat**2))*(terms==2)

def dZ_sq(sigma):
    """return expectation of squared fluctuation about Z, i.e. <(Z-Zhat)^2>"""
    return expect_Z2(sigma) - expect_Z(sigma)**2

def rZ(sigma):
    return sum(exp(random.gauss(0,sigma)) for i in range(4))

def weights(sigma):
    # p is decreasing, q is increasing with sigma
    theta = 1
    alpha = 10
    p = 1.0/(1+exp(alpha*(sigma-theta)))
    q = 1 - p
    return p, q
    
def fit(sigma):
    theta = 2
    p,q = weights(sigma)
    return approx_logZ(sigma)*p + sigma*q

def fit2(sigma):
    p, q= weights(sigma)
    return log(expect_Z(sigma)) * p + sigma * q

def fit3(sigma):
    xs = [0,1,5,11]
    ys = [approx_logZ(0),approx_logZ(1),5,11]
    cubic_spline = interp1d(xs,ys,kind='cubic')
    return cubic_spline(sigma)

def fit4(sigma):
    M = 10
    a_ref = (-2+0.5*M)/M**3
    b_ref = (-4 + M)/M**2
    c_ref = 1/2.0
    d_ref = 0
    e_ref = log(4)
    a,b,c,d,e = [  9.15888308e-04,  -2.55451774e-02,   2.50000000e-01, 0.00000000e+00,   1.38629436e+00]
    return a*sigma**4 + b*sigma**3 + c * sigma**2 + d * sigma + e

def fit5(sigma):
    return log(4+exp(sigma))

def fit6(sigma):
    """approximate <log(Z)> through fenton-wilkinson method"""
    A = 4*exp(sigma**2/2.0)
    B = 4*(exp(sigma**2)-1)*exp(sigma**2)
    s_sq = log(B/(A**2)+1)
    m = log(A) - s_sq/2.0
    return m
    
def make_plot(obs=None):
    sigmas = np.linspace(0,10,1000)
    if obs is None:
        obs = map(lambda s:show(mean(log(rZ((s))) for i in range(1000))),sigmas)
    plt.plot(sigmas,obs)
    plt.plot(*pl(lambda s:log(expect_Z(s)),sigmas),label="annealed")
    plt.plot(*pl(approx_logZ,sigmas),label="approx_logZ")
    plt.plot(*pl(fit,sigmas),label="fit")
    plt.plot(*pl(fit2,sigmas),label="fit2")
    plt.plot(*pl(fit3,sigmas),label="fit3")
    plt.plot(*pl(fit4,sigmas),label="fit4")
    plt.plot(*pl(fit4,sigmas),label="fit5")
    plt.plot(*pl(fit6,sigmas),label="fit6")
    plt.plot(*pl(lambda x:x,sigmas))
    plt.legend()
    plt.ylim(0,10)
