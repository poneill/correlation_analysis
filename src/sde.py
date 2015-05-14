"""
Library for stochastic processes and SDEs
"""

import numpy as np
from math import sqrt, log, exp
from scipy import polyfit,poly1d
from utils import sd,mean
import scipy

def bm(n):
    return np.hstack((0,np.cumsum(np.random.normal(0,1.0/sqrt(n),n-1))))

def brownian_bridge(n):
    """sample brownian bridge from 0 to 0"""
    xs = bm(n)
    x1 = xs[-1]
    ts = np.linspace(0,1,n)
    return xs - ts*x1
    
def ou(beta,sigma,y0,n,tf,K=0):
    """Return a sample path of the ornstein-uhlenbeck process:
    dY = -beta*Y + sigma*dW starting at y0, for n timesteps"""
    hist = np.zeros(n)
    dt = tf/float(n)
    y = y0
    for i in xrange(n):
        hist[i] = y
        y += -beta*(y-K)*dt + sigma*np.random.normal(0,sqrt(dt))
    return hist

def ou_relaxation_time(beta,sigma,y0):
    return log(sqrt(2*beta)*abs(y0)/sigma)/beta

def ou_relaxation_time_from_sample(xs):
    beta,sigma,k = ou_param_recovery(xs)
    if beta is None:
        return 0
    rel_time = ou_relaxation_time(beta,sigma,xs[0])
    return rel_time
    
def ou_parameters(beta,sigma,y0,t):
    """given parameters of oe process, return mean and variance of normal
    distribution at time t"""
    Ymu = y0*exp(-beta*t)
    Ysigma2 = (sigma**2)/(2*beta)*(1-exp(-2*beta*t))
    return Ymu,Ysigma2

def ou_param_recovery(xs):
    """return MLE estimate of ou parameters: dX = -beta*(X-k) + sigma*dW """
    xs = np.array(xs)
    ys = np.diff(xs)
    xs_ = xs[:-1]
    x0 = xs[0]
    neg_beta ,_ = polyfit(xs_,ys,1) # find MLE estimate of beta
    beta = - neg_beta
    if beta < 0:
        print "Warning: beta negative"
        return None,None,None
    sigma = sd(ys + beta*xs_)
    rel_time = ou_relaxation_time(beta,sigma,x0)
    if rel_time < len(xs):
        print "estimated relaxation time:",rel_time
        k = mean(xs[int(rel_time):]) # estimate mean from series after relaxation time
    else:
        print "Warning: process predicted not to reach relaxation time."
        k = None
    return beta,sigma,k

def ou_param_recovery_test(n,steps=10000):
    params = []
    rec_params = []
    for _ in xrange(n):
        beta,sigma,y0 = random.random(),random.random()*10,random.random()*100
        beta_rec,sigma_rec = ou_param_recovery(ou(beta,sigma,y0,steps))
        params.append((beta,sigma))
        rec_params.append((beta_rec,sigma_rec))
        betas,sigmas = transpose(params)
        rec_betas,rec_sigmas = transpose(rec_params)
        plt.scatter(betas,rec_betas)
        plt.scatter(sigmas,rec_sigmas,color='g')
        plt.plot([0,100],[0,100])
        
def sde(b,a,x0,n):
    """return sample path for sde:
    dX = b(X)dX + b(X)dW
    """
    hist = np.zeros(n)
    x = x0
    for i in xrange(n):
        hist[i] = x
        x += b(x) + a(x)*np.random.normal(0,1)
        print x
    return hist

def spoof_ou(xs):
    beta,sigma,k = ou_param_recovery(xs)
    return ou(beta,sigma,xs[0],len(xs),tf=len(xs),K=k)
    #def ou(beta,sigma,y0,n,tf,K=0):
def autocorr(x):
    result = numpy.correlate(x, x, mode='same')
    return result[result.size/2:]

def is_stationary(xs,alpha=0.05):
    """Test for stationarity by testing normality of increments"""
    dxs = np.diff(np.array(xs))
    val,pval = scipy.stats.ttest_1samp(dxs,0)
    return pval > alpha
