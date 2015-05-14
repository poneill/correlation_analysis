import random
from utils import mh

def sample_path(lamb,tf):
    """Simulate a sample path of a poisson process with rate lamb up to time tf"""
    t = 0
    hist = []
    while t < tf:
        print t,tf
        t += random.expovariate(lamb)
        if t < tf:
            hist.append(t)
    return (hist,tf)

def sp_density(obs,lamb):
    hist,tf = obs
    m = len(hist)
    return lamb**m*exp(-lamb*tf)

def log_sp_density(obs,lamb):
    hist,tf = obs
    m = len(hist)
    return m*log(lamb) - lamb*tf

def proposal(lamb,tf):
    
