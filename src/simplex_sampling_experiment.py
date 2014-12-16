"""
What is the most efficient way to pack info into a site: correlation or conservation?
"""
from mpl_toolkits.mplot3d import Axes3D
from utils import simplex_sample,h,norm,dot,transpose,log2,interpolate
from itertools import product
from tqdm import tqdm
import numpy as np
from math import log,exp,sqrt,acos,pi,cos,sin
from matplotlib import pyplot as plt
from scipy.stats import pearsonr
import sys
import random
sys.path.append("/home/pat/the_royal_we/src")
from we import weighted_ensemble

def sample(num_cols):
    return np.array(simplex_sample(4**num_cols))

def marginalize(ps):
    """turn ps into a psfm"""
    n = len(ps)
    assert log(n,4) == int(log(n,4))
    w = int(log(n,4))
    psfm = [[0 for j in range(4)] for i in range(w)]
    for k,digits in enumerate(product(*[[0,1,2,3] for i in range(w)])):
        for i,d in enumerate(digits):
            psfm[i][d] += ps[k]
    return psfm

def ic(ps):
    psfm = marginalize(ps)
    return sum(2-h(col) for col in psfm)

def plot_h_vs_ic(num_cols,trials,max_h=None):
    if max_h is None:
        pss = [sample(num_cols) for i in tqdm(range(trials))]
    else:
        pss = []
        while len(pss) < trials:
            ps = sample(num_cols)
            if h(ps) < max_h:
                pss.append(ps)
                print len(pss)
    ics = map(ic,tqdm(pss))
    hs = map(h,tqdm(pss))
    plt.scatter(hs,ics)
    plt.plot([0,2*num_cols],[2*num_cols,0])
    print pearsonr(ics,hs)
    plt.xlabel("Entropy")
    plt.ylabel("IC")
            
def project_to_simplex(v):
    """Project vector v onto probability simplex"""
    assert np.all(v >= 0)
    ans = v/np.sum(v)
    assert np.all(ans >= 0),v
    return ans

def normalize(p):
    """normalize to unit length"""
    return p/((np.linalg.norm(p)))
    
def grad_ref(p):
    """Return gradient of entropy constrained to simplex"""
    v = -(np.log(p) + 1) # true gradient
    n = normalize(np.ones(len(p))) # normal to prob simplex
    return v - v.dot(n)*n

def grad(p):
    return -np.log(p) + np.mean(np.log(p))
    
def grad_descent(num_cols,iterations=100,eta=1):
    #p = sample(num_cols)
    p = np.array(simplex_sample(num_cols))
    ps = [np.copy(p)]
    for i in xrange(iterations):
        g = grad(p)*min(p)
        #g *= min(p)/np.linalg.norm(g)
        #print p,g,h(p)
        p += eta*g
        ps.append(np.copy(p))
    return ps

def flow_to_h(hf,p0,tol=10**-2,eta=10**-6):
    """Given initial point p0, pursue gradient flow until reaching final entropy hf"""
    p = np.copy(p0)
    hp = h(p0)
    iterations = 0
    while abs(hp-hf) > tol:
        g = grad(p)
        p += g*(hf-hp) * eta
        hp = h(p)
        # if iterations % 1000 == 0:
        #     print "p:",p,"g:",g*eta,hp,np.linalg.norm(g*eta)
        iterations += 1
        if np.any(np.isnan(p)):
            return None
    return p

def sample_entropic_prior(K,eta=10**-3):
    max_ent = log2(K)
    hf = random.random() * max_ent
    p = flow_to_h(hf,simplex_sample(K),eta=eta)
    attempts = 0
    while p is None:
        #print attempts
        p = flow_to_h(hf,simplex_sample(K),eta=eta)
        #print p
        attempts += 1
    return p
    
def is_normalized(p):
    return abs(np.linalg.norm(p) - 1) < 10**-10
    
def circular_transport(p,q,iterations=1000):
    """return circular arc from p to q"""
    pcirc = normalize(p)
    qcirc = normalize(q)
    theta = acos(pcirc.dot(qcirc))
    k = normalize(np.cross(pcirc,qcirc))
    assert is_normalized(k),np.linalg.norm(k)
    bik = normalize(np.cross(k,pcirc))
    assert is_normalized(bik)
    return [(cos(t))*pcirc + (sin(t))*bik+k*k.dot(pcirc)*(1-cos(t)) for t in interpolate(0,theta,iterations)]
    
def plot_grad_descent(n):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    xs,ys,zs = transpose([[1,0,0],[0,1,0],[0,0,1],[1,0,0]])
    ax.plot(xs,ys,zs)
    for i in tqdm(range(n)):
        ps = grad_descent(3,iterations=1000,eta=0.01)
        ax.plot(*transpose(ps))

def plot_circular_transport(n):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    xs,ys,zs = transpose([[1,0,0],[0,1,0],[0,0,1],[1,0,0]])
    ax.plot(xs,ys,zs)
    q = project_to_simplex(np.array([1.0,1.0,1.0]))
    for i in range(n):
        p = simplex_sample(3)
        traj = circular_transport(p,q)
        ax.plot(*transpose(traj))

def plot_flattened_transport(n):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    xs,ys,zs = transpose([[1,0,0],[0,1,0],[0,0,1],[1,0,0]])
    ax.plot(xs,ys,zs)
    q = project_to_simplex(np.array([1.0,1.0,1.0]))
    for i in range(n):
        p = simplex_sample(3)
        traj = map(project_to_simplex,circular_transport(p,q))
        ax.plot(*transpose(traj))
        
def plot_h_vs_ic_entropy_prior(num_cols):
    def perturb(ps):
        qs = np.array([random.gauss(0,1) for p in ps])
        min_p = min(ps)
        qs *= min_p/(np.linalg.norm(qs))
        ps_prime = project_to_simplex(ps + qs)
        assert abs(sum(ps_prime) - 1) < 10**16
        assert np.all(ps_prime > 0)
        return ps_prime
    M = 100
    return weighted_ensemble(q=perturb,
                             f=h,
                             init_states = [sample(num_cols) for i in range(M)],
                             bins = interpolate(0,2*num_cols,1000),
                             M=M,
                             timesteps=100,
                             tau=1,verbose=1)
