"""
What is the most efficient way to pack info into a site: correlation or conservation?

Fundamental conservation: total_mi(ps) + h_np(ps) + ic(ps) = 2*w
"""
from mpl_toolkits.mplot3d import Axes3D
from utils import simplex_sample,h,norm,dot,transpose,log2,interpolate,pl,fac
from utils import maybesave,choose2,zipWith,bisect_interval,mean,show,hamming
from itertools import product,permutations
from tqdm import tqdm,trange
import numpy as np
from math import log,exp,sqrt,acos,pi,cos,sin,gamma
from matplotlib import pyplot as plt
import scipy
from scipy.stats import pearsonr,spearmanr
from scipy.special import gammaln
import sys
import random
from collections import defaultdict
import sympy
sys.path.append("/home/pat/the_royal_we/src")
from we import weighted_ensemble

base_index = {b:k for k,b in enumerate("ACGT")}

def sample(num_cols):
    return np.array(simplex_sample(4**num_cols))

def h_np_ref(ps):
    """Return entropy in bits"""
    return -np.sum(ps*np.log(ps))/log(2)

def h_np(ps):
    """Return entropy in bits"""
    psp = ps[ps>0]
    return -np.sum(psp*np.log(psp))/log(2)

def dkl(ps,qs):
    return np.sum(ps*np.log(ps/qs))

def inst_dkl_ref(ps,Mut=None,dt=10**-6):
    """Return rate of change of the KL divergence against mutation"""
    L = int(log(len(ps),4))
    if Mut is None:
        Mut = mutation_matrix_ref(mu=1,w=L)
    psp = mutate(ps,dt,Mut)
    return 2*dkl(ps,psp)/(dt**2) # why dt**2 though?

def inst_dkl(ps,Mut=None):
    L = int(log(len(ps),4))
    if Mut is None:
        Mut = mutation_matrix_ref(mu=1,w=L)
    psp = Mut.dot(ps)
    psp2 = Mut.dot(psp)
    return -np.sum(psp2 - psp**2/ps)

def dHdt(ps,Mut=None,dt=10**-6):
    """Return instantaneous rate of change of entropy"""
    L = int(log(len(ps),4))
    if Mut is None:
        Mut = mutation_matrix_ref(mu=1,w=L)
    psp = mutate(ps,dt,Mut)
    return (h(psp)-h(ps))/dt

def dHdt2(ps):
    L = num_cols_from_vector(ps)
    fs = ps - norm_lap(ps)
    term1 = cross_h(fs,ps)
    term2 = h(ps)
    return  3*L*(term1 - term2)

def dHdt3(ps):
    L = num_cols_from_vector(ps)
    return  3*L*cross_h(-norm_lap(ps),ps)
    
def inst_dkl_experiment(L,N=100,dt=10**-6,desired_entropy=None):
    K = 4**L
    if desired_entropy is None:
        desired_entropy = L # 1 bit/base
    alpha = find_alpha(K,desired_entropy)
    Mut = mutation_matrix_ref(mu=1,w=L)
    print "p_dkls"
    p_dkls = [inst_dkl(dirichlet_sample(K,alpha),Mut=Mut) for i in trange(N)]
    print "q_dkls"
    q_dkls = [show(inst_dkl(sample_qs(L,desired_entropy,col_tol=0.01),Mut=Mut)) for i in trange(N)]
    print len(q_dkls)
    plt.boxplot([p_dkls,q_dkls])
    plt.ylabel = "Dkl against mutation"
    plt.xlabel(["P","Q"])
    
def sample_qs(L,req_entropy,col_tol = 0.001):
    """Given L and required entropy, sample an independent distribution
    uniformly from the set of psfms meeting those criteria.  Return a
    full "qs" vector of length 4^L
    """
    # first decide how much entropy in each column
    valid = False
    print "assigning entropies to columns"
    while not valid:
        col_ents = np.array(simplex_sample(L))*req_entropy
        if all(col_ents <= 2): #bits/base
            valid = True
    cols = []
    print "assigning column:"
    for j in tqdm(range(L)):
        col_ent = col_ents[j]
        col = sample_with_given_entropy(4,col_ent)
        cols.append(col)
    #return cols
    return qs_from_psfm(cols)

def isocontour_walk(ps0,step_size=10**-2,steps=10,tol=0.01):
    """perform random walk along entropy isocontour"""
    ps = np.copy(ps0)
    h0 = h_np(ps0)
    K = len(ps)
    eta = .005
    print "h0:",h0
    print steps
    for step in xrange(steps):
        # 1) first move along isocontour
        g = grad(ps) # direction of greatest increase in entropy
        hps = h_np(ps)
        eps = np.random.normal(np.log(ps),scale=step_size)
        exps = np.exp(eps)
        us = exps/np.sum(exps)
        #print "us:",us
        vs = us - ps
        vs_proj_grad = vs.dot(g)/g.dot(g)*g
        ps += vs_proj_grad

        # 2) next recover to original entropy value
        unit_vs_proj_grad = vs_proj_grad/np.linalg.norm(vs_proj_grad)
        diff = h0 - h_np(ps)
        #print "diff:",diff
        i = 0
        while abs(diff) > 10**-15:
            #print "diff:",diff
            i += 1
            ps += grad(ps)*diff*eta
            #print ps
            diff = h0 - h_np(ps)
            if i % 1000 == 0:
                print diff
        # print "vs_proj_grad:",vs_proj_grad,(np.linalg.norm(vs_proj_grad)/np.linalg.norm(vs))**2
        # print "vs_proj_isoc:",vs_proj_isoc,(np.linalg.norm(vs_proj_isoc)/np.linalg.norm(vs))**2
        #dp = vs - vs_proj_g
        #ps_new = ps + dp
        #print "ps1:",ps,hps
        d = sqrt(np.sum((ps0-ps)**2))
        if abs(h0-hps) > tol:
            print "tolerance exceeded"
            return
            #raise Exception(ps,g,vs_proj_g)
        #ps = ps_new
        # exps = np.exp(ps + dp)
        # ps = exps/np.sum(exps)
        if step % 100 == 0:
            print step,hps,d,np.sum(ps)
    return ps

def find_alpha(K,entropy,tol_factor=0.01):
    ub = 1/(log2(K)-entropy)
    #print "K:%s,desired entropy:%s, ub:%s" % (K,entropy,ub)
    alpha = bisect_interval(lambda alpha:expected_entropy(K,alpha)-entropy,10**-10,ub)
    return alpha
    
def sample_with_given_entropy(K,entropy,tol_factor=0.01):
    alpha = find_alpha(K,entropy,tol_factor=0.01)
    tol = entropy * tol_factor
    sampled = False
    while not sampled:
            ps = dirichlet_sample(K,alpha)
            if (abs(h_np(ps) - entropy) < tol):
                sampled = True
    return ps

def queify(ps,M=None):
    psfm = marginalize(ps)
    return qs_from_psfm(psfm)

def sample_qs_with_given_entropy(K,entropy,tol_factor=0.01):
    ps = sample_with_given_entropy(K,entropy,tol_factor)
    return qs_from_psfm(marginalize(ps))
    
def qs_from_psfm(psfm):
    qs = np.zeros(int(4**len(psfm)))
    for k,idxs in enumerate(product(*[range(4) for i in range(len(psfm))])):
        #log_q = sum(log(psfm[j][idx]) for j,idx in enumerate(idxs))
        psfm_vals = [(psfm[j][idx]) for j,idx in enumerate(idxs)]
        if all([val > 0 for val in psfm_vals]):
            qs[k] = exp(sum(map(log,psfm_vals)))
        else:
            qs[k] = 0
    return qs
    
def marginalize_ref(ps):
    """turn ps into a psfm"""
    n = len(ps)
    assert log(n,4) == int(log(n,4))
    w = int(log(n,4))
    psfm = [[0 for j in range(4)] for i in range(w)]
    for k,digits in enumerate(product(*[[0,1,2,3] for i in range(w)])):
        for i,d in enumerate(digits):
            psfm[i][d] += ps[k]
    return psfm

def marginalize(ps,M=None):
    n = len(ps)
    w = int(log(n,4))
    if M is None:
        M = marginalization_matrix(w)
    v = M.dot(ps)
    return np.reshape(v,(w,4))

def marginalize_spec(ps):
    vander = np.vander(ps)
    
def make_kmers(w):
    return ("".join(word) for word in (product(*["ACGT" for i in range(w)])))

def marginalization_matrix_ref(w):
    kmers = make_kmers(w)
    M = np.zeros((4*w,4**w))
    for r in tqdm(xrange(4*w),total=4*w):
        psfm_col = r//4
        base = "ACGT"[r%4]
        for c, kmer in enumerate(kmers):
            if kmer[psfm_col] == base:
                M[r][c] = 1
    return M

def marginalization_matrix(w):
    n = 4**w
    M = np.zeros((4*w,n))
    for r in (xrange(4*w)):
        psfm_col = r//4
        run_length = n//(4**(psfm_col+1))
        offset = (r % 4) * run_length
        off_length = run_length * 4
        #print "row: %s, run_length %s, off_length" % (r,run_length,off_length)
        for run in xrange(4**psfm_col):
            start = offset + off_length*run
            stop = start + run_length
            M[r][start:stop] += 1
    return M

def ic(ps,M=None):
    psfm = marginalize(ps,M)
    return sum(2-h(col) for col in psfm)

def total_ic(ps):
    return log2(len(ps)) - h_np(ps)
    
def psfm_entropy(ps,M=None):
    psfm = marginalize(ps,M)
    return sum(h(col) for col in psfm)
    
def total_mi(ps,M=None):
    w = int(log(len(ps),4))
    if M is None:
        M = marginalization_matrix(w)
    psfm = marginalize(ps,M)
    w = len(psfm)
    dkl = 0
    for k,kmer in enumerate(make_kmers(w)):
        p = ps[k]
        logq = sum([log(psfm[i][base_index[c]]) for i,c in enumerate(kmer)])
        dkl += p * (log(p)-logq)
    return dkl/log(2)

def pairwise_mi_ref(ps,M=None):
    w = int(log(len(ps),4))
    dimer_freqs = defaultdict(lambda: np.zeros(16))
    dimer_base_index = {"".join(comb):k for k,comb in enumerate(product("ACGT","ACGT"))}
    psfm = marginalize(ps,M)
    for k,(kmer,p) in enumerate(zip(make_kmers(w),ps)):
        for (i,j) in choose2(range(w)):
            dimer = kmer[i] + kmer[j]
            dimer_freqs[(i,j)][dimer_base_index[dimer]]+=p
    return sum(dimer_freqs[(i,j)][k]*log2(dimer_freqs[(i,j)][k]/(psfm[i][k/4]*psfm[j][k%4]))
                      for k in range(16) for (i,j) in choose2(range(w)))


def plot_ic_vs_pairwise_mi(L,sigmas=interpolate(0.01,10,100),max_h=None,M=None):
    if max_h is None:
        print "generating samples"
        pss = [simplexify_sample(4**L,sigma=sigma)
               for sigma in sigmas]
    else:
        pss = []
        while len(pss) < trials:
            ps = sample(L)
            if h(ps) < max_h:
                pss.append(ps)
                print len(pss)
    print "computing M"
    if M is None:
        M = marginalization_matrix(L)
    print "computing ic"
    total_ics = map(lambda ps:total_ic(ps),tqdm(pss))
    psfm_ics = map(lambda ps:2*L-psfm_entropy(ps,M),tqdm(pss))
    print "computing pairwise mi"
    pair_mis = map(lambda ps:pairwise_mi_ref(ps,M),tqdm(pss))
    print "computing total mi"
    total_mis = map(lambda ps:total_mi(ps,M),tqdm(pss))
    # print "computing columnwise entropies"
    # plt.scatter(hs,hqs)
    plt.scatter(psfm_ics,pair_mis,color='g',label='pair MI')
    plt.scatter(psfm_ics,total_mis,label='total MI')
    plt.scatter(psfm_ics,total_ics,color='r',label='Total IC')
    #plt.plot([0,2*L],[2*L,0])
    #plt.plot([0,2*L],[0,2*L])
    # plt.plot([0,2],[0,4])
    # plt.plot([0,2],[0,2*L])
    # print pearsonr(ics,hs)
    # print spearmanr(ics,hs)
    # plt.plot([0,2*L],[0,2*L])
    # plt.plot(*pl(lambda icp:L*icp+2*(L-L**2),[2*(L-1),2*L]))
    plt.xlabel("PSFM IC")
    plt.ylabel("Bits")
    plt.title("Length=%s" % L)
    plt.legend()
    
def plot_h_vs_ic(L,sigmas=interpolate(0.1,10,100),max_h=None,M=None,xfunc=lambda ps:2*L):
    if max_h is None:
        print "generating samples"
        pss = [simplexify_sample(4**L,sigma=sigma)
               for sigma in tqdm(sigmas)]
    else:
        pss = []
        while len(pss) < trials:
            ps = sample(L)
            if h(ps) < max_h:
                pss.append(ps)
                print len(pss)
    print "computing M"
    if M is None:
        M = marginalization_matrix(L)
    icq_s = map(lambda ps:ic(ps,M),tqdm(pss))
    print "computing entropy"
    icp_s = map(lambda ps:2*L - h_np(ps),tqdm(pss))
    # print "computing total mi"
    # mis = map(lambda ps:total_mi(ps,M),tqdm(pss))
    # print "computing columnwise entropies"
    # hqs = map(lambda ps:psfm_entropy(ps,M),tqdm(pss))
    # plt.scatter(hs,hqs)
    plt.scatter(icp_s,icq_s)
    #plt.plot([0,2*L],[2*L,0])
    #plt.plot([0,2*L],[0,2*L])
    # plt.plot([0,2],[0,4])
    # plt.plot([0,2],[0,2*L])
    # print pearsonr(ics,hs)
    # print spearmanr(ics,hs)
    plt.plot([0,2*L],[0,2*L])
    plt.plot(*pl(lambda icp:L*icp+2*(L-L**2),[2*(L-1),2*L]),color='b')
    plt.xlabel("Distribution IC")
    plt.ylabel("PSFM IC")
    plt.title("Distribution vs. Columnwise IC, Length=%s" % L)
            
def project_to_simplex(v):
    """Project vector v onto probability simplex"""
    ans = v/np.sum(v)
    return ans

def project_to_sphere(v):
    return v/np.linalg.norm(v)
    
def normalize(p):
    """normalize to unit length"""
    return p/((np.linalg.norm(p)))

def simplex_normal(n):
    return normalize(np.array([1]*n))
    
def simplexify(p):
    q = np.exp(p)
    return q/np.sum(q)

def simplexify_sample(k,sigma=1):
    return simplexify(np.random.normal(0,sigma,k))

def dirichlet_sample(K,alpha):
    xs = np.random.gamma(alpha,1,K)
    Z = np.sum(xs)
    return xs/Z

def dirichlet_sample2(k,alpha):
    xs = np.array([random.gammavariate(alpha,1) for i in xrange(k)])
    Z = np.sum(xs)
    return xs/Z

def validate_dirichlet_sample(Ks = [2,5,10,20,50,100],N=1000):
    alphas = [10**i for i in interpolate(-5,0,100)]
    for K in Ks:
        print K
        plt.scatter(*pl(lambda a:mean(h_np(dirichlet_sample(K,a)) for i in xrange(N)),alphas))
        plt.plot(*pl(lambda alpha:expected_entropy(K,alpha=alpha),alphas),label="%s pred" % K)
    plt.xlabel("alpha")
    plt.ylabel("Entropy (bits)")
    plt.semilogx()
    plt.legend()
    
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

def plot_points(ps):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    xs,ys,zs = transpose([[1,0,0],[0,1,0],[0,0,1],[1,0,0]])
    ax.plot(xs,ys,zs)
    pxs,pys,pzs = transpose(ps)
    ax.scatter(pxs,pys,pzs)
        
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

def in_simplex(p):
    return abs(sum(p) - 1) < 10**-16 and np.all(p >= 0)

def propose(p,sigma=0.1):
    """Propose gaussian increment from p in probability simplex"""
    K = len(p)
    v = np.random.normal(0,sigma,K)
    p_new = project_to_simplex(p + v)
    if in_simplex(p_new):
        return p_new
    else:
        return p
        
            
def walk(p0,steps,sigma=0.01):
    p = np.copy(p0)
    ps = [p]
    K = len(p)
    while len(ps) < steps:
        v = np.random.normal(0,sigma,K)
        p_new = project_to_simplex(p + v)
        if in_simplex(p_new):
            ps.append(p_new)
    return ps

def mh_sample(K,iterations=50000):
    p0 = np.array(simplex_sample(K))
    f = lambda p:1/h(p)**K
    proposal = lambda p:propose(p,sigma=1)
    return mh(f,proposal,p0,iterations=iterations)

def average_energy(K,sigma):
    eps = np.random.normal(0,sigma,K)
    return eps.dot(simplexify(eps))

def predict_average_energy(K,sigma):
    return sigma*(1-1/sqrt(K)*exp(sigma**2))
    
def test_predict_average_energy(K,sigma,n=100):
    cis = mean_ci(([average_energy(K,sigma) for i in tqdm(range(n))]))
    print predict_average_energy(K,sigma),cis
    
def partition(K,sigma):
    eps = np.random.normal(0,sigma,K)
    return np.sum(np.exp(eps))

def log_partition(K,sigma):
    return log(partition(K,sigma))

def predict_log_partition_dep(K,sigma):
    Z0,sdZ = predict_partition(K,sigma)
    varZ = sdZ**2
    print Z0,varZ
    zeroth_term = log(K) + sigma**2/2.0
    second_term = -1/2.0 * varZ/(2*(Z0**2))
    print zeroth_term,second_term
    return zeroth_term + second_term

def predict_log_partition(K,sigma):
    return log(K) + (sigma**2)/2.0 - 1/(2*sqrt(K))*(exp(sigma**2) - 1)

def predict_log_partition_0(K,sigma):
    return log(K) + (sigma**2)/2.0
    
def predict_log_partition_normal(K,sigma):
    """The idea here is to taylor expand log(Z) about its mean, but
    supplying the moments through the normal approximation of the
    distribution of Z, which should be good when K is large."""
    Z0,norm_sig = predict_partition(K,sigma)
    print norm_sig/Z0
    return (log(Z0)
            - 1/2.0*(norm_sig/Z0)**2
            - 3/24.0*(norm_sig/Z0)**4
            - 15/720.0*(norm_sig/Z0)**6
            - 105/40320.0*(norm_sig/Z0)**8)
    
def predict_log_partition_normal_series(K,sigma,n):
    Z0,norm_sig = predict_partition(K,sigma)
    a = norm_sig/Z0
    print a
    def facfac(k):
        assert k % 2 == 1
        return reduce(lambda x,y:x*y,range(1,k+1,2))
    return log(Z0) - sum(1/fac(k)*a**k*facfac(k-1) for k in range(2,n+1,2))
    
def predict_partition(K,sigma):
    mean_Z = K*exp(sigma**2/2.0)
    sd_Z = sqrt(K*(exp(sigma**2)-1)*exp(sigma**2))
    return mean_Z,sd_Z

def test_predict_log_partition(sigma=1):
    for i in range(1,9):
 	K = 10**i
        sigma = 3
        ci = mean_ci([log_partition(K,sigma) for j in tqdm(range(10))])
 	print "10^%s" % i,predict_log_partition(K,sigma),ci,1/(2*sqrt(K))
        
def predict_entropy(K,sigma):
    return sigma*predict_average_energy(K,sigma) + predict_log_partition(K,sigma)

def nth_moment(K,sigma,n):
    return K*exp(n**2*sigma**2/2.0)

def count_solutions(K,n):
    """
    Count solutions to k0 + k1 + ... = K
    subject to 0k0 + 1k1 + 2k2 +... = n.
    Second condition implies that solutions are of the form k0,...,kn.
    """
    num_sols = 0
    for ks in (bin_sols(K,n+1)):
        ks = np.array(ks)
        #print ks,K,np.sum(ks),n,np.arange(0,n+1).dot(ks)
        if np.sum(ks) == K and np.arange(0,n+1).dot(ks) == n:
            print ks, "is a solution"
            num_sols += 1
    print "num_sols:",num_sols
    
def bin_sols(n,k):
    """Return ways to put n elements in k bins"""
    assert k > 0
    if k == 1:
        return [[n]]
    else:
        return concat([[[i]+sol for sol in bin_sols(n-i,k-1)]
                       for i in range(n+1)])

def bin_col(ps):
    """Stuff ps into a single column"""
    K = len(ps)
    qs = np.zeros(4)
    # can be made a little more numpythonic...
    for i in range(4):
        qs[i] += np.sum(ps[K/4*i:K/4*(i+1)])
    return qs

def plot_bin_col_ref(w,beta,N):
    K = int(4**w)
    print "pss"
    pss = [simplexify_sample(K,beta) for i in xrange(N)]
    print "hps"
    hps = map(h_np,pss)
    print "hqs"
    hqs = [h_np(bin_col(ps)) for ps in pss]
    plt.scatter(hps,hqs)
    plt.plot([0,2*w],[0,2*w])

def plot_bin_col(w,beta,N,sort=True,color='b'):
    hps = []
    hqs = []
    K = int(4**w)
    if not type(beta) is list:
        beta = [beta]*N
    for i in tqdm(xrange(N),total=N):
        ps = simplexify_sample(K,beta[i])
        if sort:
            ps.sort()
        hps.append(h_np(ps))
        hqs.append(h_np(bin_col(ps)))
    #plt.scatter(hps,hqs,color=color,label="un"*(not sort) + "sorted")
    plt.plot([0,2],[0,2],color='b')
    plt.plot([2,2*w],[2,2],color='b')
    plt.plot([2*w-2,2*w],[0,2],color='b')
    plt.plot([0,2*w-2],[0,0])
    plt.scatter(hps,hqs) #new
    plt.xlabel("H_P")
    plt.ylabel("H_Q")
    plt.title("Full vs. Marginal Entropy (K=4^%s)" % w)
    plt.legend()
    return hps,hqs    

def make_bin_col_plot(filename=None):
    plot_bin_col(10,interpolate(.01,10,100),100,sort=True,color='b')
    plot_bin_col(10,interpolate(.01,10,100),100,sort=False,color='g')
    maybesave(filename)
    
def entropy_spectrum(ps,N):
    """
    Sample psfm entropy spectrum, the set of entropies of psfms
    constructed from permutations of ps
    """
    print "computing perms"
    K = len(ps)
    perms = [np.array([ps[i] for i in knuth_shuffle(K)]) for i in xrange(N)]
    print "computing entropies"
    M = marginalization_matrix(int(log(K,4)))
    return [psfm_entropy(p,M) for p in perms]

def plot_entropy_vs_spectrum(w,beta,n,perms_per_sample):
    """
    
    """
    K = int(4**w)
    M = marginalization_matrix(w)
    print "generating samples"
    pss = [simplexify_sample(K,beta) for i in tqdm(xrange(n))]
    print "main loop"
    for ps in tqdm(pss):
        hp = h_np(ps)
        print hp
        #perms = [np.array([ps[i] for i in knuth_shuffle(K)]) for i in xrange(perms_per_sample)]
        perms = [ps[np.random.permutation(K)] for i in xrange(perms_per_sample)]
        hqs = [psfm_entropy(p,M) for p in perms]
        plt.scatter([hp]*perms_per_sample,hqs)
    plt.plot([0,2*w],[0,2*w])
    
def knuth_shuffle(K):
    """Return Knuth shuffle on range(K)"""
    xs = range(K)
    for i in range(K-1):
        j = random.randrange(K)
        xs[i],xs[j] = xs[j],xs[i]
    return xs

def expected_entropy_ref(K,num_points):
    """Compute expected entropy of uniform prior on probability simplex on
    K elements through numerical integration"""
    def f(x):
        return -x*log(x) if x > 0 else 0
    def integrand(x):
        return f(x)*(1-x)**(K-2)
    return K*(K-1)*mean(integrand(x) for x in interpolate(0,1,num_points))/log(2)

def expected_entropy(K,alpha=1,num_points=1000):
    """Compute expected entropy of uniform prior on probability simplex on
    K elements through numerical integration"""
    def phi(x):
        return -x*log(x) if x > 0 else 0
    def integrand(x):
        if x == 0 or x == 1:
            return 0
        else:
            return phi(x)*x**(alpha-1)*(1-x)**((K-1)*alpha-1)
    #prefactor_ref = K*gamma(K*alpha)/(gamma(alpha)*gamma((K-1)*alpha))
    log_prefactor = log(K) + gammaln(K*alpha) - (gammaln((alpha)) + gammaln((K-1)*alpha))
    try :
        prefactor = exp(log_prefactor)
    except:
        raise Exception("expected entropy failed on:",K,alpha)
    #print prefactor,prefactor_ref
    #prefactor = K/(gamma(alpha))*(alpha*(K-1))**alpha
    #print prefactor,approx_prefactor
    # f = lambda x:(alpha*x*log(x)*K-(x+alpha)*log(x)+x-1)/(x*(x-1)*log(x))
    # f2 = lambda x:(alpha*x*log(x)*K-(x+alpha)*log(x)+x-1)/(x*(x-1)*log(x))
    # xmax = 1#bisect_interval(f,10**-200,1/2.0)
    # prudent_num_points = int(10/xmax)
    # if num_points < prudent_num_points:
    #     num_points = prudent_num_points
    #     print "updated num_points to:",num_points
    #integral_ref = mean(integrand(x) for x in interpolate(0,1,num_points))
    #xs = interpolate(0,1,num_points)
    #print "computing xs"
    xs = [10**x for x in interpolate(-100,0,num_points)]
    #print len(xs)
    #print "min x:",xs[1]
    ys = map(integrand,xs)
    integral_trap = np.trapz(ys,xs)
    return prefactor*integral_trap/log(2)

def expected_entropy_mc(K,alpha=1,n=1000):
    hs = [h_np(dirichlet_sample(K,alpha)) for i in xrange(n)]
    return mean_ci(hs)
    
def mutation_matrix_ref(mu,w,mode="continuous",stochastic=False):
    "mutation matrix for w columns at rate mu"
    K = int(4**w)
    M = np.zeros((K,K))
    if mode == "continuous":
        """M is a transition rate matrix"""
        res = 0
    else:
        """M is a stochastic matrix"""
        res = 1
    for i,kmer_i in enumerate(make_kmers(w)):
        sanity = 0
        if not stochastic or (stochastic and random.random() < mu):
            for j,kmer_j in enumerate(make_kmers(w)):
                distance = w - sum(zipWith(lambda x,y:x==y,kmer_i,kmer_j))
                if distance == 0:
                    M[i][j] = res - 3*w*mu
                elif distance == 1:
                    M[i][j] = mu
                    sanity += 1
        else:
            M[i][i] = res
    return np.transpose(M)
        
def mutate_ref(ps,t,mu=None,M=None):
    if M is None:
        L = int(log(len(ps),4))
        M = mutation_matrix_ref(mu,L)
    return np.squeeze(np.asarray(scipy.linalg.expm(M*t).dot(ps)))

def mutate_inf(ps):
    """Return """
    pass
    
def mutate(ps,t,M):
    eigvals,eigvecs = np.linalg.eig(M)
    Q = np.matrix(eigvecs)
    return np.squeeze(np.asarray((Q.dot(np.diag(np.exp(eigvals*t))).dot(np.linalg.inv(Q)).dot(ps))))

def flow_diagram(L,band,band_tol,sigmas=interpolate(0.0005,0.01,10),steps=1,show=False,mu=10**-3,disp_fact=0.1):
    """Gather trajectories as they pass through ICp band"""
    print "sampling"
    pss = [dirichlet_sample(4**L,sigma) for sigma in tqdm(sigmas)]
    Mut = mutation_matrix_ref(mu,L,mode='discrete')
    Mar = marginalization_matrix(L)
    band_bin = []
    delta_icps = []
    delta_icqs = []
    def mut_dist(ps):
        mut_ps = Mut.dot(ps)
        return h_np(ps) - h_np(mut_ps)
    all_ps = []
    def icp(ps):
        return 2*L-h_np(ps)
    def icq(ps):
        return ic(ps,M=Mar)
    for sigma in tqdm((sigmas)):
        ps = dirichlet_sample(4**L,sigma)
        all_ps.append(ps)
        for i,step in enumerate(xrange(steps)):
            ps = Mut.dot(ps)
            if i % 10 == 0: # collect every tenth step
                all_ps.append(ps.copy())
    cmap = plt.get_cmap('jet')
    divergences = {}
    for p in tqdm(all_ps):
        pp = Mut.dot(p)
        #print p,pp
        x,y = icp(p),icq(p)
        dx, dy = disp_fact*(icp(pp)-x),disp_fact*(icq(pp)-y)
        d = abs(dx) #sqrt(dx**2 + dy**2)
        plt.arrow(x,y,dx,dy,color=cmap(100/0.7*d))
        divergences[(x,y)] = d
    plt.plot([0,2*L-2],[0,0],color='b')
    plt.plot([0,2],[0,2*L],color='b')
    plt.plot([2,2*L],[2*L,2*L],color='b')
    plt.plot([2*L-2,2*L],[0,2*L],color='b')
    plt.xlabel("ICp")
    plt.ylabel("ICq")
    # plt.show()
    return divergences
    # elif show == 1:
    #     for ps in band_bin:
    #         mut_ps = Mut.dot(ps)
    #         print ic
    #         print [total_ic(ps),total_ic(mut_ps)],[ic(ps),ic(mut_ps)]
    #         plt.plot([total_ic(ps),total_ic(mut_ps)],[ic(ps),ic(mut_ps)])
    #     plt.show()
    # else:
    #     return band_bin
    
def mutation_matrix_experiment(L,N):
    mu = 0.01
    time = 1
    print "sampling"
    pss = [dirichlet_sample(4**L,0.1) for i in xrange(N)]
    Mut = mutation_matrix_ref(mu,L)
    M = marginalization_matrix(L)
    print "dkl_mut"
    dkl_muts = [dkl(ps,mutate(ps,time,M=Mut)) for ps in tqdm(pss)]
    print "dkl_qs"
    dkl_qs = [dkl(ps,queify(ps,M)) for ps in pss]
    plt.scatter(dkl_qs,dkl_muts)

def find_theta(ps,qs):
    return acos(ps.dot(qs)/sqrt(ps.dot(ps)*qs.dot(qs)))
    
def test_isocontour_projection_hypothesis():
    """
    Conjecture: entropy isocontours given by projection of centered
    circles on unit sphere onto probability simplex.

    Conclusion: doesn't work
    """
    K = 3
    p = simplexify_sample(K)
    n = project_to_sphere(np.array([1.0/K]*K))
    theta = find_theta(p,n) # in radians
    ps = [np.asarray(general_rotation_matrix(n,theta).dot(p))[0] for theta in interpolate(0,2*pi,100)]

def general_rotation_matrix(u,theta):
    ux,uy,uz = u
    M =  np.matrix([[cos (theta) +ux**2*(1-cos (theta)) , ux *uy* (1-cos (theta)) - uz* sin (theta) , ux* uz* (1-cos (theta)) + uy *sin (theta)],
                    [uy* ux* (1-cos (theta)) + uz* sin (theta) , cos (theta) + uy**2*(1-cos (theta)) , uy *uz* (1-cos (theta)) - ux* sin (theta)],
                    [uz* ux* (1-cos (theta)) - uy *sin (theta) , uz* uy* (1-cos (theta)) + ux *sin (theta) , cos (theta) + uz**2*(1-cos (theta))]])
    return M

def integrate_lagrange_equations(q0=None,K=None,alpha=1,steps=1000,dt=0.01,v=None):
    #lamb = -6
    #q = np.array([0.5,0.25,0.25])
    if q0 is None:
        q = dirichlet_sample(K,alpha)
    else:
        q = q0.copy()
    g = grad(q)
    if v is None:
        ep = np.random.normal(np.log(q),scale=0.01)
        exps = np.exp(ep)
        u = exps/np.sum(exps)
        #print "us:",us
        v = u - q
    p = v - v.dot(g)/g.dot(g)*g
    #p = np.zeros(3)
    H0 = h_np(q)
    def dpdt(q):
        return -(H0-h_np(q))*(np.log(q) - np.sum(np.log(q)/len(q)))
    def dqdt(p):
        return p
    history = [q]
    for i in xrange(steps):
        deltaq = dqdt(p)
        deltap = dpdt(q)
        q,p = q + deltaq*dt,p + deltap*dt
        if i % 1000 == 0:
            print q,p,np.sum(q),h_np(q)
        history.append(q.copy())
    return history

def squeeze_info_into_q(ps):
    hp = h_np(ps)
    L = int(log(len(ps),4))
    psfm = sample_qs(L,hp)
    qs = qs_from_psfm(psfm)
    v = qs - ps

def integrate_lagrange_with_rk4(q0,qp0=None,steps=1000,dt=0.01):
    K = len(q0)
    H0 = h_np(q0)
    if qp0 is None:
        ep = np.random.normal(np.log(q0),scale=0.01)
        exps = np.exp(ep)
        u = exps/np.sum(exps)
        #print "us:",us
        v = u - q0
        g = grad(q0)
        p = v - v.dot(g)/g.dot(g)*g
    qp = np.hstack([q0,p])
    def dpdt(q):
        discrepancy = -(H0-h_np(q))
        entropy_gradient = (np.log(q) - np.sum(np.log(q)/len(q)))
        return discrepancy * entropy_gradient
        
    def dqdt(p):
        return p
    def f(qp):
        q = qp[:K]
        p = qp[K:]
        qp = dqdt(p)
        pp = dpdt(q)
        return np.hstack([qp,pp])
    return [x[:K] for x in rk4(f,qp,steps,dt)]
    
def rk4(f,y0,n,dt=0.01):
    """
    integrate (autonomous ODE) y' = f(y) subject to y(0) = y0 for n steps 
    """
    ys = [y0]
    for i in xrange(n):
        yn = ys[-1]
        k1 = f(yn)
        k2 = f(yn + 1/2.0*k1*dt)
        k3 = f(yn + 1/2.0*k2*dt)
        k4 = f(yn + k3*dt)
        ynp1 = yn + dt/6.0*(k1 + 2*k2 + 2*k3+k4)
        ys.append(ynp1)
    return ys

def one_point_avg_inner(ps,k):
    L = int(log(len(ps),4))
    kmers = list(make_kmers(L))
    s = kmers[k]
    acc = 0
    hits = 0
    for i,kmer in enumerate(kmers):
        if hamming(kmer,s) == 1:
            acc += ps[i]
            hits += 1
    assert hits == 3*L
    return acc/(3*L)

def one_point_avg(ps):
    return np.array([one_point_average(ps,k) for k in range(len(ps))])
    
def num_cols_from_vector(ps):
    L = int(log(len(ps),4))
    return L
    
def ddpl_dHdt_singular(ps,l):
    K = len(ps)
    L = num_cols_from_vector(ps)
    kmers = list(make_kmers(L))
    s = kmers[l]
    term1 = -one_point_avg_inner(ps,l)/(ps[l])
    term2 = -sum(log(ps[k]) for k in range(K) if hamming(kmers[k],s) == 1)/(3*L)
    term3 = log(ps[k]) + 1
    return term1 + term2 + term3

def ddpl_dHdt(ps):
    return np.array([ddpl_dHdt_singular(ps,l) for l in range(len(ps))])

def normal_vector(K):
    return np.ones(K)/sqrt(K)

def vector_projection(ps,qs):
    """return projection of ps onto qs"""
    return qs/np.linalg.norm(qs) * (ps.dot(qs))

def vector_rejection(ps,qs):
    """return rejection of ps from qs"""
    return ps - vector_projection(ps,qs)
    
def simplex_restriction(v):
    """given a vector v, flatten v to simplex, i.e. take vector rejection
    of v upon simplex normal n"""
    K = len(v)
    n = normal_vector(K)
    # = vector_rejection(v,n)
    return v - n.dot(v) * n

def simplex_entropy_gradient(ps):
    """return direction along simplex in which entropy is increasing fastest"""
    return simplex_restriction(-(np.log(ps) + 1))
    
def restrict_to_entropic_isocontour(ps):
    """take projection of ps onto entropy isocontour"""
    simplex_grad = simplex_restriction(-(np.log(ps) + 1))
    return vector_rejection(ps,simplex_grad)

def entropic_isocontour(ps):
    """return a basis of the subspace of the differential entropy isocontour at ps"""
    K = len(ps)
    n = normal_vector(len(ps))
    grad = simplex_entropy_gradient(ps)
    # find basis for subspace orthogonal to these two vectors
    def make_basis_vector(v):
        C = v.dot(n[2:])
        D = v.dot(grad[2:])
        M = np.matrix([n[:2],grad[:2]])
        Minv = np.linalg.inv(M)
        res = np.linalg.inv(M).dot(np.array([-C,-D]))
        v1,v2 = res[0,0],res[0,1]
        basis_vector = np.hstack([v1,v2,v])
        return basis_vector
    vs = [np.array([int(j==i) for j in range(K-2)]) for i in range(K-2)]
    return [make_basis_vector(v) for v in vs]

def normalize1(ps):
    """normalize by L1 norm"""
    return ps/np.sum(ps)


def minimize_dHdt(ps):
    converged = False
    eta = 10**-10
    dt = 10**-6
    hist = []
    while not converged:
        print dHdt(ps),sum(ps),h(ps),min(ps)
        bvs = entropic_isocontour(ps)
        dHp = dHdt(ps)
        grad = [(dHdt(ps + bv*dt)-dHp)/dt for bv in bvs]
        dp = sum(bv*g for (bv,g) in zip(bvs,grad))
        print "sum dp:",sum(dp)
        ps = ps + -dp*eta
        if abs(sum(dp)) < 10**-100:
            converged = True
    return ps

def minimize_dHdt_test():
    L = 4
    K = int(4**L)
    for i in range(10):
        ps = np.array(simplex_sample(K))
        print "marginalizing"
        qs = qs_from_psfm(marginalize(ps))
        print "sampling wtih given entropy"
        rs = sample_with_given_entropy(K,h(qs),tol_factor=10**-6)
        print "minimizing"
        qsp = minimize_dHdt(qs)
        rsp = minimize_dHdt(rs)
        print "qs:",dHdt(qs),dHdt(qsp)
        print "rs:",dHdt(rs),dHdt(rsp)
        
def entropy_hessian_experiment():
    L = 3
    K = int(4**L)
    eps = 10**-6
    ps = np.array(simplex_sample(K))
    qs = qs_from_psfm(marginalize(ps))
    rs = sample_with_given_entropy(K,h(qs),tol_factor=10**-2)
    bvs_p = entropic_isocontour(ps)
    bvs_q = entropic_isocontour(qs)
    bvs_r = entropic_isocontour(rs)
    dHp = dHdt(ps)
    dHps = [dHdt(normalize1(ps + bvp*eps)) for bvp in bvs_p]
    dHq = dHdt(qs)
    dHqs = [dHdt(normalize1(qs + bvq*eps)) for bvq in bvs_q]
    dHr = dHdt(rs)
    dHrs = [dHdt(normalize1(rs + bvr*eps)) for bvr in bvs_r]
    plt.plot(dHps,color='r')
    plt.plot([dHp]*(K-2),color='r',linestyle='--')
    plt.plot(dHqs,color='g')
    plt.plot([dHq]*(K-2),color='g',linestyle='--')
    plt.plot(dHrs,color='b')
    plt.plot([dHr]*(K-2),color='b',linestyle='--')
    plt.show()

def norm_laplacian(L):
    """given num of columns L, return normalized graph laplacian"""
    # L = D - A
    d = 3*L # graph is regular, so degree of each vertex = sqrt(d_i*d_j)
    Lap = -(mutation_matrix_ref(1,L))/(d) # should we take transpose?  doesn't matter because symmetric.
    return Lap

def norm_lap(ps):
    L = num_cols_from_vector(ps)
    Lap = norm_laplacian(L)
    return Lap.dot(ps)
    
def fourier(ps):
    """compute fourier transform of ps, returning a vector of coefficients
    in the basis of eigenvectors of normalized Laplacian:
    if p = \sum_k c_k v_k, return c.
    """
    L = num_cols_from_vector(ps)
    lambdas, V = np.linalg.eig(norm_laplacian(L))
    Vinv = np.linalg.inv(V)
    return Vinv.dot(ps)

def inv_fourier(ps_hat):
    L = num_cols_from_vector(ps)
    lambdas, V = np.linalg.eig(norm_laplacian(L))
    return V.dot(ps_hat)

def fourier_check1():
    L = 2
    K = int(4**L)
    ps = np.array(simplex_sample(K))
    print "L1 error:",L1(ps,inv_fourier(fourier(ps)))
    
def fourier_check2():
    """verify identity: F[L[p]] = Lambdas*ps_hat"""
    L = 2
    K = int(4**L)
    ps = np.array(simplex_sample(K))
    Lap = norm_laplacian(L)
    lambdas, V = np.linalg.eig(norm_laplacian(L))
    ans1 = fourier(Lap.dot(ps))
    ps_hat = fourier(ps)
    ans2 = np.diag(lambdas).dot(ps_hat)
    print "L1 error:",L1(ans1,ans2)

def fourier_check3():
    L = 2
    K = int(4**L)
    ps = np.array(simplex_sample(K))
    Lap = norm_laplacian(L)
    lambdas, V = np.linalg.eig(norm_laplacian(L))
    ps_hat = fourier(ps)
    print L1(ps, sum(ph*np.array(v) for ph,v in zip(ps_hat,transpose(V))))
    
def L1(ps,qs):
    return sum(abs(ps-qs))

def cross_h(ps,qs):
    return -ps.dot(np.log(qs))/log(2)
    
def what_is_fourier_independence():
    L = 2
    K = int(4**L)
    diffs = []
    ps_hats = []
    qs_hats = []
    for i in range(100):
        ps = np.array(simplex_sample(K))
        qs = qs_from_psfm(marginalize(ps))
        ps_hat = fourier(ps)
        qs_hat = fourier(qs)
        print sum(abs(ps_hat)) >= sum(abs(qs_hat)) # always true!
        ps_hats.append(ps_hat)
        qs_hats.append(qs_hat)
    diffs = ([ps_hat - qs_hat for (ps_hat,qs_hat) in zip(ps_hats,qs_hats)])
    plt.plot(transpose(diffs))
    # coefficients 0,3,4,5,9,10,11 are the same, so must control columnwise probabilities (8-1 = 7 df)
    return ps_hats,qs_hats
        #diffs.append((ps - qs))
