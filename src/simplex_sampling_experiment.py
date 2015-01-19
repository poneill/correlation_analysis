"""
What is the most efficient way to pack info into a site: correlation or conservation?

Fundamental conservation: total_mi(ps) + h_np(ps) + ic(ps) = 2*w
"""
from mpl_toolkits.mplot3d import Axes3D
from utils import simplex_sample,h,norm,dot,transpose,log2,interpolate,pl,fac,maybesave,choose2
from itertools import product,permutations
from tqdm import tqdm
import numpy as np
from math import log,exp,sqrt,acos,pi,cos,sin,gamma
from matplotlib import pyplot as plt
from scipy.stats import pearsonr,spearmanr
import sys
import random
from collections import defaultdict
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

def queify(ps,M=None):
    psfm = marginalize(ps)
    qs = np.zeros(len(ps))
    for k,idxs in enumerate(product(*[range(4) for i in range(len(psfm))])):
        log_q = sum(log(psfm[j][idx]) for j,idx in enumerate(idxs))
        qs[k] = exp(log_q)
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

def make_kmers(w):
    return (product(*["ACGT" for i in range(w)]))

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
    for r in tqdm(xrange(4*w),total=4*w):
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
    
def plot_h_vs_ic(L,trials,sigma=1,max_h=None,M=None,xfunc=lambda ps:2*L):
    if max_h is None:
        print "generating samples"
        pss = [simplexify_sample(4**L,sigma=sigma)
               for i in tqdm(range(trials))]
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
    plt.plot(*pl(lambda icp:L*icp+2*(L-L**2),[2*(L-1),2*L]))
    plt.xlabel("Distribution IC")
    plt.ylabel("PSFM IC")
    plt.title("Length=%s" % L)
            
def project_to_simplex(v):
    """Project vector v onto probability simplex"""
    ans = v/np.sum(v)
    return ans

def normalize(p):
    """normalize to unit length"""
    return p/((np.linalg.norm(p)))

def simplexify(p):
    q = np.exp(p)
    return q/np.sum(q)

def simplexify_sample(k,sigma=1):
    return simplexify(np.random.normal(0,sigma,k))

def dirichlet_sample(k,alpha):
    xs = np.random.gamma(alpha,1,k)
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

def expected_entropy(K,alpha=1,num_points=10000):
    """Compute expected entropy of uniform prior on probability simplex on
    K elements through numerical integration"""
    def phi(x):
        return -x*log(x) if x > 0 else 0
    def integrand(x):
        if x == 0 or x == 1:
            return 0
        else:
            return phi(x)*x**(alpha-1)*(1-x)**((K-1)*alpha-1)
    prefactor = K*gamma(K*alpha)/(gamma(alpha)*gamma((K-1)*alpha))
    #prefactor = K/(gamma(alpha))*(alpha*(K-1))**alpha
    #print prefactor,approx_prefactor
    integral = mean(integrand(x) for x in interpolate(0,1,num_points))/log(2)
    return prefactor*integral

def mutation_matrix_ref(mu,w,mode="continuous"):
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
        for j,kmer_j in enumerate(make_kmers(w)):
            distance = w - sum(zipWith(lambda x,y:x==y,kmer_i,kmer_j))
            if distance == 0:
                M[i][j] = res*1 - 3*w*mu
            elif distance == 1:
                M[i][j] = mu
    return M

def mutate_ref(ps,t,mu=None,M=None):
    if M is None:
        L = int(log(len(ps),4))
        M = mutation_matrix_ref(mu,L)
    return scipy.linalg.expm(M*t).dot(ps)

def mutate(ps,t,M):
    eigvals,eigvecs = np.linalg.eig(M)
    Q = np.matrix(eigvecs)
    return Q.dot(np.diag(np.exp(eigvals*t))).dot(np.linalg.inv(Q)).dot(ps)

def flow_diagram(L,sigmas=interpolate(0.0005,0.01,10),steps=100):
    pss = [dirichlet_sample(4**L,sigma) for sigma in sigmas]
    Mut = mutation_matrix_ref(0.01,L,mode='discrete')
    Mar = marginalization_matrix(L)
    delta_icps = []
    delta_icqs = []
    for sigma in sigmas:
        ps = dirichlet_sample(4**L,sigma)
        pps = queify(ps)
        icps = [total_ic(ps)]
        icqs = [2*L-psfm_entropy(ps,Mar)]
        icpps = [total_ic(pps)]
        icqps = [2*L-psfm_entropy(pps,Mar)]
        for step in xrange(steps):
            ps = Mut.dot(ps)
            pps = Mut.dot(pps)
            icps.append(total_ic(ps))
            icqs.append(2*L-psfm_entropy(ps,Mar))
            icpps.append(total_ic(pps))
            icqps.append(2*L-psfm_entropy(pps,Mar))
        plt.plot(icps,icqs,color='r')
        plt.plot(icpps,icqps,color='g')
    plt.plot([0,2*L-2],[0,0],color='b')
    plt.plot([0,2],[0,2*L],color='b')
    plt.plot([2,2*L],[2*L,2*L],color='b')
    plt.plot([2*L-2,2*L],[0,2*L],color='b')
    plt.xlabel("ICp")
    plt.ylabel("ICq")
    plt.show()
        
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
    
print "loaded"
