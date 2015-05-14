import numpy as np
import random
from utils import sorted_indices,rslice,simplex_sample,inverse_cdf_sample,pairs,mean
from math import exp,log
from tqdm import tqdm,trange

def set_vars():
    K = 3
    fs = np.random.random(K)
    mus = random_stochastic_matrix(K)
    A = np.diag(fs).dot(mus)
    v = largest_eigenvector(A)
    
def random_stochastic_matrix(K):
    return np.array([simplex_sample(K) for j in range(K)])

def random_rm_matrix(K,mu):
    fitnesses = np.random.random(K)
    Fs = np.diag(fitnesses)
    A = np.random.random((K,K))
    Mu = (A + A.transpose())/2 # arbitrary symmetric matrix
    #Mu = np.matrix([[mu if j!=i else 1-(K-1)*mu for j in range(K)] for i in range(K)])
    W = Fs.dot(Mu)
    return W

def random_nice_rm_matrix(K,mu):
    fitnesses = np.random.random(K)
    Fs = np.diag(fitnesses)
    Mu = np.matrix([[mu if j!=i else 1-(K-1)*mu for j in range(K)] for i in range(K)])
    W = Fs.dot(Mu)
    return W

def generator_from_rates(A):
    """Return stochastic generator by normalizing all rows to zero"""
    return A - np.diag(np.sum(A,axis=1))
    
def random_simplex_vector(K):
    return np.array(simplex_sample(K))

def random_symmetric_matrix(K):
    A = np.random.random((K,K))
    Mu = (A + A.transpose())/2 # arbitrary symmetric matrix
    return Mu
    
def largest_eigenvector(A,iterations=100):
    K = len(A)
    v = random_simplex_vector(K)
    for iteration in xrange(iterations):
        v = v.dot(A)
        v /= np.sum(v)
    return np.array(v.tolist())

def random_walk(A,iterations=50000):
    K = len(A)
    path = np.zeros(iterations)
    state = random.randrange(K)
    for turn in xrange(iterations):
        path[turn] = state
        state = inverse_cdf_sample(range(K),(A[state,:]).tolist(),normalized=False)
    return path
    
def random_walk_eigenvector(A,iterations=50000):
    K = len(A)
    history = np.zeros(K)
    state = random.randrange(K)
    for _ in xrange(iterations):
        history[state] += 1
        state = inverse_cdf_sample(range(K),(A[state,:]).tolist())
    return history/np.sum(history)

def mh_eigenvector(A,iterations=50000):
    K = len(A)
    history = np.zeros(K)
    i = random.randrange(K)
    for _ in xrange(iterations):
        history[i] += 1
        j = random.randrange(K)
        ar = A[i,j]/A[j,i]
        if random.random() < ar:
            i = j
    return history/np.sum(history)

def mh_eigenvector2(A,iterations=50000):
    K = len(A)
    history = np.zeros(K)
    i = random.randrange(K)
    for _ in xrange(iterations):
        history[i] += 1
        j = random.randrange(K)
        Zi = np.sum(A[i,:])
        Zj = np.sum(A[j,:])
        pij = A[i,j]/Zi
        pji = A[j,i]/Zj
        ar =  pij/pji
        if random.random() < ar:
            i = j
    return history/np.sum(history)

def mh_eigenvector3(fs,mus,iterations=50000):
    K = len(mus)
    print K
    hist = np.zeros(K)
    i = random.randrange(K)
    for _ in xrange(iterations):
        hist[i] += 1
        trans_rates = mus[i,:]
        j = inverse_cdf_sample(range(K),trans_rates,normalized=False)
        if random.random() < fs[j]/fs[i]:
            i = j
    return hist/np.sum(hist)

def moran(fs,mus,n,t):
    """do moran process of n individuals for t generations"""
    K = len(fs)
    pop = np.zeros(K)
    for i in xrange(n):
        pop[random.randrange(K)] += 1
    for _ in trange(t):
        #print "starting:",pop
        b = inverse_cdf_sample(range(K),fs*pop,normalized=False)
        m = inverse_cdf_sample(range(K),mus[b,:],normalized=False)
        d = inverse_cdf_sample(range(K),pop,normalized=False)
        #print "b:",b,"d:",d
        pop[d] -= 1
        pop[m] += 1
        #print pop
    return pop/np.sum(pop)

def trace_lineage(idx,pop_hist):
    lineage = [idx]
    while idx in pop_hist:
        typ,idx = pop_hist[idx]
        lineage.append(idx) 
    return lineage
    
def find_ancestor(pop,pop_hist):
    lineages = [set(trace_lineage(idx,pop_hist)) for (idx,typ) in pop]
    common_ancestry = reduce(lambda x,y:x.intersection(y),lineages)
    #print common_ancestry
    return max(common_ancestry)

def moran_ancestor(fs,mus,n,t):
    K = len(fs)
    pop = [(i,random.randrange(K)) for i in range(1,n+1)]
    next_idx = n+1
    pop_hist = {idx:(typ,0) for (idx,typ) in pop}
    ancestor_history = np.zeros(K)
    for time in trange(t):
        bidx,btype = inverse_cdf_sample(pop,[fs[typ] for idx,typ in pop],normalized=False)
        cidx = next_idx
        next_idx += 1
        ctype = inverse_cdf_sample(range(K),mus[btype,:],normalized=False)
        pop_hist[cidx] = (ctype,bidx)
        pop[random.randrange(n)] = (cidx,ctype)
        anc_idx = find_ancestor(pop,pop_hist)
        if anc_idx > 0:
            anc_typ,_ = pop_hist[anc_idx]
            print time,anc_idx,anc_typ if anc_idx > 0 else None
            ancestor_history[anc_typ] += 1
    return ancestor_history/np.sum(ancestor_history)

def single_moran(fs,mus,iterations=50000):
    K = len(mus)
    print K
    hist = np.zeros(K)
    i = random.randrange(K)
    for _ in trange(iterations):
        hist[i] += 1
        trans_rates = mus[i,:]
        j = inverse_cdf_sample(range(K),trans_rates,normalized=False)
        if random.random() < 1/2.0:#fs[j]/(fs[i]+fs[j]):
            i = j
    return hist/np.sum(hist)

def qs_sampling(fs,mus,iterations=50000):
    K = len(mus)
    hist = np.zeros(K)
    i = random.randrange(K)
    for _ in xrange(iterations):
        hist[i] += 1
        trans_probs = mus[i,:]
        fi = fs[i]
        offspring = [scipy.stats.poisson(mu*fi).rvs() for mu in trans_probs]
        offspring[i] += 1
        i = inverse_cdf_sample(range(K),offspring,normalized=False)
    return hist/np.sum(hist)

    
def boltzmann_distribution(A,beta=1):
    """Recover boltzmann distribution from A"""
    v = largest_eigenvector(A,iterations=1000)
    phi = np.sum(v)
    inferred_fs = np.sum(A,axis=1).transpose().tolist()[0]
    eta = 0.01
    epsilon = 0.01
    beta = 1
    rec_phi = sum

def moran_tracking(fs,mus,tf,init_state=0):
    """Track a single trajectory of a population until tf.  Assume mus are
    probabilities, i.e. row stochastic"""
    t = 0
    K = len(fs)
    x = np.zeros(K)
    i = init_state
    x[i] = 1
    hist = []
    while t < tf:
        if False:#random.random() < 0.0001:
            print t/tf,x
        hist.append((t,np.copy(x)))
        lifetime = random.expovariate(fs[i])
        t += lifetime
        j = inverse_cdf_sample(range(K),mus[i,:])
        if random.random() < fs[j]/(fs[j] + fs[i]): # track new lineage with probability 1/2
            x[i] = 0
            x[j] = 1
            i = j
        assert np.sum(x) == 1
    last_state = hist[-1][1]
    hist.append((tf,last_state))
    return hist
    
        
        
    
def gillespie(A,x0,tf,death=0,malthus=0,fs=None):
    """Given an array of fitnesses a mutation matrix and an initial
    population vector, sample the system until time t.

    Note: agrees with largest_eigenvector :)
    """
    K = len(A)
    t = 0
    x = np.array(x0)
    A = np.array(A)
    hist = []
    hist = [(t,np.copy(x))]
    while t < tf:
        birth_rates = x.dot(A)
        death_rates = x * death
        rates = np.hstack([birth_rates,death_rates])
        total_rate = 1.0/np.sum(rates)
        dt = random.expovariate(total_rate)
        selection = inverse_cdf_sample(range(2*K),rates,normalized=False)
        t += dt
        if selection < K: # birth reaction
            x[selection] += 1
        else:
            selection -= K
            x[selection] -= 1
        if malthus and np.sum(x) > malthus:
            choice = inverse_cdf_sample(range(K),x,normalized=False)
            x[choice] -= 1
        if t < tf:
            hist.append((t,np.copy(x)))
        if random.random() < 0.0001:
            print t,x,x/float(np.sum(x))
    last_state = hist[-1][1]
    hist.append((tf,last_state))
    return hist
        
def average_gillespie_hist(hist):
    avg = np.zeros(len(hist[0][1]))
    t,state = hist[0]
    for new_t,new_state in hist[1:]:
        delta_t = new_t - t
        #print state,delta_t
        assert delta_t >= 0,delta_t
        avg += state * delta_t
        t,state = new_t,new_state
    return avg/np.sum(avg)

def naive_gillespie_hist(hist):
    tot = sum([arr for (t,arr) in hist])
    return tot/np.sum(tot)

def infinitesimal_generator_from_rate_matrix(A):
    """given rates for CTMC, return inifinitesimal generator i.e. coefficients of ODEs"""
    K = len(A[0,:])
    return np.array([[A[i,j]/sum(A[i,:]) if i != j else -sum(A[i,:]) for j in range(3)] for i in range(3)])
    
def explore_dft():
    """illustrate an example of the detailed fluctuation theorem"""
    K = 3
    fs = np.random.random(K)
    mus = random_stochastic_matrix(K)
    A = np.diag(fs).dot(mus)
    M = infinitesimal_generator_from_rate_matrix(A)
    v = largest_eigenvector(A)
    #path = [0] + [random.randrange(K) for i in xrange(100000)]
    N = 10000
    acc = 0
    for trial in xrange(N):
        path = random_walk(A)
        sigma = sum(log(M[j,i]/M[i,j]) for i,j in pairs(path))
        i_final = -log(v[path[-1]])
        acc += exp(sigma-i_final)
        print trial,-sigma,i_final,acc/float(trial+1)
    return acc/float(N)
# A = random_stochastic_matrix(3)
# evals,evecs = np.linalg.eig(A.transpose())
# js = sorted_indices(evals)[::-1]
# evals,evecs = rslice(evals,js),rslice(evecs,js)
# assert evals[0] == max(evals)
# lamb = evals[0]
# v = evecs[0]
