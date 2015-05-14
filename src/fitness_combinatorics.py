L = 1
n = 2
# s runs from 0 to n
# j runs from 0 to 4^L
import random
from math import log,exp,pi
from utils import transpose
from utils import choose,report_vars
from itertools import product,chain,combinations
from collections import defaultdict
from tqdm import tqdm

def log_fac(n):
    if n >= 50:
        return stirling(n)
    else:
        return exact_log_fac(n)

def exact_log_fac(n):
    return sum(log(i) for i in range(1,n+1))
    
def stirling(n):
    """approximate log(fac(n))"""
    return n * log(n) - n + 1/2.0*log(2*pi*n)
    
def log_choose(N,k):
    return log_fac(N) - (log_fac(k) + log_fac(N-k))

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def enumerate_genotypes(n,L):
    K = 4**L
    sites = list(product(*[range(K) for i in range(n)]))
    recognizers = powerset(range(K))
    return product(sites,recognizers)

def enumerate_genotypes_by_sr(n,L):
    d = defaultdict(int)
    for motif,recognizer in tqdm(enumerate_genotypes(n,L),total=total_num_genotypes(n,L)):
        r = len(recognizer)
        s = len([s for s in motif if s in recognizer])
        d[(s,r)] += 1
    for s in range(n+1):
        for r in range(4**L+1):
            print s,r,d[(s,r)]
    return d
    
def num_genotypes(s,r,n,L):
    """return number of genotypes with r sites in recognizer, s in motif recognized"""
    K = 4**L
    num_recs_with_r = choose(K,r)
    # perms_of_rec_sites = N_dig_balls_in_k_bins(s,r)
    # perms_of_unrec_sites = N_dig_balls_in_k_bins(n-s,(4**L)-r)
    # num_motifs = perms_of_rec_sites * perms_of_unrec_sites * choose(n,s)
    num_motifs = (r**s)*((K-r)**(n-s))
    ans = num_recs_with_r * num_motifs*choose(n,s) # why this extra factor of choose(n,s)???
    print "s,r,n,L:",s,r,n,L,"num_recs_with_r:",num_recs_with_r,"num_motifs:",num_motifs,"ans:",ans
    return ans

def check_num_genotypes(n,L):
    return sum(num_genotypes(s,r,n,L) for s in range(n+1) for r in range(4**L+1))

def compare_num_genotypes(n,L):
    return check_num_genotypes(n,L),total_num_genotypes(n,L)
    
def N_indig_balls_in_k_bins(N,k):
    return choose(N+k-1,k)

def N_dig_balls_in_k_bins(N,k):
    return choose(N+k-1,N)
    
def total_num_genotypes(n,L):
    num_recognizers = 2**(4**L)
    num_motifs = 4**(n*L)
    return num_motifs * num_recognizers

def log_total_num_genotypes(n,L):
    log_num_recognizers = (4**L)*log(2)
    log_num_motifs = (n*L)*log(4)
    return log_num_motifs + log_num_recognizers
    
def log_num_genotypes_wrong(s,r,n,L):
    log_num_recs_with_r = log_choose(4**L,r)
    log_ways_to_recognize_s_sites_with_r = log_choose(s+r-1,s)
    #print "recs with r:",log_num_recs_with_r,"log_motifs_per_r:",log_ways_to_recognize_s_sites_with_r
    return log_num_recs_with_r + log_ways_to_recognize_s_sites_with_r

def log_num_genotypes(s,r,n,L):
    K = 4**L
    if K - r <= 0 or (K==r and s < n):
        return None
    log_num_recs_with_r = log_choose(K,r)
    log_num_motifs = s*log(r) + (n-s)*log(K-r)
    ans = log_num_recs_with_r + log_num_motifs + log_choose(n,s) # why this extra factor of choose(n,s)???
    return ans
    
def fitness_distribution(n,L):
    K = 4**L
    f_dict = defaultdict(float)
    for s in range(n+1):
        for r in range(1,K+1):
            #print s,r
            ans = (log_num_genotypes(s,r,n,L))
            if ans is None:
                continue
            if random.random() < 0.0001:
                print s,r,ans
            cur = f_dict[s/r]
            if cur == 0:
                f_dict[s/r] = ans
            else:
                f_dict[s/r] = cur + log(1+exp(ans-cur))
    return f_dict

def log_sum(log_xs):
    """Given log_xi = log(Xi), return log(sum(Xs))"""
    return reduce(lambda x,y:x+log(1+exp(y-x)),log_xs)

def log_partition(xs):
    """from the relation Z = sum(exp(x) for x in xs), find y such that y = log(Z), or exp(y) = Z"""
    xm = max(xs)
    return xm + log(sum(exp(x-xm) for x in xs))
    
def plot_fitness_distribution(f_dict):
    plt.scatter(*transpose([(i,v) for i,v in f_dict.items()]))
    
def max_ent_mean_fitness(n,L,f_mean,f_dict=None,eta=10**-2,tol=10**-2,lamb0=1):
    """Given a value of mean fitness, return a maximum entropy distribution over genotype space."""
    if f_dict is None:
        log_N = log_total_num_genotypes(n,L)
        f_freqs = {k:log_pop-log_N for k,log_pop in fitness_frequencies().items()}
    lamb = lamb0
    err = 1
    def log_Z(lamb):
        return log_partition([lamb*f+log_pop for f,log_pop in f_dict.items()])
    while err > tol:
        log_Z = log_Z(lamb)
        f_hat = sum(f*exp(lamb*f+log_pop-log_Z) for f,log_pop in f_dict.items())
        lamb += (f_mean-f_hat)*eta
        err = abs(f_mean-f_hat)
        print f_hat,lamb
    return lamb
        
