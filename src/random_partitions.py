#from sage.all import *
import random
from utils import inverse_cdf_sample, log_fac
from maxent_sampling import log_counts_to_cols, entropy_from_counts
from math import exp, log, ceil

D = {}

def num_parts(N, K, M=None):
    """return number of partitions of M in to at most K parts, where the largest part is no more than M"""
    #print "num_parts:", N, K, M
    if M is None:
        M = N
    if N < 0:
        return 0
    if (N, K, M) in D:
        return D[N, K, M]
    if K == 1:
        ans = 1 if 0 <= N <= M else 0
    else:
        ans = sum(num_parts(N-i, K-1, i) for i in range(M+1))
    D[N, K, M] = ans
    return ans

def random_partition(N, K):
    part = []
    K_ = K
    last = N
    for _ in range(K_):
        #ws = [num_parts(N-i, K-1, i) for i in range(N+1)]
        if K == 1:
            i = N
        else:
            ws = [num_parts(N-i, K-1, i) if i <= last else 0 for i in range(N+1)]
            i = inverse_cdf_sample(range(N+1), ws, normalized=False)
        part.append(i)
        N -= i
        K -= 1
        last = i
        #print part, N, K
    return tuple(part)

def estimate_entropy(N, beta, trials=1000):
    parts = [random_partition(N, 4) for _ in xrange(trials)]
    ents = map(entropy_from_counts, parts)
    log_ws = map(log_counts_to_cols, parts)
    weights = [exp(-beta*ent + log_w) for ent, log_w in zip(ents, log_ws)]
    mean_ent = sum(ent*weight for (ent, weight) in zip(ents, weights))/sum(weights)
    return mean_ent

def rpart(N, beta):
    log_M = log(4) + log_fac(ceil(N/4))
    while True:
        part = random_partition(N, 4)
        log_p = -beta*entropy_from_counts(part) + log_counts_to_cols(part)
        log_r = log(random.random())
        if log_r < log_p - log_M:
            return part

def rcol(N):
    return [random.choice("ACGT") for _ in xrange(N)]

def sample_col(N, beta):
    while True:
        col = rcol(N)
        if random.random() < exp(-beta*(2-motif_ic(col))):
            return col
            
def find_beta(N, des_ic_per_col):
    pre_factor = N
    f = lambda beta:(2-estimate_entropy(N, pre_factor*beta, trials=1000)) - des_ic_per_col
    beta = pre_factor * robbins_munro(f, 1, verbose=True)
