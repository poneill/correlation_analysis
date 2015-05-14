"""
Replicate the results of simplex_sampling_experiment using discrete motifs
"""
import itertools
from simplex_sampling_experiment import make_kmers
from utils import random_site,motif_ic,mutate_motif,random_motif,h,pairs
from utils import mutate_site,entropy,fac,log_fac,choose,transpose,inverse_cdf_sample
import random
from tqdm import *
from collections import defaultdict, Counter
from itertools import product
from math import exp,pi,sqrt,log

def pairwise_motif_mi(motif):
    cols = transpose(motif)
    return sum([mi(col1,col2,correct=False) for (col1,col2) in choose2(cols)])

def degrade_motif(L,n):
    motif = ["A"*L for i in range(n)]
    #motif = random_motif(L,n)
    column_ics = []
    total_ics = []
    for iteration in trange(n*L):
        col_ic = motif_ic(motif,correct=False)
        column_ics.append(col_ic)
        total_ics.append(col_ic + pairwise_motif_mi(motif))
        motif = mutate_motif(motif)
    plt.plot(total_ics,column_ics)
    plt.plot([0,2*L],[0,2*L],linestyle='--',color='b')
    plt.xlabel("Total IC")
    plt.ylabel("Column IC")
    plt.title("n=%s, L=%s" % (n,L))

def best_ic_motif(L,n,trials):
    best_ic = 0
    for i in trange(trials):
        motif = random_motif(L,n)
        cur_ic = motif_ic(motif,correct=False)
        if  cur_ic > best_ic:
            best_motif = motif
    return best_motif

def sample_col_ents(ent_dist,L,a,b):
    ents = ent_dist.keys()
    solutions = interval_subset_sum(ents,L,a,b)
    weights = [prod(ent_dist[s] for s in sol) for sol in solutions]
    return inverse_cdf_sample(solutions,weights,normalized=False)
    
def motif_with_ent(L,n,desired_ent,ent_dist=None):
    """return motif of specified dimensions with desired entropy"""
    # Here we rely on a theorem of v Campenhout and Cover: if X1...XN
    # ~ g(X) and mean(X1...XN) = alpha, then conditional distribution
    # of X1 is exp(lamb*X)g(X), where lamb satisfies <exp(lamb X)*X> =
    # alpha.  Full theorem slightly more general than this; see v
    # Campenhout and Cover 1981 for details.
    if ent_dist is None:
        ent_dist = entropy_distribution(n)
    alpha = desired_ent/float(L)
    lamb = -1.0/alpha
    cond_dict = {x:p*exp(lamb*x) for (x,p) in ent_dist.items()}
    # now normalize
    Z = sum(cond_dict.values())
    cond_dict = {x:p/Z for (x,p) in cond_dict.items()}
    s = sample_prob_dict(cond_dict)
    print L,n,desired_ent,alpha,lamb,sum(x*p for x,p in cond_dict.items()),s
    if L == 1:
        return [s]
    else:
        return [s] + motif_with_ent(L-1,n,desired_ent-s,ent_dist=ent_dist)
    
def motif_with_ent2(L,n,desired_ent,ent_dist=None):
    if ent_dist is None:
        ent_dist = entropy_distribution(n)
    def f(x):
        """probability mass function of x"""
        return ent_dist[x]
    xs = product(*[ent_dist.keys() for i in range(L)])
    def logP(xs,mu):
        return sum(log(f(x)) for x in xs) + (mu*sum(xs))
    ### parametrize mu correctly
    mu = 2
    x0 = [random.choice(ent_dist.keys()) for i in range(L)]
    def proposal(xs):
        xsp = xs[:]
        xsp[random.randrange(L)] = random.choice(ent_dist.keys())
        return xsp
    def target(xs):
        return logP(xs,mu)
    chain = mh(target,proposal,x0,use_log=True)

def gibbs_sample_ent_dist(ent_dict,L,des_ent):
    lamb = 1

def mgf_from_prob_dict(d):
    def f(t):
        return sum(p*exp(x*t) for x,p in d.items())
    return f
    
def motif_mh(L,n,desired_ic):
    x0 = random_motif(L,n)
    def logf(motif,mu):
        return (mu*motif_ic(motif,correct=False))
    return mh()

def motif_hamming_distance(m1,m2):
    return sum(hamming(s1,s2) for (s1,s2) in zip(m1,m2))
    
def sample_prob_dict(d):
    """given a discrete distribution expressed as a dict, sample therefrom"""
    xs,ps = transpose(d.items())
    return inverse_cdf_sample(xs,ps)
    
def interval_subset_sum_ref(xs,k,a,b):
    """return all k-subsets of xs summing to within [a,b)"""
    def ordered(ys):
        return all(y1 <= y2 for y1,y2 in pairs(ys))
    return [list(comb) for comb in itertools.product(*[xs for _ in range(k)])
            if ordered(comb) and a<= sum(comb) < b]

def interval_subset_sumref2(xs,k,a,b):
    """Assume xs is sorted!"""
    min_remaining = xs[0]*k
    max_remaining = xs[-1]*k
    if k == 0 and a <= 0 < b:
        return []
    elif k == 0 and not (a <= 0 < b) or max_remaining < a or min_remaining > b:
        return [None]
    else:
        return [[x] + interval_subset_sum(xs[i:],k-1,a-x,b-x) for i,x in enumerate(xs)]

def interval_subset_sum_(xs,k,a,b,partial_sol=None):
    """Assume xs is sorted!"""
    if partial_sol is None:
        partial_sol=[]
    min_remaining = xs[0]*k
    max_remaining = xs[-1]*k
    #print "calling len(xs)=%s,k=%s,a=%1.2f,b=%1.2f,partial_sum=%1.2f,min=%1.2f,max=%1.2f" % (len(xs),k,a,b,sum(partial_sol),min_remaining,max_remaining)
    if k == 0 and a <= 0 < b:
        #print "found sol"
        return partial_sol
    elif k == 0 and not (a <= 0 < b) or max_remaining < a or min_remaining > b:
        return None
    else:
        return (filter(lambda x:x,[interval_subset_sum_(xs[i:],k-1,a-x,b-x,partial_sol+[x])
                                   for i,x in enumerate(xs)]))


def flatten_sols(sols):
    def atomic_list(xs):
        return all([not type(x) is list for x in xs])
    def list_of_atomic_lists(xxs):
        return all(atomic_list(xs) for xs in xxs)
    if sols is None:
        return None
    elif list_of_atomic_lists(sols):
        return sols
    else:
        return concat([flatten_sols((sol)) for sol in sols])

def interval_subset_sum(xs,k,a,b):
    xs = sorted(xs)
    return flatten_sols(interval_subset_sum_(xs,k,a,b))

def interval_subset_sum_spec(xs,k,a,b):
    xs = sorted(xs)
    mat = [[None for x in xs] for _ in range(k)]
    mat[-1] = [(x,x) for x in xs]
    for i in range(k-2,0-1,-1):
        for j in range(len(xs)):
            a = xs[j] + mat[i+1][j][0]
            b = xs[j] + mat[i+1][-1][1]
            mat[i][j] = (a,b)
    return mat

def interval_subset_sum_mechanical(xs,k,a,b):
    xs = sorted(xs)
    idxs = [0]*k
    parts = []
    
    
def partition_sequence(seq):
    return sorted([seq.count(c) for c in "ACGT"])

def count_partitions_ref(n):
    d = defaultdict(int)
    for kmer in make_kmers(n):
        d[tuple(partition_sequence(kmer))] += 1
    for part,v in sorted(d.items(),key=lambda (k,v):k):
        abstract_sequences = fac(n)/prod(fac(i) for i in part)
        assignments = choose(4,sum(1 for i in part if i > 0))
        foo = [part.count(i) for i in set(part)]
        bar = fac(4)/prod(fac(part) for i in foo)
        print part,v,abstract_sequences*bar#,abstract_sequences,float(v)/abstract_sequences,foo,bar
    return d

def count_partitions(n):
    d = defaultdict(int)
    for kmer in make_kmers(n):
        d[tuple(partition_sequence(kmer))] += 1
    for k,v in sorted(d.items(),key=lambda (k,v):k):
        abstract_sequences = exp(log_fac(n) - sum(log_fac(i) for i in k))
        assignments = choose(4,sum(1 for i in k if i > 0))
        foo = [k.count(i) for i in set(k)]
        bar = fac(4)/prod(fac(i) for i in foo)
        print k,v,abstract_sequences*bar#,abstract_sequences,float(v)/abstract_sequences,foo,bar
    return d

def weight_of_partition(part):
    n = sum(part)
    #abstract_sequences = fac(n)/prod(fac(i) for i in part)
    abstract_sequences = exp(log_fac(n) - sum(log_fac(i) for i in part))
    foo = [part.count(i) for i in set(part)]
    bar = fac(4)/prod(fac(i) for i in foo)
    return abstract_sequences*bar

def entropy_distribution(n):
    """compute distribution of entropy of sequences of length n, returning
    dictionary of form {h:p}"""
    d = defaultdict(float)
    # construct dictionary this way to avoid collisions if distinct
    # partitions have same entropy values
    for part in tqdm(restricted_partitions(n)):
        d[entropy_from_partition(part)] += (weight_of_partition(part)/4**n)
    return dict(d)
    
def enumerate_partitions(n):
    """idiotic way to do this"""
    stems = filter(lambda x:len(x)<=4,accelAsc(n))
    return [tuple(([0]*(4-len(stem))) + stem) for stem in stems]

def restricted_partitions(n):
    for w in range(n/4+1):
        if w * 4 > n:
            continue
        for x in range(w,n+1):
            if x*3 > (n - w):
                continue
            for y in range(x,n+1):
                if y*2 > (n-w-x):
                    continue
                for z in range(y,n+1):
                    if z > (n-w-x-y):
                        continue
                    elif w + x + y + z == n:
                        yield (w,x,y,z)
                        
                    
                    
def entropy_from_partition(part):
    Z = float(sum(part))
    ps = [c/Z for c in part]
    return h(ps)
    
def prod(xs):
    return reduce(lambda x,y:x*y,xs)

def ruleAsc(n):
    """from Jerome Kelleher"""
    a = [0 for i in range(n + 1)]
    k = 1
    a[1] = n
    while k != 0:
        x = a[k - 1] + 1
        y = a[k] - 1
        k -= 1
        while x <= y:
            a[k] = x
            y -= x
            k += 1
        a[k] = x + y
        yield a[:k + 1]

def accelAsc(n):
    """from Jerome Kelleher"""
    a = [0 for i in range(n + 1)]
    k = 1
    y = n - 1
    while k != 0:
        x = a[k - 1] + 1
        k -= 1
        while 2*x <= y:
            a[k] = x
            y -= x
            k += 1
        l = k + 1
        while x <= y:
            a[k] = x
            a[l] = y
            yield a[:k + 2]
            x += 1
            y -= 1
        a[k] = x + y
        y = x + y - 1
        yield a[:k + 1]

def random_partition(n):
    satisfied = False
    while not satisfied:
        part = random_unrestricted_partition(n)
        if len(part) <= 4:
            return [0]*(4-len(part)) + part
    
def random_unrestricted_partition(n):
    x = exp(-pi/sqrt(6*n))
    satisfied = False
    while not satisfied:
        Zs = [rgeom(1-x**i) for i in range(1,n+1)]
        if sum(i*Zi for i,Zi in zip(range(1,n+1),Zs)) == n:
            satisfied = True
    return sum([[i]*Zi for i,Zi in zip(range(1,n+1),Zs)],[])


def rgeom(p):
    i = 0
    while random.random() > p:
        i += 1
    return i

def trace(n,trials=None):
    if trials is None:
        trials = n
    seq = "A"*n
    entropies = []
    for i in trange(trials):
        entropies.append(entropy(seq,correct=False))
        seq = mutate_site(seq)
    return entropies

