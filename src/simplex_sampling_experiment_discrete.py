"""
Replicate the results of simplex_sampling_experiment using discrete motifs
"""
import itertools
from simplex_sampling_experiment import make_kmers
from utils import random_site,motif_ic,mutate_motif,random_motif,h,pairs,mutate_site,entropy,fac,choose
import random
from tqdm import *
from collections import defaultdict, Counter
from itertools import product
from math import exp,pi,sqrt

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

def motif_with_ent(L,n,ent_min,ent_max):
    ent_dist = entropy_distribution(n)
    min_possible,max_possible = min(ent_dist.keys()),max(ent_dist.keys())
    if ent_min < min_possible * L or ent_max > max_possible * L:
        print "Warning: cannot meet desired entropy"
        return None
    searching_for_cols = True
    print "sampling cols"
    while searching_for_cols:
        desired_col_ents = [random.choice(sample_col_ics) for i in xrange(L)]
        sum_ic = sum(desired_col_ics)
        print sum_ic
        if ic_min <= sum_ic < ic_max:
            searching_for_cols = False
    print "filling cols:",desired_col_ics
    final_cols = []
    for col_ic in tqdm(desired_col_ics):
        searching_for_col = True
        print col_ic
        while searching_for_col:
            prop_col = random_motif(1,n)
            if motif_ic(prop_col,correct=False) == col_ic:
                final_cols.append(prop_col)
                searching_for_col = False
    return ["".join(w) for w in transpose(final_cols)]
    
def interval_subset_sum_ref(xs,k,a,b):
    """return all k-subsets of xs summing to within [a,b)"""
    return [comb for comb in itertools.combinations(xs,k)
            if a<= sum(comb) < b]

def interval_subset_sum(xs,k,a,b):
    if k == 0 and a < 0 < b:
        return []
    elif k == 0 and not (a < 0 < b):
        return [None]
    else:
        return [[x] + interval_subset_sum(xs,k-1,a-x,b-x) for x in xs]
        
    
    
    
def partition_sequence(seq):
    return sorted([seq.count(c) for c in "ACGT"])

def count_partitions(n):
    d = defaultdict(int)
    for kmer in make_kmers(n):
        d[tuple(partition_sequence(kmer))] += 1
    for k,v in sorted(d.items(),key=lambda (k,v):k):
        abstract_sequences = fac(n)/prod(fac(i) for i in k)
        assignments = choose(4,sum(1 for i in k if i > 0))
        foo = [k.count(i) for i in set(k)]
        bar = fac(4)/prod(fac(i) for i in foo)
        print k,v,abstract_sequences*bar#,abstract_sequences,float(v)/abstract_sequences,foo,bar
    return d

def weight_of_partition(part):
    n = sum(part)
    abstract_sequences = fac(n)/prod(fac(i) for i in part)
    foo = [part.count(i) for i in set(part)]
    bar = fac(4)/prod(fac(i) for i in foo)
    return abstract_sequences*bar

def entropy_distribution(n):
    """compute distribution of entropy of sequences of length n, returning
    dictionary of form {h:p}"""
    d = defaultdict(float)
    # construct dictionary this way to avoid collisions if distinct
    # partitions have same entropy values
    for part in restricted_partitions(n):
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

def trace(n,trials=n):
    seq = "A"*n
    entropies = []
    for i in trange(trials):
        entropies.append(entropy(seq,correct=False))
        seq = mutate_site(seq)
    return entropies
