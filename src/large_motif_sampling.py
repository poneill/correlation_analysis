import random
from utils import pairs, normalize, h

def random_integer_partition(N, A=4):
    cuts = [0] + sorted([random.randrange(N+1) for i in range(A-1)]) + [N]
    ns = [y - x for (x,y) in pairs(cuts)]
    if not sum(ns) == N:
        raise Exception(ns)
    return ns

def random_integer_partition_sorted(N, A=4):
    cuts = [0] + sorted([random.randrange(N+1) for i in range(A-1)]) + [N]
    ns = [y - x for (x,y) in pairs(cuts)]
    if not sum(ns) == N:
        raise Exception(ns)
    return sorted(ns)

def random_integer_partition2(N):
    ns = [0]*4
    for i in range(N):
        ns[random.randrange(4)] += 1
    return ns
    
