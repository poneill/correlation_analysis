from utils import choose, log_choose, permute, subst, motif_ic
from utils import inverse_cdf_sample, transpose
from math import exp, log
import random

def mutation_motif_with_ic(n,L,desired_ic,epsilon=0.1):
    ringer = ["A"*L]*n
    N = L*n
    ks_rel_motifs = []
    while not ks_rel_motifs:
        motifs = [mutate_motif_k_times(ringer,k) for k in range(N+1)]
        ks_rel_motifs = [(k,motif) for k, motif in enumerate(motifs) if inrange(motif,desired_ic,epsilon)]
    ks, rel_motifs = transpose(ks_rel_motifs)
    ps = [exp(log_prob_motif_with_mismatch(n,L,k)) for k in ks]
    ringified_motif = inverse_cdf_sample(rel_motifs,ks,normalized=False)
    return deringify_motif(ringified_motif)


def prob_motif_with_mismatch(n,L,k):
    N = n*L
    return choose(N,k)*(3/4.0)**k*(1/4.0)**(N-k)

def log_prob_motif_with_mismatch(n,L,k):
    N = n*L
    return log_choose(N,k) + k*log(3/4.0) + (N-k)*log(1/4.0)

def deringify_motif(motif):
    """if motif is constructed by mutating from ringer, transpose bases
    randomly in order to debias it."""
    def ringify_col(col):
        perm = {a:b for (a,b) in zip("ACGT",permute("ACGT"))}
        return [perm[c] for c in col]
    return ["".join(row) for row in transpose(map(ringify_col,transpose(motif)))]
    
def mutate_motif_k_times(motif,k):
    motif_ = motif[:]
    n = len(motif)
    L = len(motif[0])
    N = n * L
    rs = range(N)
    choices = []
    for _ in range(k):
        r = random.choice(rs)
        choices.append(r)
        rs.remove(r)
    for r in choices:
        i = r / L
        j = r % L
        b = motif[i][j]
        new_b = random.choice([c for c in "ACGT" if not c == b])
        motif_[i] = subst(motif_[i],new_b,j)
    return motif_

def inrange(M,I,epsilon):
    return abs(motif_ic(M)-I) < epsilon
