import random
from utils import motif_ic, bisect_interval_noisy,bisect_interval_noisy_ref
from utils import bisect_interval

def pmotif(n,L,p):
    return ["".join([random.choice("CGT") if random.random() < p else "A"
                     for j in range(L)]) for i in range(n)]

def spoof_pmotifs(motif, num_motifs=10,trials=1):
    n = len(motif)
    L = len(motif[0])
    des_ic = motif_ic(motif)
    f = lambda p:-mean(motif_ic(pmotif(n,L,p))-des_ic for i in range(trials))
    lb = 0
    ub = 0.75
    xs = np.linspace(lb,ub,100)
    ys = map(f,xs)
    fhat = kde_regress(xs,ys)
    p = bisect_interval(fhat,lb,ub,verbose=False,tolerance=10**-3)
    return [pmotif(n,L,p) or _ in xrange(num_motifs)]
