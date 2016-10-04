from pwm_utils import psfm_from_motif
from utils import score_seq, mmap, argmax, argmin
from math import log, exp
import random

def psfm_from_motif_(motif, L, pc=1):
    if motif:
        return psfm_from_motif(motif, pc=1)
    else:
        return [[0.25]*4 for _ in range(L)]
        
def make_clusters_with_k(motif, k):
    print "k:", k
    L = len(motif[0])
    N = float(len(motif))
    clusters = [[] for i in range(k)]
    print "len clusters:",len(clusters)
    for site in motif:
        i = random.randrange(k)
        clusters[i].append(site)
    print "finished initializing"
    pssms = [mmap(log,psfm_from_motif_(cluster, L, pc=1)) for cluster in clusters]
    alphas = [len(cluster)/N for cluster in clusters]
    def log_likelihood():
        return sum(log(sum(alpha*exp(score_seq(pssm, site))
                           for alpha, pssm in zip(alphas, pssms)))
                   for site in motif)
    last_ll = 0
    done_yet = False
    #for i in range(iterations):
    while not done_yet:
        cur_ll = log_likelihood()
        print "log likelihood:", cur_ll
        if last_ll == cur_ll:
            done_yet = True
            break
        else:
            last_ll = cur_ll
        clusters = [[] for i in range(k)]
        for site in motif:
            i = argmax([score_seq(pssm,site) for pssm in pssms])
            clusters[i].append(site)
        pssms = [mmap(log,psfm_from_motif_(cluster, L, pc=1)) for cluster in clusters]
    return clusters, log_likelihood()

def print_motif(motif):
    print "\n".join(motif)

def cluster_motif(motif, max_k=5):
    L = len(motif[0])
    ks = range(1,max_k + 1)
    llss = [[] for k in ks]
    print "ks:", ks
    for k, lls in zip(ks, llss):
        print "loop k:",k
        _, ll = make_clusters_with_k(motif, k)
        lls.append(ll)
    mean_lls = map(mean, llss)
    #plt.plot(range(10), mean_lls)
    aics = [-2*mean_ll + 2*((3*L)*k + k-1) for mean_ll, k in zip (mean_lls, ks)]
    plt.plot(ks, aics)
    best_k, best_aic = min(zip(ks, aics), key=lambda (k,aic):aic)
    clusters, ll = make_clusters_with_k(motif, best_k)
    print "selected %d clusters with ll: %s" % (best_k, ll)
    return clusters
