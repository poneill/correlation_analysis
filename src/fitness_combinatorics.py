L = 5
n = 16
# s runs from 0 to n
# j runs from 0 to 4^L
from utils import choose

def log_fac(n):
    return sum(log(i) for i in range(1,n+1))

def log_choose(N,k):
    return log_fac(N) - (log_fac(k) + log_fac(N-k))
    
def num_genotypes(s,r):
    """return number of genotypes with r sites in recognizer, s in motif recognized"""
    num_recs_with_r = choose(4**L,r)
    ways_to_recognize_s_sites_with_r = choose(s+r-1,s)
    return num_recs_with_r * ways_to_recognize_s_sites_with_r

def log_num_genotypes(s,r):
    log_num_recs_with_r = log_choose(4**L,r)
    log_ways_to_recognize_s_sites_with_r = log_choose(s+r-1,s)
    print "recs with r:",log_num_recs_with_r,"log_motifs_per_r:",log_ways_to_recognize_s_sites_with_r
    return log_num_recs_with_r + log_ways_to_recognize_s_sites_with_r
