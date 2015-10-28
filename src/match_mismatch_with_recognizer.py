"""Explore a match mismatch model with a recognizer that either
recognizes match/mismatch at each position or does not.

"""
from utils import dot,mutate_motif,mh,random_motif
from math import exp,log

G = 5*10**6
beta = 2

def score_site(tf,site):
    return (-sum((b == "A")*t for t,b in zip(tf,site)))

def compute_Zbar_ref(tf):
    L = len(tf)
    return sum(exp(-beta*score_site(tf,kmer)) for kmer in kmers(L))/(4**L)
    # agrees with compute_Zbar
    
def compute_Zbar(tf):
    L = len(tf)
    k = sum(tf) # num active bases
    # m = match
    return sum(choose(k,m)*(1/4.0)**(m)*(3/4.0)**(k-m)*exp(-beta*(-m)) for m in range(k+1))
    
    
def fitness((tf,motif)):
    L = len(motif[0])
    # +2 kbt mismatch for every non-A if position in tf is active, else nothing
    #eps = [(sum((b != "A")*t for t,b in zip(tf,bs))) for bs in motif]
    eps = [score_site(tf,site) for site in motif]
    fg = sum(exp(-beta*ep) for ep in eps)
    Z = G*compute_Zbar(tf)
    return fg/(fg+Z)

def mutate((tf,motif)):
    L = len(tf)
    if random.random() < 0.5:
        tfp = tf[:]
        i = random.randrange(L)
        tfp[i] = 1 - tf[i]
        return (tfp,motif)
    else:
        return tf,mutate_motif(motif)

def ringer(n,L):
    tf = [1]*L
    motif = ["A"*L]*n
    return tf,motif
    
def bsh_chain(N=1000,iterations=50000):
    L = 20
    n = 10
    log_f = lambda(tf,motif):N*log(fitness((tf,motif)))
    prop = mutate
    x0 = ringer(n,L)#([random.choice([0,1]) for i in range(L)],random_motif(L,n))
    chain = mh(log_f,prop,x0,use_log=True,iterations=iterations)
    return chain
