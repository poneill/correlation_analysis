"""Sample from linear evo sim using APM, adaptive mh"""

from exact_evo_sim_sampling import sample_from_matrix
from utils import score_seq
from pwm_utils import sample_matrix
from adjacent_pairwise_model import score, prob_site, sample_site, code_from_motif
from adjacent_pairwise_model import best_binder

def fitness(site, matrix, mu, Ne):
    ep = score_seq(matrix, site)
    return (1/(1+exp(ep-mu)))**(Ne-1)

def apm_from_evo_sim(matrix, mu, Ne, iterations=50000):
    L = len(matrix)
    linearized_motif = [sample_from_matrix(matrix,lamb=1) for i in range(100)]
    code = code_from_motif(linearized_motif)
    x = sample_from_matrix(matrix,lamb=1)
    phat = lambda site:fitness(site, matrix, mu, Ne)
    hist = [x]
    accept = 0
    accepts = [0]
    for it in trange(iterations):
        xp = sample_site(code)
        ar = (phat(xp)/phat(x))*(prob_site(x,code)/prob_site(xp,code))
        if random.random() < ar:
            x = xp
            accept += 1
        accepts.append(accept/float(it+1))
        if it % 100 == 0:
            print accept/float(it+1)
        hist.append(x)
        code = code_from_motif(hist[-100:])
    return code,hist,accepts
            
    
def sample_site_apm_ar(matrix, mu, Ne, code):
    best_site = best_binder(code)
    M = prob_site(best_site, code)
    print best_site, M
    while True:
        site = sample_site(code)
        p = fitness(site, matrix, mu, Ne)
        q = prob_site(site, code)
        ar = p/(M*q)
        print site, p,q, M
        assert ar < 1
        if random.random() < ar:
            return site
    
    
