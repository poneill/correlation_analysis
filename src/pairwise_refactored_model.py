from utils import random_motif, score_seq, pairs
from pwm_utils import sample_matrix
from chem_pot_model_on_off import approx_mu
from tqdm import *

G = 5*10**6

def random_genotype(n,L,linear_sigma,pairwise_sigma,copies):
    motif = random_motif(L, n)
    pwm = sample_matrix(L, linear_sigma)
    pairwise_weights = [[[random.gauss(0,pairwise_sigma)
                          for i in range(4)]
                         for j in range(4)]
                        for k in range(L-1)]
    return motif, copies, (pwm, pairwise_weights)

def btoi(b):
    return "ACGT".index(b)
    
def energy_score((pwm, pairwise_weights), seq):
    linear_score = score_seq(pwm, seq)
    pairwise_score = sum(weight[btoi(b1)][btoi(b2)]
                         for weight, (b1,b2) in zip(pairwise_weights,pairs(seq)))
    return linear_score + pairwise_score

def compute_Zb(G, (linear_weights, pairwise_weights)):
    pure_pairwise_weights = [[[pw[i][j] + lwi[i] + lwj[j] for j in range(4)] for i in range(4)]
                             for pw, (lwi,lwj)
                             in zip(pairwise_weights, pairs(linear_weights))]
    Ws = [np.matrix([[exp(w[btoi(b1)][btoi(b2)]) for b2 in "ACGT"] for b1 in "ACGT"])
          for w in pure_pairwise_weights]
    return np.array([1,1,1,1]).dot(reduce(lambda x,y:x.dot(y),Ws)).dot(np.array([1,1,1,1]))[0,0]

def test_Zb(G, (linear_weights, pairwise_weights)):
    L = len(linear_weights)
    pred_Zb = G/(4.0**L)*compute_Zb(G, (linear_weights, pairwise_weights))
    obs_Zb = G*mean(exp(energy_score((linear_weights, pairwise_weights), random_site(L))) for i in trange(G))
    return pred_Zb, obs_Zb, (obs_Zb - pred_Zb)/pred_Zb

def fitness((matrix, copies, weights)):
    eps = [energy_score(weights,site) for site in matrix]
