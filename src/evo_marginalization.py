"""sample evo sims by re-marginalizing PSFM as proposal distribution"""
from utils import score_seq, dnorm, argmin, argmax
from pwm_utils import site_sigma_from_matrix
from scipy import integrate

def site_mu_from_matrix(matrix):
    return sum(map(mean, matrix))
    
def remarginalize(matrix, mu, Ne):
    def phat(site):
        ep = score_seq(matix, site)
        return 1/(1+exp(ep-mu))**(Ne-1)
    def f(ep):
        return 1/(1+exp(ep-mu))**(Ne-1)
    site_mu = site_mu_from_matrix(matrix)
    site_sigma = site_sigma_from_matrix(matrix)
    ep_min = sum(map(argmin, matrix))
    ep_max = sum(map(argmax, matrix))
    def marginal(i, j):
        red_matrix = [row for jp, row in enumerate(matrix) if not j == jp]
        red_site_mu = site_mu_from_matrix(red_matrix)
        red_site_sigma = site_sigma_from_matrix(red_matrix)
        ep = matrix[i][j]
        nom = integrate.quad(lambda ep_rest:f(ep + ep_rest)*dnorm(ep_rest, red_site_mu, red_site_sigma),
                             ep_min, ep_max)
        denom = integrate.quad(lambda ep_rest:f(ep_rest)*dnorm(ep_rest, site_mu, site_sigma), ep_min, ep_max)

def sample_motif_reestimate(matrix, mu, Ne, N):
    best_site = "".join("ACGT"[argmin(col)] for col in matrix)
    psfm = [best_site for i in range(N)]
    
