"""treat sites as elements of product simplex, do Hamiltonian Monte Carlo"""

from utils import simplex_sample

def score_simplex(matrix, site):
    return sum(r*p for row, ps in zip(matrix, site) for r,p in zip(row, ps))

def random_simplex(L):
    return [simplex_sample(4) for i in range(L)]
    
def grad((matrix, mu, Ne), site):
    def phat(site):
        ep = score_simplex(matrix, site)
        return 1/(1+exp(mu-ep))**(Ne-1)
    dp = 0.01
    phat_site = phat(site)
    def perturb(i, j):
        def delta(i, j, ip, jp):
            if i == ip and j == jp:
                return dp
            elif i == ip:
                return -dp/3
            else:
                return 0
        return [[p + delta(i,j, ip, jp) for (jp, p) in enumerate(ps)]
                for (ip, ps) in enumerate(site)]
    return [[(phat(perturb(i,j))-phat_site)/dp for j in range(4)] for i in range(L)]
