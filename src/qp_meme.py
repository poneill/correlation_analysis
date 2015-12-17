from gle_gini_analysis import matrix_from
from scipy.optimize import minimize
import numpy as np
    
def qp_meme(motif,p={b:0.25 for b in "ACGT"}):
    """Implements QPMEME from Djordjevic et al. 2002"""
    L = len(motif[0])
    """find E, mu by minimizing variance of energy matrix subject to all sites less than mu"""
    def M_entry(site_a,site_b):
        return sum(((site_a[i]==alpha) - p[alpha])*1.0/p[alpha] *((site_b[i]==alpha) - p[alpha])
                        for i in range(L) for alpha in "ACGT")
    M = np.matrix([[M_entry(site_i,site_j) for site_i in motif] for site_j in motif])
    def cost(gamma):
        return (gamma.transpose().dot(M).dot(gamma) - sum(gamma))[0,0] + 10**6*np.any(gamma < 0)
    gamma = minimize(cost,np.array([1 for _ in motif])).x
    pwm = [[-sum(gamma[a]*p[alpha]*(motif[a][i] == alpha) for a in range(len(motif)))
            for alpha in "ACGT"]
           for i in range(L)]
    return pwm
