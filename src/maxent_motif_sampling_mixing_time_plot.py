"""In this script we justify the mixing time estimate:

N^{*}\approx \frac{2\beta\log\delta}{\log(1-e^{-1})}

presented in eq:imh-mixing-time.
"""

from utils import maybesave, motif_ic
from math import exp, log, ceil
from tqdm import *
from uniform_motif_sampling import uniform_motif_with_ic_imh, uniform_motif_imh_tv
from maxent_motif_sampling import find_beta_for_mean_motif_ic
from matplotlib import pyplot as plt

def make_plot(filename=None):
    trials_per_iteration = 3
    iterations = [10**i for i in [0,1,2,3,4]]
    n          = 50
    L          = 10
    desired_ic = 10
    tv         = 0.01
    correction_per_col = 3/(2*log(2)*n)
    desired_ic_for_beta = desired_ic + L * correction_per_col
    beta       = find_beta_for_mean_motif_ic(n,L,desired_ic_for_beta)
    epsilon    = 0.1
    alpha = exp(-2*beta*epsilon)
    opt_iterations = int(ceil(log(tv)/log(1-alpha)))
    opt_epsilon = 1/(2*beta)
    print "optimum iterations:", opt_iterations
    print "optimum epsilon:", opt_epsilon
    results = {}
    for iteration in iterations:
        print "starting on:", iteration
        motifs = [uniform_motif_with_ic_imh(n,L,desired_ic,epsilon=epsilon,beta=beta,iterations=iteration)[-1]
                  for trial in trange(trials_per_iteration)]
        ics = map(motif_ic, motifs)
        results[iteration] = ics
    opt_ics = [uniform_motif_imh_tv(n,L,desired_ic,beta=beta,epsilon=epsilon)
               for trial in range(trials_per_iteration)]
    icss = [results[iteration] for iteration in iterations]
    plt.boxplot(icss + [opt_ics])
    maybesave(filename)
    
    
