from utils import mh, motif_ic, total_motif_mi, random_motif, mutate_motif, mean
import random
from math import exp, log
from scipy import polyfit, poly1d

def match_ic_mi(N, L, des_ic, des_mi, iterations=50000, take_stock=None, eta=0.01, alpha=1,beta=0):
    if take_stock is None:
        take_stock = int((N*L)*log(N*L))
    x = random_motif(L, N)
    xs = [None]*iterations
    ics = [0.0]*iterations
    mis = [0.0]*iterations
    alphas = [0.0]*iterations
    betas = [0.0]*iterations
    ic = motif_ic(x)
    mi = total_motif_mi(x)
    accepts = 0
    for i in xrange(iterations):
        # if i == iterations/2:
        #     eta *= 0.1
        xp = mutate_motif(x)
        icp = motif_ic(xp)
        mip = total_motif_mi(xp)
        log_y = (alpha*ic + beta*mi)
        log_yp = (alpha*icp + beta*mip)
        if log(random.random()) < log_yp - log_y:
            accepts += 1
            x = xp
            ic = icp
            mi = mip
        ics[i] = (ic)
        mis[i] = (mi)
        xs[i] = (x)
        #print sum(site.count("A") for site in x)
        
        alphas[i] = (alpha)
        betas[i] = (beta)
        if i > 0 and i % take_stock == 0:
            if i < iterations/10:
                mean_ic = mean(ics[i-take_stock:i])
                mean_mi = mean(mis[i-take_stock:i])
                alpha += eta * (des_ic - mean_ic) * exp(-i/(10*float(iterations)))
                beta  += eta * (des_mi - mean_mi) * exp(-i/(10*float(iterations)))
            else:
                mean_ic = mean(ics[i-take_stock:i])
                mean_mi = mean(mis[i-take_stock:i])
                alpha = poly1d(polyfit(ics[:i], alphas[:i],1))(des_ic)
                beta = poly1d(polyfit(mis[:i], betas[:i],1))(des_mi)
            fmt_string = ("mean ic: % 1.2f, mean mi: % 1.2f, alpha: % 1.2f, beta: % 1.2f" %
                          (mean_ic, mean_mi, alpha, beta))
            print i, "AR:", accepts/(i+1.0), fmt_string
    return xs, ics, mis, alphas, betas
        
