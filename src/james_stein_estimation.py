from pwm_utils import psfm_from_motif
from utils import cv, score_seq, mmap

def js_psfm(motif, target="uniform"):
    assert target in ("uniform", "mean")
    N = len(motif)
    ml_psfm = psfm_from_motif(motif)
    if target == "uniform":
        us = [0.25]*4
    elif target == "mean":
        us = map(mean, transpose(ml_psfm))
    #var_hats = [[p*(1-p)/float(N-1) for p in ps] for ps in ml_psfm]
    lambdas = [(1-sum(p**2 for p in ps))/((N-1) * sum((u-p)**2 for (u, p) in zip(us, ps))) for ps in ml_psfm]
    lambdas = map(lambda x:min(x,1), lambdas)
    print "mean shrinkage:", mean(lambdas)
    js_psfm = [[lamb * 0.25 + (1-lamb) * p for p in ps] for lamb,ps in zip(lambdas,ml_psfm)]
    if any([x <= 0 for x in concat(js_psfm)]):
        print motif
        raise Exception()
    return js_psfm

def cv_experiment(motifs, target='uniform'):
    """see if js_psfm outperforms ml_psfm in 10x cv"""
    all_mls, all_js = [], []
    for motif in motifs:
        ml_lls = []
        js_lls = []
        for train, test in cv(motif):
            ml_mat = mmap(log,psfm_from_motif(train))
            js_mat = mmap(log,js_psfm(train, target=target))
            ml_ll = mean(score_seq(ml_mat,site) for site in test)
            js_ll = mean(score_seq(js_mat,site) for site in test)
            ml_lls.append(ml_ll)
            js_lls.append(js_ll)
        avg_ml_ll, avg_js_ll = mean(ml_lls), mean(js_lls)
        all_mls.append(avg_ml_ll)
        all_js.append(avg_js_ll)
        print avg_ml_ll, avg_js_ll, avg_ml_ll < avg_js_ll
    return all_mls, all_js

"""

Conclusion: The James-Stein estimator offers a reduction in *MSE* over
the ML estimate.  But if we are comparing *likelihood* over CV, then of
course the MLE does better.

"""
