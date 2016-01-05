"""In this plot we analyze the run-time characteristics of the three algorithms described in the paper:
    - ME: MaxEnt
    - TU(IMH): Truncated Uniform with Independent Metropolis-Hastings
    - TU(RW):  Truncated Uniform with Random Walk
"""
import time
from maxent_motif_sampling import maxent_motif_with_ic, maxent_motifs_with_ic, find_beta_for_mean_motif_ic
from uniform_motif_sampling import uniform_motif_with_ic_imh, uniform_motif_with_ic_rw
from uniform_motif_sampling import uniform_motif_accept_reject, uniform_motifs_accept_reject, uniform_motifs_with_ic_rw_harmonic, uniform_motifs_with_ic_rw
from utils import transpose, maybesave, concat
from matplotlib import pyplot as plt
from math import log, exp
from tqdm import *
import random
import seaborn as sns

def main_experiment():
    start_time = time.time()
    ns = [10,20,50,100]
    Ls = [5,10,15,20]
    ics_per_base = [0.5,1,1.5]
    trials = 10
    results_dict = {}
    for n in ns:
        for L in Ls:
            for ic_per_base in ics_per_base:
                print "starting on n: %s L: %s IC per base: %s" % (n,L,ic_per_base)
                ic = ic_per_base * L
                t = time.time()
                beta = find_beta_for_mean_motif_ic(n,L,ic)
                maxent_motifs = [maxent_motif_with_ic(n,L,ic,beta=beta) for i in range(trials)]
                maxent_time = time.time() - t
                print "maxent time:",maxent_time
                t = time.time()
                rw_motifs = uniform_motif_with_ic_rw(n,L,ic,num_chains=trials)
                rw_time = time.time() - t
                print "rw time:",rw_time
                t = time.time()
                imh_motifs = uniform_motif_with_ic_imh(n,L,ic,num_chains=trials)
                imh_time = time.time() - t
                print "imh time:",imh_time
                results_dict[(n,L,ic)] = {"me":maxent_time,"imh":imh_time,"rw":rw_time}
    print "total time:", time.time() - start_time
    return results_dict

def main_experiment_me_vs_ar(trials=10):
    """runtime analysis for maxent vs uniform(AR)"""
    start_time = time.time()
    ns = [10,20,50,100]
    Ls = [5,10,15,20]
    ics_per_base = [0.5,1,1.5]
    epsilon = 0.1
    results_dict = {}
    for n in ns:
        for L in Ls:
            for ic_per_base in ics_per_base:
                print "starting on n: %s L: %s IC per base: %s" % (n,L,ic_per_base)
                ic = L * (ic_per_base)
                print "corrected ic:", ic
                t = time.time()
                beta = find_beta_for_mean_motif_ic(n,L,ic)
                print "beta:", beta
                print "e^(2*beta*epsilon):",exp(2*beta*epsilon)
                maxent_motifs = [maxent_motif_with_ic(n,L,ic,beta=beta) for i in trange(trials)]
                maxent_time = time.time() - t
                print "maxent time:",maxent_time
                t = time.time()
                beta = find_beta_for_mean_motif_ic(n,L,ic)
                ar_motifs = [uniform_motif_accept_reject(n,L,ic,beta=beta,epsilon=epsilon) for i in trange(trials)]
                ar_time = time.time() - t
                print "ar time:",ar_time
                results_dict[(n,L,ic)] = {"me":maxent_time,"ar":ar_time}
    print "total time:", time.time() - start_time
    return results_dict
    
def plot_results(results_dict,filename=None):
    mes, imhs, rws = transpose([(d["me"], d["imh"], d["rw"]) for d in results_dict.values()])
    plt.plot(mes,label="ME")
    plt.plot(imhs,label="TU(IMH)")
    plt.plot(rws,label="TU(RW)")
    plt.legend()
    plt.semilogy()
    maybesave(filename)

def plot_results2(results_dict,filename=None):
    mes = []
    imhs = []
    rws = []
    ns = []
    ls = []
    ics = []
    for i,((n,L,ic), d) in enumerate(results_dict.items()):
        me = d["me"]
        imh = d["imh"]
        rw = d["rw"]
        bp = n*L
        x = n*L*ic
        stat = ic * bp*log(bp)
        mes.append(me)
        imhs.append(imh)
        rws.append(rw)
        ns.append(ns)
        ls.append(L)
        ics.append(ic)
        plt.scatter(stat,me,label="MaxEnt"*(i==0))
        plt.scatter(stat,imh,color='g',marker='^',label="Uniform (IMH)"*(i==0))
        plt.scatter(stat,rw,color='r',marker='s',label="Uniform (RW)"*(i==0))
    plt.semilogx()
    plt.semilogy()
    plt.legend()
    return mes,imhs,rws,ns,ls,ics
    maybesave(filename)

def plot_results3(results_dict, num_trials, filename=None):
    plot_max = max(concat(d.values() for d in results_dict.values()))/num_trials
    ns_seen_yet = []
    ics_seen_yet = []
    for (n,L,ic),val in sorted(results_dict.items(),key= lambda ((n,L,ic),val):n):
        ic_per = ic/L
        if not n in ns_seen_yet:
            label="n=%s" % n
            #
            ns_seen_yet.append(n)
        else:
            label = ""
        plt.scatter(val['me']/num_trials,val['ar']/num_trials,color={10:"b",20:"g",50:"r",100:"y"}[n],
                    label=label)
    plt.xlabel("MaxEnt Time (s)")
    plt.ylabel("Truncated Uniform Time (s)")
    plt.plot([0, plot_max],[0, plot_max],linestyle='--')
    plt.title("MaxEnt vs. Truncated Uniform (Rejection Sampling) Runtime")
    plt.legend(loc='lower right')

def main_experiment_final(trials=3):
    """compare ME, TU(RS), TU(RW) TU(GR) runtimes"""
    me_times = []
    ar_times = []
    rw_times = []
    gr_times = []
    L = 10
    ic = 1.5 * L
    for n in [10, 50, 100, 200]:
        print "starting on:",n, L, ic
        print "ME"
        t = time.time()
        me_motifs = maxent_motifs_with_ic(n,L,ic,trials)
        me_time = time.time() - t
        print "me_time:",me_time
        print "AR"
        t = time.time()
        ar_motifs = uniform_motifs_accept_reject(n,L,ic,trials)
        ar_time = time.time() - t
        print "ar_time:",ar_time
        print "RW"
        t = time.time()
        rw_motifs = uniform_motifs_with_ic_rw_harmonic(n, L, ic, trials)
        rw_time = time.time() - t
        print "rw_time:",rw_time
        print "GR"
        t = time.time()
        gr_motifs = uniform_motifs_with_ic_rw(n, L, ic, num_motifs=trials)
        gr_time = time.time() - t
        print "gr_time:",gr_time
        me_times.append(me_time)
        ar_times.append(ar_time)
        rw_times.append(rw_time)
        gr_times.append(gr_time)
    return (me_times, ar_times, rw_times, gr_times)

def plot_results_final((me_times, ar_times, rw_times), num_motifs=3, filename=None):
    for tup in transpose((me_times, ar_times, rw_times)):
        tup = [x/float(num_motifs) for x in tup]
        plt.plot(tup,marker='o',linewidth=0.1)
    plt.semilogy()
    plt.xticks([0,1,2], ["ME", "TURS", "TURW"])
    plt.xlim(-0.1,2.1)
    plt.xlabel("Method")
    plt.ylabel("Time per Sample (s)")
    maybesave(filename)

def plot_results3((me_times, ar_times, rw_times, gr_times),num_motifs=10,filename=None):
    def div(xs):
        return [x/float(num_motifs) for x in xs]
    plt.plot([10,50,100,200],div(me_times),label="ME",marker='o')
    plt.plot([10,50,100,200],div(ar_times),label="TURS",marker='o')
    plt.plot([10,50,100,200],div(rw_times),label="TURW",marker='o')
    plt.plot([10,50,100,200],div(gr_times),label="TUGR",marker='o')
    plt.xlabel("# Sites")
    plt.xlim(0,210)
    plt.ylim(10**-4,10**3)
    plt.ylabel("Time per Motif (s)")
    plt.semilogy()
    plt.legend(loc='lower right')
    maybesave(filename)
