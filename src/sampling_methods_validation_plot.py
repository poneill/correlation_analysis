from maxent_motif_sampling import maxent_motif_with_ic, maxent_motifs_with_ic
from uniform_motif_sampling import uniform_motif_with_ic_imh, uniform_motif_with_ic_rw
from uniform_motif_sampling import uniform_motif_accept_reject, uniform_motifs_accept_reject
from uniform_motif_sampling import uniform_motifs_with_ic_rw_harmonic, uniform_motifs_with_ic_rw
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
from utils import motif_ic, motif_gini, maybesave, concat, mmap, total_motif_mi
from utils import pairs, qqplot, fdr, count
import numpy as np
from tqdm import *
from math import exp, log, floor, ceil
from scipy import stats
import time
import cPickle
import random
import pandas as pd

def plot(filename=None):
    L = 10
    n = 50
    iterations = 1000
    trials = 30
    ic_range = np.linspace(2,2*L-1,10)
    maxent_motifs = [[maxent_motif_with_ic(n,L,des_ic) for trial in range(trials)]
                     for des_ic in tqdm(ic_range)]
    uniform_imh_motifs = [[chain[-1]
                           for chain in uniform_motif_with_ic_imh(n,L,des_ic,iterations=None,num_chains=trials)]
                          for des_ic in tqdm(ic_range)]
    uniform_rw_motifs = [[chain[-1]
                                 for chain in uniform_motif_with_ic_rw(n,L,des_ic,iterations=None,num_chains=trials)]
                                for des_ic in ic_range]
    plt.scatter(map(motif_ic,concat(maxent_motifs)), map(motif_gini, concat(maxent_motifs)),label="MaxEnt")
    plt.scatter(map(motif_ic,concat(uniform_imh_motifs)), map(motif_gini, concat(uniform_imh_motifs)),
                label="Uniform (IMH)",color='green',marker='^')
    plt.scatter(map(motif_ic,concat(uniform_rw_motifs)), map(motif_gini, concat(uniform_rw_motifs)),label="Uniform (RW)",
                color='red',marker='s')
    plt.legend()
    plt.xlim(0,2*L)
    plt.ylim(0,1)
    plt.xlabel("Information Content (bits)")
    plt.ylabel("Gini Coefficient")
    ps = []
    for xs,ys,zs in zip(maxent_motifs, uniform_imh_motifs, uniform_rw_motifs):
        statistic, p = stats.kruskal(*mmap(motif_gini,[xs,ys,zs]))
        ps.append(p)
    print "groups unequal?",min(ps) < (0.05/len(ps))
    maybesave(filename)
    return maxent_motifs, uniform_imh_motifs, uniform_rw_motifs

def plot2(filename=None,trials=50,motif_statistic=motif_gini):
    """Compare statistical properties of motifs sampled via MaxEnt, Uniform(AR) algorithms"""
    L = 10
    n = 50
    ic_samples = 10
    ic_range = np.linspace(2,2*L-1,ic_samples)
    # maxent_motifs = [[maxent_motif_with_ic(n,L,des_ic) for trial in range(trials)]
    #                  for des_ic in tqdm(ic_range)]
    # uniform_motifs = [[uniform_motif_accept_reject(n,L,des_ic) for trial in range(trials)]
    #                  for des_ic in tqdm(ic_range)]
    maxent_motifs = []
    uniform_motifs = []
    for des_ic in tqdm(ic_range):
        correction_per_col = 3/(2*log(2)*n)
        desired_ic = des_ic + (L * correction_per_col)
        beta = find_beta_for_mean_motif_ic(n,L,desired_ic)
        maxent_motifs.append([maxent_motif_with_ic(n,L,des_ic,beta=beta) for trial in range(trials)])
        uniform_motifs.append([uniform_motif_accept_reject(n,L,des_ic,beta=beta) for trial in range(trials)])
    plt.scatter(map(motif_ic,concat(maxent_motifs)), map(motif_statistic, concat(maxent_motifs)),label="MaxEnt")
    plt.scatter(map(motif_ic,concat(uniform_motifs)), map(motif_statistic, concat(uniform_motifs)),
                label="Uniform (AR)",color='green',marker='^')
    plt.legend()
    # plt.xlim(0,2*L)
    # plt.ylim(0,1)
    plt.xlabel("Information Content (bits)")
    #plt.ylabel("Gini Coefficient")
    gini_ps = []
    for xs,ys in zip(maxent_motifs, uniform_motifs):
        gini_statistic, gini_p = stats.kruskal(*mmap(motif_statistic,[xs,ys]))
        gini_ps.append(gini_p)
    print "statistics unequal?",min(gini_ps) < (0.05/len(gini_ps))
    maybesave(filename)

def plot3(filename=None, trials=50):
    L = 10
    n = 50
    ics_bp = [0.5, 1, 1.5]
    ME = {}
    TURS = {}
    TURW = {}
    TUGR = {}
    for ic in ics_bp:
        desired_ic = L * ic
        print "starting on ic:", ic
        print "ME"
        ME[ic] = map(motif_ic,maxent_motifs_with_ic(n, L, desired_ic, trials))
        print "TURS"
        TURS[ic] = map(motif_ic,uniform_motifs_accept_reject(n, L, desired_ic, trials))
        print "TURW"
        TURW[ic] = map(motif_ic, uniform_motifs_with_ic_rw_harmonic(n, L, desired_ic, trials))
        print "TUGR"
        TUGR[ic] = map(motif_ic,uniform_motifs_with_ic_rw(n, L, desired_ic, trials))
    plt.violinplot(concat([[ME[ic], TURS[ic], TURW[ic], TUGR[ic]] for ic in ics_bp]))
    plt.annotate("IC = 5 bits",(2.5,17))
    plt.annotate("IC = 10 bits",(6.5,17))
    plt.annotate("IC = 15 bits",(10.5,17))
    plt.xticks(range(1,13),("ME TU(RS) TU(RW) TU(GR) "*3).split())
    plt.xlabel("Method")
    plt.ylabel("IC (bits)")
    maybesave(filename)
    for ic in ics_bp:
        for x,y in pairs("TURS TURW TUGR".split()):
            xs = eval(x)[ic]
            ys = eval(y)[ic]
            print ic,x,y,stats.mannwhitneyu(xs,ys)
            qqplot(xs,ys)
            plt.xlabel("%s IC (bits)" % x)
            plt.ylabel("%s IC (bits)" % y)
            maybesave("%s-vs-%s-IC%s.eps" % (x,y,ic))
    return ME, TURS, TURW, TUGR

def validation_and_runtime_final(filename=None,pickle_filename=None):
    """Figure 1 in Gini paper"""
    trials = 3
    num_motifs = 100
    ns = [20,50,100,200]
    L = 10
    ics_per = [0.5,1,1.5]
    if pickle_filename is None:
        maxent_times = {}
        maxent_motifs = {}
        uniform_times = {}
        uniform_motifs = {}
        for n in ns:
            for ic_per in ics_per:
                print "n,ic:",n,ic_per
                t = time.time()
                maxents = [maxent_motifs_with_ic(n,L,ic_per*L, num_motifs=num_motifs) for _ in xrange(trials)]
                total_time = time.time() - t
                maxent_motifs[(n,ic_per)] = maxents
                print "total time:",total_time
                maxent_times[(n,ic_per)] = total_time/(trials*num_motifs)
                t = time.time()
                uniforms = [uniform_motifs_accept_reject(n,L,ic_per*L, num_motifs=num_motifs) for _ in xrange(trials)]
                total_time = time.time() - t
                print "total time:",total_time
                uniform_motifs[(n,ic_per)] = uniforms
                uniform_times[(n,ic_per)] = total_time/(trials*num_motifs)
        with open("validation_and_runtime_final.pkl",'w') as f:
            cPickle.dump((maxent_motifs,uniform_motifs,maxent_times,uniform_times),f)
    else:
        with open("validation_and_runtime_final.pkl") as f:
            (maxent_motifs,uniform_motifs,maxent_times,uniform_times) = cPickle.load(f)
    
    boxwidth = 9
    boxspace = 9
    boxplot_labels = "abcde".split()
    positions = concat([[n-boxspace,n,n+boxspace] for n in ns])
    num_points = trials * num_motifs
    maxent_legend_offset = 11
    uniform_legend_offset = 0.1
    time_xmin, time_xmax = -3,18
    time_positions = [0,5,10,15]
    #palette = (sns.color_palette('gray', 3))
    palette = sns.cubehelix_palette(4) #
    markers = {i:c for (i,c) in zip(ics_per,'o x ^'.split())}
    linestyles = {i:c for (i,c) in zip(ics_per,'- -- :'.split())}
    sns.set_style('white')
    #sns.set(style='ticks')
    #sns.axes_style(legend.numpoints=3)
    plt.subplot(2,2,1)
    plt.title("MaxEnt Motif IC",fontsize='large')
    
    #maxent_ics = [((map(motif_ic,concat(maxent_motifs[n,ic_per])))) for n in ns for ic_per in ics_per]

    # took this out to deal with NAR's grayscale requirements
    # [plt.scatter([jitter(n,dev=2) + 10*(ics_per.index(ic_per)-1)  for _ in range(num_points)],
    #              [(map(motif_ic,concat(maxent_motifs[n,ic_per])))],
    #              color=colors[ic_per],
    #              s=1,
    #              marker=markers[ic_per],
    #              label=('IC=%d' % (ic_per*L))*(n==20)) for n in ns for ic_per in ics_per]
    #labels = "IC=5 IC=10 IC=15".split()
    # sns.boxplot([(map(lambda m:motif_ic(m) - ic_per * L,concat(maxent_motifs[n,ic_per])))
    #              for n in ns
    #              for ic_per in ics_per],
    #             positions = concat([[n-boxspace,n,n+boxspace] for n in ns]),
    #             widths=boxwidth,sym='.',color=colors,label=boxplot_labels)

    maxent_df = pd.DataFrame([[n, int(ic_per*L), motif_ic(m) - ic_per * L]
                       for n in ns
                       for ic_per in ics_per
                       for m in concat(maxent_motifs[n,ic_per])],
                      columns="n ic resid".split())
    
    sns.boxplot(x="n",y="resid",hue='ic',data=maxent_df,palette=palette)
    plt.legend(title="IC")
    # box = plt.boxplot(maxent_ics,patch_artist=True)
    # colors = concat([['blue','green','red'] for n in ns])
    # for patch, color in zip(box['boxes'], colors): patch.set_facecolor(color)
    #plt.legend(markerscale=5,frameon=True,loc='upper right',fontsize='medium')
    #plt.legend(markerscale=5,frameon=False,fontsize='medium')
    plt.ylim(floor(min(maxent_df['resid']))-1,ceil(max(maxent_df['resid'])) + maxent_legend_offset)
    plt.ylabel("$\Delta$ IC (bits)",fontsize='large')
    #plt.xlim(0,350)
    #plt.xticks(ns,ns)
    plt.xlabel("N",fontsize='large')
    
    plt.subplot(2,2,2)
    plt.title("TU Motif IC",fontsize='large')
    #uniform_ics = [((map(motif_ic,concat(uniform_motifs[n,ic_per])))) for n in ns]
    # [plt.scatter([jitter(n,dev=5)  for _ in range(num_points)],
    #              [(map(motif_ic,concat(uniform_motifs[n,ic_per])))],
    #              color=colors[ic_per],
    #              s=1,
    #              marker=markers[ic_per],
    #              label=('IC=%d' % (ic_per*L))*(n==20)) for n in ns for ic_per in ics_per]
    uniform_df = pd.DataFrame([[n, int(ic_per*L), motif_ic(m) - ic_per * L]
                       for n in ns
                       for ic_per in ics_per
                       for m in concat(uniform_motifs[n,ic_per])],
                      columns="n ic resid".split())
    
    sns.boxplot(x="n",y="resid",hue='ic',data=uniform_df,palette=palette)
    plt.legend(title="IC")
    #plt.legend()
    # box = plt.boxplot(uniform_ics,patch_artist=True)
    # colors = concat([['blue','green','red'] for n in ns])
    # for patch, color in zip(box['boxes'], colors): patch.set_facecolor(color)
    #plt.legend(markerscale=5,frameon=True,loc='lower right',fontsize='medium')
    #plt.legend(markerscale=5,frameon=False,fontsize='medium')
    #plt.ylim(floor(min(uniform_df['resid'])),ceil(max(uniform_df['resid'])) + uniform_legend_offset)
    plt.ylabel("$\Delta$ IC (bits)",fontsize='large')
    #plt.xlim(0,350)
    #plt.xticks(ns,ns)
    plt.xlabel("N",fontsize='large')

    # sns.boxplot([(map(lambda m:motif_ic(m)-ic_per*L,concat(uniform_motifs[n,ic_per])))
    #              for n in ns for ic_per in ics_per],
    #             positions = concat([[n-boxspace,n,n+boxspace] for n in ns]),
    #             widths=boxwidth,sym='.',color=sns.cubehelix_palette(3))
    # # box = plt.boxplot(uniform_ics,patch_artist=True)
    # # colors = concat([['blue','green','red'] for n in ns])
    # # for patch, color in zip(box['boxes'], colors): patch.set_facecolor(color)
    # plt.legend(markerscale=5,frameon=True,loc='lower right',fontsize='medium')
    # #plt.ylim(0,20)
    # plt.ylabel("$\Delta$ IC (bits)",fontsize='large')
    # plt.xticks([2,5,8,11],ns)
    # plt.xlim(0,350)
    # plt.xticks(ns,ns)
    # plt.xlabel("N",fontsize='large')

    plt.subplot(2,2,3)
    plt.title("MaxEnt Runtime",fontsize='large')
    #[plt.plot(ns,[maxent_times[n,ic_per] for n in ns],label='IC:%s' % (ic_per*L),marker='o') for ic_per in ics_per]
    [plt.plot(time_positions,
              [maxent_times[n,ic_per] for n in ns],
              label='%d' % (ic_per*L),
              marker=markers[ic_per], mew=1,
              color=palette[i],
              linestyle=linestyles[ic_per],
              alpha=1) for i,ic_per in enumerate(ics_per)]
    plt.semilogy()
    plt.xlabel("N",fontsize='large')
    plt.xlim(time_xmin,time_xmax)
    plt.xticks(time_positions,ns)
    plt.ylim(10**-4,1)
    plt.ylabel("Time (s)",fontsize='large')
    plt.legend(loc='lower right',fontsize='medium',title="IC")

    plt.subplot(2,2,4)
    plt.title("TU Runtime",fontsize='large')
    #[plt.plot(ns,[uniform_times[n,ic_per] for n in ns],label='IC=%d' % (ic_per*L),marker='o') for ic_per in ics_per]
    [plt.plot(time_positions,
              [uniform_times[n,ic_per] for n in ns],label='%d' % (ic_per*L),
              marker=markers[ic_per],color=palette[i],linestyle=linestyles[ic_per],mew=1)
     for i,ic_per in enumerate(ics_per)]
    plt.semilogy()
    plt.xlim(time_xmin,time_xmax)
    plt.ylim(10**-4,1)
    #plt.xticks([2,5,8,11],ns)
    plt.xticks(time_positions,ns)
    plt.semilogy()
    plt.xlabel("N",fontsize='large')
    plt.ylabel("Time (s)",fontsize='large')
    plt.legend(loc='lower right',fontsize='medium',title="IC")
    plt.tight_layout()
    maybesave(filename)

def jitter(x,dev=0.1):
    return x + (random.random() - 0.5)/0.5 * dev

def validation_ic_vs_gini(filename=None,pickle_filename=None):
    """figure 2 in gini paper"""
    L = 10; N = 50
    ics = np.linspace(0.1,19,100)
    #palette = sns.color_palette('gray',2)
    #sns.set_style('darkgrid')
    sns.set_style('white')
    palette = sns.cubehelix_palette(3)
    if pickle_filename is None:
        maxentses = [maxent_motifs_with_ic(N,L,ic,num_motifs=10) for ic in tqdm(ics)]
        uniformses = [uniform_motifs_accept_reject(N,L,ic,num_motifs=10) for ic in tqdm(ics)]
        maxent_ics = map(motif_ic,concat(maxentses))
        maxent_ginis = map(motif_gini,concat(maxentses))
        uniform_ics = map(motif_ic,concat(uniformses))
        uniform_ginis = map(motif_gini,concat(uniformses))
        with open("validation_ic_vs_gini.pkl",'w') as f:
            cPickle.dump((maxentses,uniformses,maxent_ics,maxent_ginis,uniform_ics,uniform_ginis),f)
    else:
        with open(pickle_filename) as f:
            (maxentses,uniformses,maxent_ics,maxent_ginis,uniform_ics,uniform_ginis) = cPickle.load(f)
    plt.scatter(maxent_ics, maxent_ginis,label='MaxEnt',s=5,color=palette[2])
    plt.scatter(uniform_ics, uniform_ginis,marker='^', label='TU',s=5,color=palette[1])
    plt.xlim(-0.5,2*L+0.5)
    plt.ylim(0,1) # added upon revisions
    plt.xlabel("IC (bits)",fontsize='large')
    plt.ylabel("IGC",fontsize='large')
    plt.legend(markerscale=3,fontsize='large',frameon=True)
    maybesave(filename)

    ps = []
    for maxents, uniforms in zip(maxentses,uniformses):
        maxent_ginis = map(motif_gini,maxents)
        uniform_ginis = map(motif_gini,uniforms)
        stat,p = stats.kruskal(maxent_ginis,uniform_ginis)
        ps.append(p)
    q = fdr(ps)
    print "significant differences:",count(lambda p:p<q,ps)
        
