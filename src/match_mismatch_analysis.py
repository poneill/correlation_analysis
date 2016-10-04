from estremo_lite import *
from utils import secant_interval,inverse_cdf_sampler,motif_ic,maybesave,choose,log_choose
from utils import motif_gini,total_motif_mi,kmers,bisect_interval,pl, normalize, sd, make_pssm
from tqdm import *
from linear_gaussian_ensemble_gini_analysis import mutate_motif_k_times
import sys
sys.path.append("/home/pat/Dropbox/weighted_ensemble_motif_analysis")
from formosa import maxent_motifs
from we_vs_mh_validation import all_boxplot_comparisons
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from math import log,exp,ceil,sqrt

G = 5*10**6

def excess_mi_experiment(filename=None):
    """Do artificial motifs with linear BEMs show the same patterns of excess MI as biological motifs? (Yes)"""
    n = 10
    L = 10
    G = 1000
    desired_ic = 10
    replicates = 1000
    ics = np.array([mean_ic_from_eps(eps,n,L) for eps in enumerate_eps(n,L)])
    def mean_ic(N):
        ps = sella_hirsch_predictions(n,L,G,N)
        return ics.dot(ps)
    Ne = secant_interval(lambda N:mean_ic(N)-desired_ic,0,2000,tolerance=0.1,verbose=True)# ~= 1525
    ps = sella_hirsch_predictions(n,L,G,Ne)
    sh_sampler = inverse_cdf_sampler(list(enumerate_eps(n,L)),ps)
    sh_motifs = [sample_motif_from_mismatches(sh_sampler(),L) for i in trange(replicates)]
    sh_mean_ic = mean(map(motif_ic,sh_motifs)) # may undershoot desired due to approximation
    maxent_motifs = maxent_sample_motifs_with_ic(n,L,sh_mean_ic,replicates)
    plt.suptitle("Motif Statistics for Match/Mismatch Model vs. MaxEnt Ensembles (n=10,L=10,G=1000)")
    all_boxplot_comparisons([sh_motifs,maxent_motifs],
                            labels=["M/MM","MaxEnt"],
                            plot_titles="IC Gini MI".split(),
                            filename=filename)
# results: lower gini, higher mi

def w(n,L,rho):
    """count number of motifs rho mismatches from ringer"""
    return choose(n*L,rho)*3**rho

def log_w(n,L,rho):
    return log_choose(n*L,rho) + rho*log(3)

def log_w_approx(n,L,rho):
    """use stirling's approximation"""
    return n*L*(log(n*L)-1) - log(fac(rho)) - (n*L-rho)*(log(n*L-rho)-1) + rho*log(3)
    
def counterbalancing_experiment(n,L):
    """how large must nu be to counterbalance entropy?"""
    lamb = 1
    log_f = lambda rho: -lamb*rho
    rhos = range(n*L)
    plt.plot(normalize(map(exp,[log_w(n,L,rho) for rho in rhos])))
    nus = np.linspace(0,10,10)
    for nu in nus:
        plt.plot(normalize(map(exp,[nu*log_f(rho) + log_w(n,L,rho) for rho in rhos])))
    #plt.semilogy()

def sample_log_f(rho,n,L,sigma,G=5*10**6):
    ringer = ["A"*L]*n
    motif = mutate_motif_k_times(ringer,rho)
    mismatches = [L-site.count("A") for site in motif]
    Zb = G*((1+3*exp(-sigma))/4.0)**L
    return -(sigma*sum(mismatches) + n*log(sum(exp(-sigma*mm) for mm in mismatches) + Zb))

def illustrate_rho_partitioning(filename=None,Ne=2):
    n = 10
    L = 10
    sigma = 1
    nu = Ne - 1
    rhos = range(n*L+1)
    log_fs = [predict_log_f(rho,n,L,sigma=sigma) for rho in rhos]
    log_ws = [log_w(n,L,rho) for rho in rhos]
    mean_ics = [mean_ic_from_rho(rho,n,L) for rho in rhos]
    plt.plot(map(exp,log_fs),label="Mean Fitness")
    plt.plot(map(exp,log_ws),label="Degeneracy")
    plt.plot(mean_ics,label = "Mean IC")
    ps = normalize(map(exp,[nu*lf + lw for lf,lw in zip(log_fs,log_ws)]))
    integrand = [ic*p for ic,p in zip(mean_ics,ps)]
    print sum(ps)
    plt.plot(ps,label="Probability")
    plt.plot(integrand, label="Integrand")
    plt.semilogy()
    plt.xlabel("Rho (Mutational distance from optimal genotype)")
    plt.legend()
    maybesave(filename)
    
def compute_Zb(n,L,sigma,G=5*10**6):
    Zb = G*((1+3*exp(-sigma))/4.0)**L
    return Zb

def approx_Zf_ref(rho,n,L,sigma):
    p = rho/float(n*L) # probability of mismatch per base
    q = 1 - p
    return n*sum(exp(-sigma*k)*choose(L,k)*p**k*(q)**(L-k) for k in range(L+1))

def approx_Zf(rho,n,L,sigma):
    p = rho/float(n*L) # probability of mismatch per base
    q = 1 - p
    return n*(q + p*exp(-sigma))**L
    
def predict_log_f(rho,n,L,sigma,G=5*10**6):
    Zb = G*((1+3*exp(-sigma))/4.0)**L
    avg_mismatch = rho/float(n*L)
    #Zf = n*exp(-sigma*avg_mm)   #sum(exp(-sigma*mm) for mm in mismatches)
    Zf = approx_Zf(rho,n,L,sigma)
    #print rho,Zf
    return -(sigma*rho + n*log(Zf + Zb))

def predict_stat(n,L,sigma,Ne,T,G=5*10**6):
    """predict <T> for match-mismatch model"""
    nu = Ne - 1
    rhos = range(n*L+1)
    log_ps = [(nu*predict_log_f(rho,n,L,sigma,G) + log_w(n,L,rho)) for rho in rhos]
    #mean_log_p = mean(log_ps)
    #diff_log_ps = [log_p - mean_log_p for log_p in log_ps]
    #diff_ps = np.array(normalize(map(exp,diff_log_ps)))
    max_log_p = max(log_ps)
    diff_log_ps = [log_p - max_log_p for log_p in log_ps]
    diff_ps = np.array(normalize(map(exp,diff_log_ps)))
    #ps = np.array(normalize([exp((nu*predict_log_f(rho,n,L,sigma,G) + log_w(n,L,rho))) for rho in rhos]))
    #print sum(abs(x - y) for x,y in zip(ps,diff_ps))
    Ts = list(tqdm(np.array(map(T,rhos))))
    return diff_ps.dot(Ts)
    
def sample_log_Z(rho,n,L,sigma,G=5*10**6):
    ringer = ["A"*L]*n
    motif = mutate_motif_k_times(ringer,rho)
    mismatches = [L-site.count("A") for site in motif]
    Zf = sum(exp(-sigma*mm) for mm in mismatches)
    Zb = G*((1+3*exp(-sigma))/4.0)**L
    return log(Zf+Zb)
    
def test_predict_log_f(n,L,sigma,G=5*10**6,trials=1000):
    plt.scatter(*transpose((rho,sample_log_f(rho,n,L,sigma)) for rho in [random.randrange(n*L)
                                                                       for i in trange(trials)]))
    plt.plot([predict_log_f(rho,n,L,sigma) for rho in range(n*L)])

def mean_ic_from_rho(rho,n,L):
    """compute approximate information content of motif with c mismatches in each site"""
        #"""Should depend only on permutations, not on substitutions"""
        # cs counts number of MISMATCHES in each site
    p_mismatch = rho/float(n*L)
    p_match = 1 - p_mismatch
    col_ent = h([p_match,p_mismatch/3.0,p_mismatch/3.0,p_mismatch/3.0])
    return L*(2 - col_ent)

def sample_motif_from_rho(rho,n,L):
    motif = ["A"*L]*n
    return mutate_motif_k_times(motif,rho)
    
def mean_ic_from_rho2(rho,n,L,trials=100):
    ringer = ["A"*L]*n
    motifs = [mutate_motif_k_times(ringer,rho) for i in xrange(trials)]
    return mean(map(motif_ic,motifs))

def mean_gini_from_rho(rho,n,L,trials=100):
    ringer = ["A"*L]*n
    motifs = [mutate_motif_k_times(ringer,rho) for i in xrange(trials)]
    return mean(map(motif_gini,motifs))

def mean_mi_from_rho(rho,n,L,trials=100):
    ringer = ["A"*L]*n
    motifs = [mutate_motif_k_times(ringer,rho) for i in xrange(trials)]
    return mean(map(total_motif_mi,motifs))
    
def mean_occ_from_rho(rho,n,L,sigma,G=5*10**6):
    log_f = predict_log_f(rho,n,L,sigma,G=5*10**6)
    f = exp(log_f)
    mean_occ = f**(1.0/n)
    return n * mean_occ

def occ_matrix_analysis(n=10,occ_matrices=None,filename=None):
    Ls = range(1,30)
    sigmas = (np.linspace(0,20,50))
    Nes = np.linspace(1,5,25)
    num_plots = len(Nes)
    rc = int(ceil(sqrt(num_plots)))
    if occ_matrices is None:
        occ_matrices = [[[predict_stat(n,L,sigma=sigma,Ne=Ne,
                                       T=lambda rho:mean_occ_from_rho(rho,n,L,sigma=sigma))
                          for L in Ls] for sigma in sigmas] for Ne in tqdm(Nes)]
    fig,axes = plt.subplots(nrows=rc,ncols=rc,sharex=True,sharey=True)
    for i,ax in zip(range(len(Nes)),axes.flat):
    #for i,ax in enumerate(Nes):
        im = ax.imshow(np.matrix(occ_matrices[i]).transpose()[::-1],interpolation='none',aspect='auto',vmin=0,vmax=1)
        #ax.set_xticks(Ls)
        #ax.set_xticks()
        #plt.tick_params(axis='x',pad=15)
        #plt.xticks(rotation=90)
        ax.axis('off')
        #ax.set_yticks(sigmas)
    # cax,kw = mpl.colorbar.make_axes([ax for ax in axes.flat])
    # plt.colorbar(im, cax=cax, **kw)
    fig.colorbar(im, ax=axes.ravel().tolist())
    #plt.set_xticks(Ls)
    #maxes = [max(map(max,mat)) for mat in occ_matrices]
    #print maxes
    maybesave(filename)
    return occ_matrices

def gini_matrix_analysis(n=10,occ_matrices=None,filename=None):
    Ls = range(1,30)
    sigmas = (np.linspace(0,20,50))
    Nes = np.linspace(1,5,25)
    num_plots = len(Nes)
    rc = int(ceil(sqrt(num_plots)))
    if occ_matrices is None:
        occ_matrices = [[[predict_stat(n,L,sigma=sigma,Ne=Ne,
                                       T=lambda rho:mean_gini_from_rho(rho,n,L,trials=1))
                          for L in Ls] for sigma in sigmas] for Ne in tqdm(Nes)]
    fig,axes = plt.subplots(nrows=rc,ncols=rc,sharex=True,sharey=True)
    for i,ax in zip(range(len(Nes)),axes.flat):
    #for i,ax in enumerate(Nes):
        im = ax.imshow(np.matrix(occ_matrices[i]).transpose()[::-1],interpolation='none',aspect='auto',vmin=0,vmax=1)
        #ax.set_xticks(Ls)
        #ax.set_xticks()
        #plt.tick_params(axis='x',pad=15)
        #plt.xticks(rotation=90)
        ax.axis('off')
        #ax.set_yticks(sigmas)
    # cax,kw = mpl.colorbar.make_axes([ax for ax in axes.flat])
    # plt.colorbar(im, cax=cax, **kw)
    fig.colorbar(im, ax=axes.ravel().tolist())
    #plt.set_xticks(Ls)
    #maxes = [max(map(max,mat)) for mat in occ_matrices]
    #print maxes
    maybesave(filename)
    return occ_matrices

def log_Zb_chem_pot_dep(L,sigma,G,mu):
    return G * log(1+exp(-mu)*((1+3*exp(-sigma))/4.0)**L)

def log_Zb_chem_pot_ref_dep(L,sigma,G,mu,upto=4):
    sites = kmers(L)
    scores = [sigma*(L-site.count("A")) for site in sites]
    Zs = [sum([exp(-sum(comb)-mu*k) for comb in itertools.combinations(scores,k)]) for k in trange(upto)]
    Z0 = sum(Zs)
    return log(G/(4**L)*Z0)

def total_occupancy(L,sigma,G,mu):
    return G * sum(choose(L,k)*3**k*1**(L-k)*(1.0/(1+exp(k*sigma + mu))) for k in range(L+1))/(4.0**L)

def solve_mu_for_copy_num(L,sigma,G,copy_num):
    f = lambda mu: total_occupancy(L,sigma,G,mu) - copy_num
    return bisect_interval(f,-100,100)

def chem_pot_occ(sigma,mm,mu):
    return 1/(1+exp(sigma*mm+mu))

def crit_L(sigma):
    return -log(G)/log((1+3*exp(-sigma))/4.0)

def crit_sigma(L, c=1):
    return -log((4*(c/float(G))**(1.0/L)-1)/3)

def L_vs_sigma_plot(filename=None,with_bio=False):
    if with_bio:
        tfdf = extract_motif_object_from_tfdf()
        motifs = [getattr(tfdf,tf) for tf in tfdf.tfs]
        Ls = [len(motif[0]) for motif in motifs]
        cs = [len(motif) for motif in motifs]
        ics = [motif_ic(motif) for motif in motifs]
        ic_density = [ic/L for ic,L in zip(ics,Ls)]
        sigmas = [mean(map(sd, make_pssm(motif))) for motif in motifs]
        ginis = [motif_gini(motif,correct=False) for motif in motifs]
        mi_density = [total_motif_mi(motif)/choose(L,2) for motif,L in zip(motifs,Ls)]
    min_sigma = 0.1
    max_sigma = 10
    plt.xlim(0,max_sigma)
    plt.ylim(0,60)
    plt.plot(*pl(crit_L,np.linspace(min_sigma,max_sigma,1000)),label="Binding Transition")
    plt.plot([min_sigma,max_sigma],[log(G,2)/2,log(G,2)/2],linestyle='--',label="Info Theory Threshold")
    # plt.plot(*pl(lambda sigma:log(G)/sigma,np.linspace(min_sigma,max_sigma,1000)),
    #          linestyle='--',label="Zero Discrimination Asymptote")
    if with_bio:
        plt.scatter(sigmas,Ls,label="Biological Motifs")
    plt.xlabel("sigma")
    plt.ylabel("L")
    plt.legend()
    maybesave(filename)

def crit_L2(sigma):
    return log(G)*(1.0/sigma + 1/log(4))

def infer_Ne_from_motif(motif):
    n = len(motif)
    L = len(motif[0])
    bio_ic = motif_ic(motif)
    sigma = 2*mean(map(sd,make_pssm(motif))) # XXX REVSIT THIS ISSUE
    ic_from_Ne = lambda Ne:predict_stat(n,L,sigma,Ne,G=5*10**6,
                                        T=lambda rho:mean_ic_from_rho(rho,n,L))
    Ne = bisect_interval(lambda Ne:ic_from_Ne(Ne)-bio_ic,0.01,5)
    return Ne

def spoof_motif(motif,T):
    n = len(motif)
    L = len(motif[0])
    bio_ic = motif_ic(motif)
    sigma = 2*mean(map(sd,make_pssm(motif))) # XXX REVSIT THIS ISSUE
    ic_from_Ne = lambda Ne:predict_stat(n,L,sigma,Ne,G=5*10**6,
                                        T=lambda rho:mean_ic_from_rho(rho,n,L))
    Ne = bisect_interval(lambda Ne:ic_from_Ne(Ne)-bio_ic,0.01,5)
    return predict_stat(n,L,sigma,Ne,T)

def spoof_gini(motif,trials=1):
    n = len(motif)
    L = len(motif[0])
    bio_ic = motif_ic(motif)
    sigma = 2*mean(map(sd,make_pssm(motif))) # XXX REVSIT THIS ISSUE
    ic_from_Ne = lambda Ne:predict_stat(n,L,sigma,Ne,G=5*10**6,
                                        T=lambda rho:mean_ic_from_rho(rho,n,L))
    Ne = bisect_interval(lambda Ne:ic_from_Ne(Ne)-bio_ic,0.01,5)
    return predict_stat(n,L,sigma,Ne,T=lambda rho:mean_gini_from_rho(rho,n,L,trials=trials))

def spoof_mi(motif,trials=1):
    n = len(motif)
    L = len(motif[0])
    bio_ic = motif_ic(motif)
    sigma = 2*mean(map(sd,make_pssm(motif))) # XXX REVSIT THIS ISSUE
    ic_from_Ne = lambda Ne:predict_stat(n,L,sigma,Ne,G=5*10**6,
                                        T=lambda rho:mean_ic_from_rho(rho,n,L))
    Ne = bisect_interval(lambda Ne:ic_from_Ne(Ne)-bio_ic,0.01,5)
    return predict_stat(n,L,sigma,Ne,T=lambda rho:mean_mi_from_rho(rho,n,L,trials=trials))

def infer_all_Nes():
    Nes = []
    for tf in Escherichia_coli.tfs:
        print tf
        motif = getattr(Escherichia_coli,tf)
        Nes.append(infer_Ne_from_motif(motif))
    return Nes

def mi_pred_vs_obs_plot(filename=None):
    plt.scatter(np.array(pred_mis)/np.array(sizes),np.array(obs_mis)/np.array(sizes))
    plt.xlabel("Predicted MI Density (bits/comparison)")
    plt.ylabel("Observed MI Density (bits/comparison)")
    plt.plot([0,0.3],[0,0.3],linestyle='--')
    plt.text(0.05,0.2,"r^2 = 0.98")
    maybesave(filename)

def gini_pred_vs_obs_plot(filename=None):
    plt.scatter(np.array(pred_ginis),np.array(obs_ginis))
    plt.xlabel("Predicted Gini coeff")
    plt.ylabel("Observed Gini coeff")
    plt.text(0.8,0.2,"r^2 = 0.25")
    plt.plot([0,1],[0,1],lienstyle='--')
    maybesave(filename)
