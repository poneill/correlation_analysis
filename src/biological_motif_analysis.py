from motifs import *
from maxent_motif_sampling import maxent_motifs_with_ic, find_beta_for_mean_motif_ic
from uniform_motif_sampling import uniform_motifs_accept_reject as uniform_motifs_with_ic
from utils import sample_until,maybesave,concat,motif_ic,motif_gini,total_motif_mi,choose2, choose
from utils import transpose,pl,bs,random_motif,inverse_cdf_sample,mmap,mean,mode
from utils import mi_permute,dna_mi,log2, transpose
from matplotlib import pyplot as plt
from collections import defaultdict
import sys
sys.path.append("/home/pat/motifs")
#from parse_merged_data import tfdf
from parse_tfbs_data import tfdf
from scipy import stats, polyfit, poly1d
import numpy as np
from scipy.stats import pearsonr, spearmanr
import time
import numpy as np
from chem_pot_model_on_off import spoof_motif
from exact_evo_sim_sampling import spoof_motif_cftp
from motif_profile import find_pattern

def coverage_region(xs,alpha=0.95):
    n = len(xs)
    xs_ = sorted(xs)
    start = int((1-alpha)/2 * n)
    stop = n - start
    return xs_[start],xs_[stop]

def val_in_coverage(t,xs):
    lo,hi = coverage_region(xs)
    if t < lo:
        return "L"
    elif t > hi:
        return "H"
    else:
        return "-"
        
def biological_experiment(replicates=1000):
    delta_ic = 0.1
    results_dict = defaultdict(lambda:defaultdict(dict))
    for tf_idx,tf in enumerate(Escherichia_coli.tfs):
        print tf,"(%s/%s)" % (tf_idx,len(Escherichia_coli.tfs))
        bio_motif = getattr(Escherichia_coli,tf)
        n,L = motif_dimensions(bio_motif)
        bio_ic = motif_ic(bio_motif)
        bio_gini = motif_gini(bio_motif)
        bio_mi = total_motif_mi(bio_motif)
        results_dict[tf]["bio"]["motif_ic"] = bio_ic
        results_dict[tf]["bio"]["motif_gini"] = bio_gini
        results_dict[tf]["bio"]["total_motif_mi"] = bio_mi
        beta = find_beta_for_mean_motif_ic(n,L,bio_ic)
        maxent = maxent_motifs_with_ic(n,L,bio_ic,replicates)
        #maxent_truncated = maxent_truncated_sample_motifs_with_ic(n,L,bio_ic,delta_ic,replicates,beta=beta)
        uniform = uniform_motifs_with_ic(n,L,bio_ic,delta_ic,replicates)
        #chain_spoofs = chain_sample_motifs_with_ic(n,L,bio_ic,delta_ic,replicates,beta=beta)
        #for spoof_name in "maxent maxent_truncated envelope".split():
        for spoof_name in "maxent uniform".split():
            spoofs = eval(spoof_name)
            for motif_statname in "motif_ic motif_gini total_motif_mi".split():
                motif_stat = eval(motif_statname)
                results_dict[tf][spoof_name][motif_statname] = map(motif_stat,spoofs)
        #all_spoofs = [maxent_spoofs,maxent_truncated_spoofs,envelope_spoofs]#,chain_spoofs]
        # print "IC:",bio_ic,map(lambda xs:val_in_coverage(bio_ic,xs),mmap(motif_ic,all_spoofs))
        # print "Gini:",bio_gini,map(lambda xs:val_in_coverage(bio_gini,xs),mmap(motif_gini,all_spoofs))
        # print "MI:",bio_mi,map(lambda xs:val_in_coverage(bio_mi,xs),mmap(total_motif_mi,all_spoofs))
    return results_dict

def check_tfdf():
    genomes = set(tfdf['genome_accession'])
    for genome_idx,genome in enumerate(genomes):
        tfs = set(tfdf[tfdf['genome_accession'] == genome]['TF'])
        for tf_idx,tf in enumerate(tfs):
            print genome,tf
            if not type(tf) is str:
                print "Exception"

def extract_tfdf_sites(genome,tf):
    sites = tfdf[tfdf['genome_accession'] == genome][tfdf['TF']==tf]['site_sequence']
    # convert to list matrix form,remove nans
    bio_motif_all_lens = [site for site in sites if type(site) is str] 
    modal_length = mode(map(len,bio_motif_all_lens))
    bio_motif = filter(lambda x:len(x)==modal_length,bio_motif_all_lens)
    if len(bio_motif) != len(bio_motif_all_lens):
        print "removed", len(bio_motif_all_lens) - len(bio_motif),"of",
    return bio_motif

def extract_motif_object_from_tfdf():
    obj = Organism()
    genomes = set(tfdf['genome_accession'])
    setattr(obj,"tfs",[])
    for genome in genomes:
        tfs = set(tfdf[tfdf['genome_accession'] == genome]['TF'])
        for tf in tfs:
            print tf,genome
            if not type(tf) is str:
                continue
            tf_name = genome + "_" + tf
            sites = extract_tfdf_sites(genome,tf)
            if len(sites) >= 10 and motif_ic(sites) > 5:
                setattr(obj,tf_name,sites)
                obj.tfs.append(tf_name)
    return obj

#tfdf = extract_motif_object_from_tfdf()
bio_motifs = [getattr(tfdf,tf) for tf in tfdf.tfs]

def tfdf_experiment(replicates=1000,delta_ic=0.1,tolerance=10**-5):
    genomes = set(tfdf['genome_accession'])
    results_dict = defaultdict(lambda:defaultdict(dict))
    for genome_idx,genome in enumerate(genomes):
        print "genome:",genome, genome_idx,len(genomes)
        tfs = set(tfdf[tfdf['genome_accession'] == genome]['TF'])
        for tf_idx,tf in enumerate(tfs):
            if not type(tf) is str:
                continue
            print "tf:",tf,tf_idx,len(tfs)
            print genome,tf
            bio_motif = extract_tfdf_sites(genome,tf)
            if len(bio_motif) < 10:
                print "skipping:"
                continue
            tf_name = genome + "_" + tf
            n,L = motif_dimensions(bio_motif)
            print "dimensions:",n,L
            bio_ic = motif_ic(bio_motif)
            bio_gini = motif_gini(bio_motif)
            bio_mi = total_motif_mi(bio_motif)
            results_dict[tf_name]["bio"]["motif_ic"] = bio_ic
            results_dict[tf_name]["bio"]["motif_gini"] = bio_gini
            results_dict[tf_name]["bio"]["total_motif_mi"] = bio_mi
            correction_per_col = 3/(2*log(2)*n)
            desired_ic = bio_ic + L * correction_per_col
            t = time.time()
            beta = find_beta_for_mean_motif_ic(n,L,desired_ic,tolerance=tolerance)
            beta_time = time.time() - t
            print "beta, time:",beta, beta_time
            print "maxent sampling"
            maxent = maxent_motifs_with_ic(n,L,bio_ic,replicates,beta=beta)
            print "uniform sampling"
            uniform = uniform_motifs_with_ic(n,L,bio_ic,replicates,epsilon=delta_ic,beta=beta)
            #print "envelope sampling"
            #envelope = envelope_sample_motifs_with_ic(n,L,bio_ic,delta_ic,replicates,beta=beta)
            for spoof_name in "maxent uniform".split():
                spoofs = eval(spoof_name)
                for motif_statname in "motif_ic motif_gini total_motif_mi".split():
                    print "recording results for:",spoof_name,motif_statname
                    motif_stat = eval(motif_statname)
                    results_dict[tf_name][spoof_name][motif_statname] = map(motif_stat,spoofs)
    return results_dict

def evo_sim_experiment(filename=None):
    """compare bio motifs to on-off evosims"""
    tfdf = extract_motif_object_from_tfdf()
    bio_motifs = [getattr(tfdf,tf) for tf in tfdf.tfs]
    evosims = [spoof_motif(motif,num_motifs=100,Ne_tol=10**-4)
               for motif in tqdm(bio_motifs)]
    evo_ics = [mean(map(motif_ic,sm)) for sm in tqdm(evosims)]
    evo_ginis = [mean(map(motif_gini,sm)) for sm in tqdm(evosims)]
    evo_mis = [mean(map(total_motif_mi,sm)) for sm in tqdm(evosims)]
    plt.subplot(1,3,1)
    scatter(map(motif_ic,bio_motifs),evo_ics)
    plt.title("Motif IC (bits)")
    plt.xlabel("Biological Value")
    plt.ylabel("Simulated Value")
    plt.subplot(1,3,2)
    scatter(map(motif_gini,bio_motifs),
            evo_ginis)
    plt.title("Motif Gini Coefficient")
    plt.xlabel("Biological Value")
    plt.ylabel("Simulated Value")
    plt.subplot(1,3,3)
    scatter(map(total_motif_mi,bio_motifs),
            evo_mis)
    plt.xlabel("Biological Value")
    plt.ylabel("Simulated Value")
    plt.title("Pairwise Motif MI (bits)")
    plt.loglog()
    plt.tight_layout()
    plt.savefig(filename)
    return evosims

def plot_results_dict_gini_vs_ic(results_dict,filename=None):
    for i,k in enumerate(results_dict):
        g1,g2,tf = k.split("_")
        genome = g1 + "_" + g2
        bio_motif = extract_tfdf_sites(genome,tf)
        bio_ic = motif_ic(bio_motif)
        bio_gini = motif_gini(bio_motif)
        d = results_dict[k]
        plt.scatter(bio_ic,bio_gini,color='b',label="Bio"*(i==0))
        plt.scatter(mean(d['maxent']['motif_ic']),mean(d['maxent']['motif_gini']),color='g',label='ME'*(i==0))
        plt.scatter(mean(d['uniform']['motif_ic']),mean(d['uniform']['motif_gini']),color='r',label="TURS"*(i==0))
    plt.xlabel("IC (bits)")
    plt.ylabel("Gini Coefficient")
    plt.legend()
    maybesave(filename)

def plot_results_dict_gini_qq(results_dict,filename=None):
    bios = []
    maxents = []
    uniforms = []
    for i,k in enumerate(results_dict):
        g1,g2,tf = k.split("_")
        genome = g1 + "_" + g2
        bio_motif = extract_tfdf_sites(genome,tf)
        bio_ic = motif_ic(bio_motif)
        bio_gini = motif_gini(bio_motif)
        d = results_dict[k]
        bios.append(bio_gini)
        maxents.append(mean(d['maxent']['motif_gini']))
        uniforms.append(mean(d['uniform']['motif_gini']))
    plt.scatter(bios,maxents,label='ME')
    plt.scatter(bios,uniforms,label='TURS',color='g')
    minval = min(bios+maxents+uniforms)
    maxval = max(bios+maxents+uniforms)
    plt.plot([minval,maxval],[minval,maxval],linestyle='--')
    plt.xlabel("Observed Gini Coefficient")
    plt.ylabel("Mean Sampled Gini Coefficient")
    plt.legend(loc='upper left')
    print "bio vs maxent:",pearsonr(bios,maxents)
    print "bio vs uniform:",pearsonr(bios,uniforms)
    maybesave(filename)




    
def interpret_biological_experiment(results_dict):
    spoof_names = sorted([k for k in results_dict.values()[0] if not k == 'bio'])
    stat_names = sorted([k for k in results_dict.values()[0]['bio']])
    tf_names = sorted(results_dict.keys(),key=lambda tf:results_dict[tf]["bio"]["motif_ic"])
    def order_tfs_by(stat_name):
        return sorted(results_dict.keys(),key=lambda tf:results_dict[tf]["bio"][stat_name])
    def bio_stats(fname,order_by_stat="motif_ic"):
        ordered_tfs = order_tfs_by(order_by_stat)
        return [results_dict[tf]["bio"][fname] for tf in ordered_tfs for _ in range(len(spoof_names))]
    def all_spoof_stats(fname,order_by_stat="motif_ic"):
        ordered_tfs = order_tfs_by(order_by_stat)
        return concat([[results_dict[tf][spoof_name][fname]
                             for spoof_name in spoof_names] for tf in ordered_tfs])
    for stat_idx,stat_name in enumerate(stat_names):
        plt.subplot(1,len(stat_names),stat_idx+1)
        plt.boxplot(all_spoof_stats(stat_name,order_by_stat=stat_name))
        plt.scatter(range(1,len(bio_stats(stat_name))+1),bio_stats(stat_name,order_by_stat=stat_name),
                    marker='^',color='r')
        plt.title(stat_name)
        plt.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom='off',      # ticks along the bottom edge are off
            top='off',         # ticks along the top edge are off
            labelbottom='off') 
    # for tf in order_tfs_by('motif_ic'):
    #     for stat_name in stat_names:
    #         bio_stat = results_dict[tf]['bio'][stat_name]
    #         votes = [val_in_coverage(bio_stat,results_dict[tf][spoof_name][stat_name])
    #                  for spoof_name in spoof_names]
    #         all_spoofs = concat([results_dict[tf][spoof_name][stat_name]
    #                              for spoof_name in spoof_names])
    #         score = votes.count("H") - votes.count("L")
    #         #n,L = motif_dimensions(getattr(Escherichia_coli,tf))
    #         coverage = tuple(map(fmt,coverage_region(all_spoofs)))
    #         #print tf,"(%s,%s)" % (n,L),":",stat_name,fmt(bio_stat),coverage,votes,score
    # plt.subplot(1,3,1)
    # # bio_ics = [results_dict[tf]["bio"]["motif_ic"] for tf in tf_names]
    # # all_spoof_ics = concat([[results_dict[tf][spoof_name]["motif_ic"]
    # #                          for spoof_name in spoof_names] for tf in tf_names])
    # plt.boxplot(all_spoof_ics)
    # plt.plot(range(1,len(all_spoof_ics)+1),[bio_ic for bio_ic in bio_ics for i in range(len(spoof_names))])

def interpret_biological_experiment2(results_dict,filename=None):
    spoof_names = sorted([k for k in results_dict.values()[0] if not k == 'bio'])
    stat_names = sorted([k for k in results_dict.values()[0]['bio']])
    tf_names = sorted(results_dict.keys(),key=lambda tf:results_dict[tf]["bio"]["motif_ic"])
    def order_tfs_by(stat_name):
        return sorted(results_dict.keys(),key=lambda tf:results_dict[tf]["bio"][stat_name])
    def bio_stats(fname,order_by_stat="motif_ic"):
        ordered_tfs = order_tfs_by(order_by_stat)
        return [results_dict[tf]["bio"][fname] for tf in ordered_tfs]
    def spoof_stats(spoof_name,fname,order_by_stat="motif_ic"):
        ordered_tfs = order_tfs_by(order_by_stat)
        return [results_dict[tf][spoof_name][fname] for tf in ordered_tfs]
    for spoof_idx,spoof_name in enumerate(spoof_names):
        for stat_idx,stat_name in enumerate(stat_names):
            plt.subplot(len(spoof_names),len(stat_names),spoof_idx*3+stat_idx+1)
            if spoof_idx == 0 and stat_idx == 0:
                plt.ylabel("MaxEnt Sampling")
            if spoof_idx == 1 and stat_idx == 0:
                plt.ylabel("Uniform Sampling")
            plt.boxplot(spoof_stats(spoof_name,stat_name,order_by_stat=stat_name))
            plt.scatter(range(1,len(bio_stats(stat_name))+1),bio_stats(stat_name,order_by_stat=stat_name),
                        marker='^',color='r')
            plt.title(stat_name)
            plt.tick_params(
                axis='x',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom='off',      # ticks along the bottom edge are off
                top='off',         # ticks along the top edge are off
                labelbottom='off')
    maybesave(filename)

def summarize_tfdf_experiment(results_dict):
    gini_scores = []
    mi_scores = []
    for genome_tf in results_dict:
        i = genome_tf.rfind("_")
        genome,tf = genome_tf[:i],genome_tf[i+1:]
        print genome,tf
        motif = extract_tfdf_sites(genome,tf)
        n,L = motif_dimensions(motif)
        bio_ic = results_dict[genome_tf]['bio']['motif_ic']
        ic_score = val_in_coverage(bio_ic,results_dict[genome_tf]['maxent_truncated']['motif_ic'])
        bio_gini = results_dict[genome_tf]['bio']['motif_gini']
        gini_score = val_in_coverage(bio_gini,results_dict[genome_tf]['maxent_truncated']['motif_gini'])
        bio_mi = results_dict[genome_tf]['bio']['total_motif_mi']
        mi_score = val_in_coverage(bio_mi,results_dict[genome_tf]['maxent_truncated']['total_motif_mi'])
        print genome,tf,n,L,fmt(bio_ic),fmt(bio_gini),fmt(bio_mi),ic_score,gini_score,mi_score
        gini_scores.append(gini_score)
        mi_scores.append(mi_score)
    print "gini scores: L: %s -: %s H: %s" % tuple([gini_scores.count(c) for c in "L-H"])
    print "mi scores: L: %s -: %s H: %s" % tuple([mi_scores.count(c) for c in "L-H"])
        
def fmt(x):
    return round(x,2)

def motif_corr(motif,n=1000):
    """find correlated columns in motif, correcting for multiple hypothesis testing"""
    ps = [mi_permute(col1,col2,p_value=True,n=n,mi_method=lambda xs,ys:mi(xs,ys,correct=False))
          for (col1,col2) in (choose2(transpose(motif)))]
    q = fdr(ps)
    if q is None:
        return None
    else:
        L = len(motif[0])
        return [((i,j),p) for (i,j),p in zip(choose2(range(L)),ps) if p <= q]

def analyze_prodoric_collection_for_mi():
    for tf in Escherichia_coli.tfs:
        motif = getattr(Escherichia_coli,tf)
        cols = transpose(motif)
        n,L = motif_dimensions(motif)
        corrs = motif_corr(motif)
        if corrs:
            print tf,n,L,[((i,j),p,mi(cols[i],cols[j])) for (i,j),p in corrs]
        else:
            print tf,n,L

def analyze_all_pvals_at_once(org_obj=Escherichia_coli):
    """conclusion: fdr-adjusted p-values identify 25 significantly
    correlated column-pairs in 3753 pairwise tests (0.5%).  
    """
    ps = [mi_permute(col1,col2,p_value=True,n=1000,mi_method=lambda xs,ys:mi(xs,ys,correct=False))
          for tf in tqdm(org_obj.tfs)
          for (col1,col2) in (choose2(transpose(getattr(org_obj,tf))))]
    q_bh = fdr(ps)
    q_bhy = bhy(ps)
    print "bh procedure: %s/%s" % (len(filter(lambda p:p <= q_bh,ps)),len(ps))
    print "bhy procedure: %s/%s" % (len(filter(lambda p:p <= q_bhy,ps)),len(ps))
    return ps


# Correlated pairs:
#     Fis 7 8
#     Fis 10 11
#     GlpR 12 13
#     DnaA 2 3
#     DnaA 2 4
#     MarA 8 9
#     LexA 4 13
#     Crp 17 18
#     Fur 7 8
#     Fnr 7 8
#     Fnr 12 13
#     GlnG 2 4
#     MetJ 0 9
#     IHF 0 2
#     IHF 0 10
#     IHF 1 13
#     IHF 2 7
#     IHF 2 8
#     IHF 2 9
#     IHF 2 10
#     IHF 2 13
#     IHF 6 8
#     IHF 7 8
#     IHF 10 13
#     FliA 1 11

def analyze_composition_of_correlated_columns(obj,ps):
    p_idx = 0
    cor_adj_counts = defaultdict(int)
    cor_nonadj_counts = defaultdict(int)
    uncor_counts = defaultdict(int)
    fdr_cutoff = 0
    for tf in obj.tfs:
        motif = getattr(obj,tf)
        cols = transpose(motif)
        for (i,col1),(j,col2) in choose2(list(enumerate(cols))):
            if ps[p_idx] <= 0:
                print tf,i,j
                for pair in zip(cols[i],cols[j]):
                    if i + 1 == j:
                        cor_adj_counts[pair] += 1
                    else:
                        cor_nonadj_counts[pair] += 1
                #print mi_table(col1,col2)
            else:
                for pair in zip(cols[i],cols[j]):
                    uncor_counts[pair] += 1
            p_idx += 1
    cor_adj_N = float(sum(cor_adj_counts.values()))
    cor_nonadj_N = float(sum(cor_nonadj_counts.values()))
    uncor_N = float(sum(uncor_counts.values()))
    # all_N = float(sum(all_counts.values()))
    # print "---"
    # for b1,b2 in sorted(counts.keys()):
    #     
    #     print b1,b2,"freq:",fmt(counts[(b1,b2)]/N),"background:",fmt(all_counts[(b1,b2)]/all_N),"OR:",fmt(counts[(b1,b2)]/N/(all_counts[(b1,b2)]/all_N)),p
    print "bases, adj, nonadj, noncor | adj freq, nonadj freq | noncor freq| adj OR, nonadj OR"
    # XXX split into adj_uncor, nonadj_uncor
    for b1,b2 in sorted(cor_adj_counts.keys()):
        cor_adj_freq = fmt(cor_adj_counts[(b1,b2)]/cor_adj_N)
        cor_nonadj_freq = fmt(cor_nonadj_counts[(b1,b2)]/cor_nonadj_N)
        uncor_freq = fmt(uncor_counts[(b1,b2)]/uncor_N)
        cor_adj_OR = fmt(cor_adj_freq/uncor_freq)
        cor_nonadj_OR = fmt(cor_nonadj_freq/uncor_freq)
        _,adj_p,_,_ = stats.chi2_contingency(np.array([[uncor_N,uncor_counts[(b1,b2)]],
                                                       [cor_adj_N,cor_adj_counts[(b1,b2)]]]))
        _,non_adj_p,_,_ = stats.chi2_contingency(np.array([[uncor_N,uncor_counts[(b1,b2)]],
                                                       [cor_nonadj_N,cor_nonadj_counts[(b1,b2)]]]))
        print b1,b2,cor_adj_counts[b1,b2],cor_nonadj_counts[b1,b2],uncor_counts[b1,b2],"|",cor_adj_freq,cor_nonadj_freq,"|",uncor_freq,"|",cor_adj_OR, significance(adj_p),cor_nonadj_OR,significance(non_adj_p)
    return cor_adj_counts, cor_nonadj_counts, uncor_counts

def significance(x):
    if x < 0.01:
        return "**"
    elif x < 0.05:
        return "*"
    else:
        return ""

def rfreq_rseq_experiment(obj,filename="rfreq_vs_rseq_in_sefas_collection.png"):
    Rfreqs = []
    Rseqs = []
    G = 5.0*10**6
    min_rfreq = log2(G/500)
    for tf in obj.tfs:
        motif = getattr(obj,tf)
        Rfreqs.append(log(G/len(motif),2))
        Rseqs.append(motif_ic(motif))
    plt.scatter(Rfreqs,Rseqs)
    plt.xlabel("log(G/n) (bits)")
    plt.ylabel("Motif Information Content (bits)")
    plt.plot([0,20],[0,20],linestyle='--',label='Theory')
    plt.plot([min_rfreq,min_rfreq],[0,30],linestyle='--',label='Maximum Plausible Regulon Size')
    plt.title("Motif Information Content vs. Search Difficulty")
    plt.legend(loc='upper left')
    maybesave(filename)
        
def length_vs_sigma(obj):
    lens = []
    sigmas = []
    def get_sigma(motif):
        pssm = make_pssm(motif)
        return mean(map(sd,pssm))
    for tf in obj.tfs:
        motif = getattr(obj,tf)
        lens.append(len(motif[0]))
        sigmas.append(get_sigma(motif))
    print pearsonr(lens,sigmas)
    print spearmanr(lens,sigmas)
    plt.scatter(sigmas,lens)
    plt.plot(*pl(length_from_sigma,np.linspace(0,100,1000)))
    plt.xlabel("Sigma")
    plt.ylabel("Length")
    return lens,sigmas

def length_from_sigma(sigma):
    G = 5.0*10**6
    return log(G)*(1.0/sigma + 1/log(4))

def basic_statistics(tfdf=None,filename="basic_motif_statistics.png"):
    if tfdf is None:
        tfdf = extract_motif_object_from_tfdf()
    motifs = [getattr(tfdf,tf) for tf in tfdf.tfs]
    Ls = [len(motif[0]) for motif in motifs]
    ns = [len(motif) for motif in motifs]
    ics = [motif_ic(motif) for motif in motifs]
    ic_density = [ic/L for ic,L in zip(ics,Ls)]
    sigmas = [mean(map(sd,make_pssm(motif))) for motif in motifs]
    ginis = [motif_gini(motif,correct=False) for motif in motifs]
    mi_density = [total_motif_mi(motif)/choose(L,2) for motif,L in zip(motifs,Ls)]
    plt.subplot(2,3,1)
    #plt.tick_params(axis='x',pad=15)
    plt.xticks(rotation=90)
    plt.hist(Ls)
    plt.xlabel("Length (bp)")
    
    plt.subplot(2,3,2)
    #plt.tick_params(axis='x',pad=30)
    plt.xticks(rotation=90)
    plt.hist(ns)
    plt.xlabel("Number of sites")

    plt.subplot(2,3,3)
    plt.hist(ics)
    plt.xticks(rotation=90)
    plt.xlabel("IC (bits)")

    plt.subplot(2,3,4)
    #plt.tick_params(axis='x',pad=30)
    plt.xticks(rotation=90)
    plt.hist(ic_density)
    plt.xlabel("IC Density (bits/bp)")

    plt.subplot(2,3,5)
    #plt.tick_params(axis='x',pad=15)
    plt.xticks(rotation=90)
    plt.hist(ginis)
    plt.xlabel("Gini coeff")

    plt.subplot(2,3,6)
    #plt.tick_params(axis='x',pad=30)
    plt.xticks(rotation=90)
    plt.hist(mi_density)
    plt.xlabel("MI Density (bits/comparison)")
    plt.tight_layout()
    if filename:
        plt.savefig(filename,dpi=600)
    plt.close()

def analyze_column_frequencies():
    """Do columnwise frequencies reveal stable patterns that could be
explained by amino acid preferences?"""
    def dna_freqs(xs):
        return [xs.count(b)/float(len(xs)) for b in "ACGT"]
    all_freqs = concat([map(dna_freqs,transpose(getattr(tfdf_obj,tf)))
                         for tf in tfdf_obj.tfs])
    for k,(i,j) in enumerate(choose2(range(4))):
        plt.subplot(4,4,k)
        cols = transpose(all_freqs)
        plt.scatter(cols[i],cols[j])

def kmeans(xs,k=2):
    centroids = [simplex_sample(4) for i in range(k)]
    old_within_ss = 10**300
    while True:
        # assign to clusters
        clusters = [[] for i in range(k)]
        for x in xs:
            idx = argmin([l2(x,centroid) for centroid in centroids])
            clusters[idx].append(x)
        # recompute centroids
        centroids = [map(mean,transpose(cluster)) for cluster in clusters]
        cur_within_ss = sum([sum((l2(x,centroid)**2 for x in cluster)) for centroid,cluster in zip(centroids,clusters)])
        print cur_within_ss
        if cur_within_ss == old_within_ss:
            break
        else:
            old_within_ss = cur_within_ss
    return clusters,centroids,within_ss
        
def motif_dimensions(motif):
    return len(motif), len(motif[0])

def make_gle_evo_sim_spoofs(trials_per_motif = 3):
    start_time = time.time()
    spoofs = []
    failures = 0
    for it, motif in enumerate(tqdm(bio_motifs, desc='bio_motifs')):
        bio_ic = motif_ic(motif)
        these_spoofs = [spoof_motif_cftp(motif,num_motifs=10, Ne_tol=10**-2)
                        for i in range(trials_per_motif)]
        spoofs.append(these_spoofs)
        spoof_ics = map(motif_ic, concat(these_spoofs))
        lb, ub = mean_ci(spoof_ics)
        out_of_bounds = (not (lb <= bio_ic <= ub))
        failures += out_of_bounds
        fail_perc = failures/float(it+1)
        print it,"bio_ic:", bio_ic, "spoof_ci: (%s,%s)" % (lb, ub), "*" * out_of_bounds,"failures:","%1.2f" % fail_perc
    stop_time = time.time()
    print "total time:", stop_time  - start_time
    return spoofs
    
def interpret_gle_evo_sim_spoofs(spoofs,filename=None):
    #bio_motifs = [getattr(tfdf,tf) for tf in tfdf.tfs]
    # spoofs = [[spoof_motif_cftp(motif,Ne_tol=10**-2)
    #            for i in range(trials_per_motif)]
    #           for motif in tqdm(bio_motifs,desc='bio motifs')]
    trials_per_motif = len(spoofs[0])
    bio_ics = [motif_ic(motif) for motif in bio_motifs
               for _ in range(trials_per_motif)]
    sim_ics = [mean(map(motif_ic,motifs))
               for spoof in spoofs for motifs in spoof]
    bio_ginis = [motif_gini(motif) for motif in bio_motifs
               for _ in range(trials_per_motif)]
    sim_ginis = [mean(map(motif_gini,motifs))
                 for spoof in spoofs for motifs in spoof]
    bio_log_mis = [log(total_motif_mi(motif)) for motif in bio_motifs
               for _ in range(trials_per_motif)]
    sim_log_mis = map(log,[mean(map(total_motif_mi,motifs))
               for spoof in tqdm(spoofs) for
               motifs in spoof])
    lens = [len(motif[0]) for motif in bio_motifs for _ in range(trials_per_motif)]
    bio_mis = [exp(bio_log_mi)/choose(l,2)
               for (l, bio_log_mi) in zip(lens, bio_log_mis)]
    sim_mis = [exp(sim_log_mi)/choose(l,2)
               for (l, sim_log_mi) in zip(lens, sim_log_mis)]
    bio_patterns = [find_pattern(motif)[0] for motif in bio_motifs]
    pattern_colors = {'direct-repeat':'g','inverted-repeat':'b','single-box':'r'}
    colors = [pattern_colors[p] for p in bio_patterns]
    plt.subplot(1,3,1)
    plt.title("Motif IC (bits)") 
    scatter(bio_ics,sim_ics,color=colors,
            line_color='black')
    ic_f = poly1d(polyfit(bio_ics, sim_ics,1))
    #plt.plot(*pl(ic_f,[min(bio_ics),max(bio_ics)]),linestyle='--',color='b')
    plt.xlim(*find_limits(bio_ics, sim_ics))
    plt.ylim(*find_limits(bio_ics, sim_ics))
    plt.ylabel("Simulated")
    plt.subplot(1,3,2)
    plt.xlabel("Observed")
    plt.title("Gini Coefficient")
    scatter(bio_ginis,sim_ginis,color=colors,
            line_color='black')
    gini_f = poly1d(polyfit(bio_ginis, sim_ginis,1))
    #plt.plot(*pl(gini_f,[min(bio_ginis),max(bio_ginis)]),
     #        linestyle='--',color='b')
    plt.xlim(*find_limits(bio_ginis, sim_ginis))
    plt.ylim(*find_limits(bio_ginis, sim_ginis))
    plt.subplot(1,3,3)
    plt.title("Pairwise MI per pair (bits)")
    draft = False
    end = 10 if draft else 108
    scatter(bio_mis,sim_mis,color=colors,
            line_color='black')
    mi_f = poly1d(polyfit(bio_mis, sim_mis,1))
    # plt.plot(*pl(mi_f,[min(bio_mis),max(bio_mis)]),
    #          linestyle='--',color='b')
    plt.xlim(*find_limits(bio_mis, sim_mis))
    plt.ylim(*find_limits(bio_mis, sim_mis))
    plt.legend()
    # #ax.set_bg_color('none')
    # ax.set_xlabel("Biological")
    # ax.set_ylabel("Simulated")
    plt.tight_layout()
    maybesave(filename)
    
def find_limits(xs,ys,pad=None):
    zs = xs + ys
    if pad is None:
        pad = (max(zs) - min(zs)) * 0.05
    return min(zs) - pad, max(zs) + pad
