from mi_analysis import get_motifs
from utils import motif_ic, motif_gini, maybesave
import pandas as pd

def main(prok_motifs, euk_motifs, filename='motif_summary_stats.eps'):
    sns.set(style="darkgrid", color_codes=True)
    #df = pd.DataFrame(columns="Type N L IC Gini".split(), index=range(len(prok_motifs) + len(euk_motifs)))
    df = pd.DataFrame()
    df['Domain'] =  ["Eukaryotic" for _ in euk_motifs] + ["Prokaryotic" for _ in prok_motifs]
    motifs = euk_motifs + prok_motifs
    df['N'] = [log(len(motif))/log(10) for motif in motifs]
    df['L (bp)'] = [len(motif[0]) for motif in motifs]
    df['IC (bits)'] = [motif_ic(motif) for motif in motifs]
    df['IGC'] = [motif_gini(motif) for motif in motifs]
    pg = sns.pairplot(df,hue='Domain',markers='s o'.split(),palette='cubehelix')
                      #hue_order=["Prokaryotic", "Eukaryotic"])
    for i in range(4):
        pg.axes[i][3].set_xlim(-0.01,0.6)
    for j in range(4):
        pg.axes[3][j].set_ylim(-0.01,0.6)
    pg.axes[0][0].set_yticks(range(1, 5))
    pg.axes[0][0].set_yticklabels(["$10^%i$" % i for i in range(1, 5)])
    pg.axes[3][0].set_xticks(range(1, 5))
    pg.axes[3][0].set_xticklabels(["$10^%i$" % i for i in range(1, 5)])
    maybesave(filename)

