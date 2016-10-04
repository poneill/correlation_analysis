from utils import make_pssm, score_seq, percentile
from formosa import spoof_maxent_motifs

def jackknife_distribution(motif):
    scores = []
    for i in range(len(motif)):
        motif_p = [site for j, site in enumerate(motif) if not i == j]
        scores.append(score_seq(make_pssm(motif_p), motif[i]))
    return scores

def self_score_percentile(motif):
    spoofs = spoof_maxent_motifs(motif, 1000)
    spoof_jk_sds = [sd(jackknife_distribution(spoof)) for spoof in tqdm(spoofs)]
    motif_jk_sd = sd(jackknife_distribution(motif))
    return percentile(motif_jk_sd, spoof_jk_sds)
