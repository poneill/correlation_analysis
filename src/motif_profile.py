# From Sefa
from Bio import motifs
import glob
import itertools


def load_sites(motif_file):
    """Loads binding sites from the given file.
    Assumes all sites are of same length.
    """
    with open(motif_file) as f:
        sites = [line.strip() for line in f]
    return sites


def build_motif(sites):
    """Builds a Biopython motifs.Motif object out of given sites."""
    motif = motifs.create(sites)
    motif.pseudocounts = 0.8   # http://www.ncbi.nlm.nih.gov/pmc/articles/PMC2647310/
    return motif


def save_weblogo(motif, filename):
    """Saves weblogo to the given file."""
    motif.weblogo(filename, color_scheme='color_classic')


def slice_sites(sites, start, end):
    """Slices each site."""
    return [site[start:end] for site in sites]


def score_site(pssm, site):
    """Scores the given site with the given PSSM."""
    return sum(pssm[site[i]][i] for i in range(len(site)))


def score_sites(pssm, sites):
    """Computes the average PSSM score of a list of sites."""
    return sum(score_site(pssm, site) for site in sites) / len(sites)


def direct_repeat(seq):
    """Match for a sequence in a direct-repeat pattern."""
    return seq


def inverted_repeat(seq):
    """Match for a sequence in an inverted-repeat pattern."""
    complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}
    return ''.join(complement[b] for b in seq[::-1])


def mirror_repeat(seq):
    """Match for a sequence in a mirror-repeat pattern."""
    return seq[::-1]


def score_pattern(sites, complement_fn):
    """Computes a score for the pattern.
    For each k-mer, scores all other non-overlapping k-mers.
    """
    sites_len = len(sites[0])
    pattern = (float('-inf'), None, None)

    min_k, max_k = (4, 4)
    for k in range(min_k, max_k+1):
        # Consider all k-mers as binding unit
        for unit_start in range(sites_len - 2*k + 1):
            unit_end = unit_start + k
            unit_motif = build_motif(slice_sites(sites, unit_start, unit_end))
            # Score every other k-mer.
            for other_kmer_start in range(unit_end, sites_len - k + 1):
                other_kmer_end = other_kmer_start + k
                other_kmer = slice_sites(sites, other_kmer_start, other_kmer_end)
                score = score_sites(unit_motif.pssm,
                                    map(complement_fn, other_kmer))
                if score > pattern[0]:
                    pattern = (
                        score,
                        (unit_start, unit_end),
                        (other_kmer_start, other_kmer_end))
    return pattern


def find_pattern(sites):
    """Looks for direct-, inverted- and mirror-repeats in the binding motif."""
    pattern_scores = [
        ('direct-repeat', score_pattern(sites, direct_repeat)),
        ('inverted-repeat', score_pattern(sites, inverted_repeat)),
        ('single-box', (0,))]

    pattern = max(pattern_scores, key=lambda x: x[1][0])
    return pattern[0], pattern[1]


def find_pattern_all():
    motif_files = glob.glob('./data/*.txt')
    for motif_file in motif_files:
        sites = load_sites(motif_file)
        find_pattern_alternative(sites)


def find_pattern_alternative(sites,
                             self_score_ratio_threshold=0.6,
                             kmer_pair_score_ratio_threshold=0.3):

    """Finds pattern in a motif."""
    k = 4
    sites_len = len(sites[0])
    all_kmers = [{'start': i, 'end': i+k, 'seqs': slice_sites(sites, i, i+k)}
                 for i in range(sites_len-k+1)]
    # Compute self-PSSM scores
    for kmer in all_kmers:
        motif = build_motif(kmer['seqs'])
        kmer['pssm'] = motif.pssm
        kmer['self_score'] = score_sites(kmer['pssm'], kmer['seqs'])

    max_self_score = max(kmer['self_score'] for kmer in all_kmers)
    all_kmers = [kmer for kmer in all_kmers
                 if kmer['self_score'] > self_score_ratio_threshold*max_self_score]


    pattern = (0, 'single_box')
    for kmer_a, kmer_b in itertools.combinations(all_kmers, 2):
        if not (kmer_a['start'] >= kmer_b['end'] or
                kmer_b['start'] >= kmer_a['end']):
            continue

        if kmer_a['self_score'] < kmer_b['self_score']:
            kmer_a, kmer_b = kmer_b, kmer_a

        # Look for direct-repeat
        score = score_sites(kmer_a['pssm'], kmer_b['seqs'])
        if (score > kmer_pair_score_ratio_threshold*kmer_a['self_score'] and
            score > pattern[0]):
            pattern = (score, 'direct_repeat', kmer_a['start'], kmer_b['start'])

        # Look for inverted-repeat
        score = score_sites(
            kmer_a['pssm'], [inverted_repeat(site) for site in kmer_b['seqs']])
        if (score > kmer_pair_score_ratio_threshold*kmer_a['self_score'] and
            score > pattern[0]):
            pattern = (score, 'inverted_repeat', kmer_a['start'], kmer_b['start'])

    print pattern
