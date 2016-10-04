from utils import mh, motif_ic, mutate_motif, random_motif

def sample_motif(N, L, beta):
    x0 = random_motif(L, N)
    def log_f(x):
        return -beta*(2*L - motif_ic(x))
    chain = mh(log_f, mutate_motif, x0, use_log=True)
