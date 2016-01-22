from exact_evo_sim_sampling import sample_motif_cftp, approx_mu
from utils import prod, random_site, score_seq
from pwm_utils import sample_matrix

def fitness(matrix,motif, mu, Ne):
    nu = Ne - 1
    def phat(s):
        ep = score_seq(matrix,s)
        return (1 + exp(ep - mu))**(-nu)
    return prod(map(phat,motif))

def mean_occupancy(matrix, motif, mu):
    def occ(s):
        ep = score_seq(matrix,s)
        return 1/(1 + exp(ep - mu))
    return mean(map(occ,motif))

def mean_occ(matrix,motif):
    mu = approx_mu(matrix,10*n)
    return mean_occupancy(matrix,motif,mu)
    
L = 10
n = 20
Ne = 5

matrix1 = [[-2,0,0,0] for i in range(L)]
matrix2 = [[-4,0,0,0] for i in range(5)] + [[0,0,0,0] for i in range(L-5)]
matrix5 = [[-10,0,0,0] for i in range(2)] + [[0,0,0,0] for i in range(L-2)]

sigmas = np.linspace(0,10,11)
Nes = np.linspace(1,10,10)

def experiment1():
    matrices = [[sample_matrix(10,sigma) for sigma in sigmas] for Ne in Nes]
    motifses = mmap(lambda matrix:[sample_motif_cftp(matrix,approx_mu(matrix,10*n),Ne,n) for i in range(10)],
                    tqdm(matrices))

    occs = [[mean(mean_occupancy(matrix,m,approx_mu(matrix,10*n)) for m in motif)
             for (matrix,motif) in zip(matrix_row,motif_row)]
            for (matrix_row,motif_row) in zip(matrices,motifses)]

def sample_mean_occ(sigma,Ne):
    matrix = sample_matrix(L,sigma)
    mu = approx_mu(matrix,10*n)
    motif = sample_motif_cftp(matrix,mu,Ne,n)
    return mean_occupancy(matrix,motif,mu)

def sample_gini(sigma,Ne):
    matrix = sample_matrix(L,sigma)
    mu = approx_mu(matrix,10*n)
    motif = sample_motif_cftp(matrix,mu,Ne,n)
    return motif_gini(motif)

def sample_pair(sigma,Ne):
    matrix = sample_matrix(L,sigma)
    mu = approx_mu(matrix,10*n)
    motif = sample_motif_cftp(matrix,mu,Ne,n)
    return matrix, motif
    
def experiment2():
    occs = [[mean(sample_mean_occ(sigma,Ne) for i in range(3)) for sigma in sigmas] for Ne in tqdm(Nes)]
    ginis = [[mean(sample_gini(sigma,Ne) for i in range(3)) for sigma in sigmas] for Ne in tqdm(Nes)]

def experiment3():
    sample_pairs = [[sample_pair(sigma,Ne) for sigma in sigmas] for Ne in tqdm(Nes)]
# mu1 = approx_mu(matrix1,10*n,G=5*10**6)
# mu2 = approx_mu(matrix2,10*n,G=5*10**6)
# mu5 = approx_mu(matrix5,10*n,G=5*10**6)

# motifs1 = [sample_motif_cftp(matrix1,mu1,Ne,n=n) for i in trange(10)]
# motifs2 = [sample_motif_cftp(matrix2,mu2,Ne,n=n) for i in trange(10)]
# motifs5 = [sample_motif_cftp(matrix5,mu5,Ne,n=n) for i in trange(10)]

def matrix_gini(matrix):
    return gini([h(normalize([exp(-ep) for ep in row])) for row in matrix])
