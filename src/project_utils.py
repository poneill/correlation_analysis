from utils import motif_ic, entropy, transpose, mean, sd, dnorm

def trim_motif(motif):
    L = len(motif[0])
    col_hs = map(lambda col:entropy(col,correct=True,alphabet_size=4),
                 transpose(motif))
    mu,sigma = mean(col_hs),sd(col_hs)
    threshold = mu + 1.96*sigma
    l,r = 0, L-1
    while col_hs[l] > threshold:
        print "trimming left endpoint:",l,col_hs[l]
        l += 1
    while col_hs[r] > threshold:
        print "trimming left endpoint:",col_hs[r]
        r -= 1
    return [site[l:r+1] for site in motif]

def detect_chunks(motif):
    L = len(motif[0])
    col_ics = map(lambda col:2-entropy(col,correct=True,alphabet_size=4),
                 transpose(motif))
    f = cluster2(col_ics)
    profiles = [f(col_ic) > 0.5 for col_ic in col_ics]
    transitions = [profiles[0]] + [x != y for (x,y) in pairs(profiles)] + [profiles[-1]]
    return sum(transitions)/2.0
    

def cluster(xs):
    """given scalars xs, perform 2-component Gaussian clustering via EM"""
    mu0 = min(xs)
    mu1 = max(xs)
    sigma0 = 1
    sigma1 = 1
    for i in range(10):
        probs0 = [dnorm(x,mu0,sigma0) for x in xs]
        probs1 = [dnorm(x,mu1,sigma1) for x in xs]
        assignments = [int (prob1 > prob0) for prob0,prob1 in zip(probs0,probs1)]
        xs0 = [x for (x,a) in zip(xs,assignments) if a == 0]
        xs1 = [x for (x,a) in zip(xs,assignments) if a == 1]
        mu0,sigma0 = mean(xs0), sd(xs0,correct=False)
        mu1,sigma1 = mean(xs1), sd(xs1,correct=False)
        if sigma0 == 0:
            sigma0 = sigma1
        if sigma1 == 0:
            sigma1 = sigma0
        print "mu0: {} sigma0: {} mu1: {}: sigma1: {} xs0: {} xs1: {}".format(mu0,sigma0,mu1,sigma1,len(xs0),len(xs1))
    def f(x):
        return dnorm(x,mu1,sigma1)/(dnorm(x,mu0,sigma0) + dnorm(x,mu1,sigma1))
    return f

def cluster2(xs):
    """given scalars xs, perform 2-component Gaussian clustering via EM"""
    mu0 = min(xs)
    mu1 = max(xs)
    sigma0 = 1
    sigma1 = 1
    for i in range(10):
        def f(x):
            return dnorm(x,mu1,sigma1)/(dnorm(x,mu0,sigma0) + dnorm(x,mu1,sigma1))
        ps = map(f,xs)
        mu1 = sum(x*p for (x,p) in zip(xs,ps))/sum(ps)
        mean_xsq_1 = sum(x**2 * p for (x,p) in zip(xs,ps))/sum(ps)
        sigma1 = sqrt(mean_xsq_1 - mu1**2)
        mu0 = sum(x*(1-p) for (x,p) in zip(xs,ps))/sum(1-p for p in ps)
        mean_xsq_0 = sum(x**2  * (1-p) for (x,p) in zip(xs,ps))/sum(1-p for p in ps)
        sigma0 = sqrt(mean_xsq_0 - mu0**2)
        print i,"mu0: {} sigma0: {} mu1: {} sigma1: {} xs0: {} xs1: {}".format(mu0,sigma0,mu1,sigma1,len(xs0),len(xs1))
    def f(x):
        return dnorm(x,mu1,sigma1)/(dnorm(x,mu0,sigma0) + dnorm(x,mu1,sigma1))
    return f


def my_score_seq(matrix,seq):
    #base_dict = {'A':0,'C':1,'G':2,'T':3}
    def base_dict(b):
        if b <= "C":
            if b == "A":
                return 0
            else:
                return 1
        elif b == "G":
            return 2
        else:
            return 3
    ans = 0
    for i in xrange(len(seq)):
        ans += matrix[i][base_dict(seq[i])] if seq[i] in "ACGT" else 0
    return ans
