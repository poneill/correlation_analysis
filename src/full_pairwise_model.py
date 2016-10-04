from utils import choose, choose2, kmers, transpose, random_site, mutate_site, motif_ic, mh
from collections import Counter
import random
from math import log, exp, sqrt

dinucs = [(b1, b2) for b1 in "ACGT" for b2 in "ACGT"]

def get_pairwise_freqs(motif, pc=1/16.0):
    cols = transpose(motif)
    L = len(cols)
    N = len(motif)
    fs = [{(b1, b2):0 for (b1,b2) in dinucs} for _ in range(int(choose(L,2)))]
    for f, (col1, col2) in zip(fs, choose2(cols)):
        for b1, b2 in zip(col1, col2):
            f[b1, b2] += 1
        for b1, b2 in dinucs:
            f[b1, b2] += pc
            f[b1, b2] /= float(N + 16*pc)
    return fs

def train_pairwise_model(motif, pc=1/16.0, decay_timescale=10000, take_stock=1000, eta=0.01, stop_crit=0.01):
    L = len(motif[0])
    N = len(motif)
    fs = get_pairwise_freqs(motif, pc=pc)
    ws = [{(b1, b2):0 for (b1,b2) in dinucs} for _ in range(int(choose(L,2)))]
    x = random_site(L)
    log_y = score(ws, x)
    chain = []
    # sses = [0.0] * (int(iterations/take_stock) + 1)
    #chain = []
    #for iteration in xrange(iterations):
    iteration = 0
    stock_counter = take_stock
    while True:
        xp = mutate_site(x)
        log_yp = score(ws, xp)
        if log(random.random()) < log_yp - log_y:
            x = xp
            log_y = log_yp
        chain.append(x)
        if iteration > 0 and iteration % stock_counter == 0:
            current_fs = get_pairwise_freqs(sample(N,chain[iteration-stock_counter : iteration], replace=False))
            sse = 0
            for w, f, cur_f in zip(ws, fs, current_fs):
                for b1, b2 in dinucs:
                    delta = f[b1, b2] - cur_f[b1,b2]
                    sse += delta**2
                    w[b1, b2] += eta*(delta) #* exp(-iteration/float(decay_timescale))
            #sses[iteration/take_stock] = sse
            sse_per_col_pair = sse/choose(L,2)
            print iteration, stock_counter, sse_per_col_pair, exp(-iteration/float(decay_timescale)), ws[0]['A','A']
            stock_counter += random.randrange(2)
            #print "motif_ic:", motif_ic(chain[iteration-stock_counter : iteration])
            if iteration > 0 and sse_per_col_pair < stop_crit:
                print "breaking:", sse, sse_per_col_pair
                break
            log_y = score(ws, x) # recalculate this because weights change
            #stock_counter += take_stock * (iteration > take_stock)
        iteration += 1
    return ws

def train_pairwise_model2(motif, pc=1/16.0, decay_timescale=10000, take_stock=1000, eta=0.01, stop_crit=0.01):
    L = len(motif[0])
    N = len(motif)
    fs = get_pairwise_freqs(motif, pc=pc)
    ws = [{(b1, b2):0 for (b1,b2) in dinucs} for _ in range(int(choose(L,2)))]
    iteration = 0
    while True:
        cur_motif = [sample_model(ws,x0=site,iterations=10*L)[-1] for site in motif]
        current_fs = get_pairwise_freqs(cur_motif)
        sse = 0
        for w, f, cur_f in zip(ws, fs, current_fs):
            for b1, b2 in dinucs:
                delta = f[b1, b2] - cur_f[b1,b2]
                sse += delta**2
                w[b1, b2] += eta*(delta) #* exp(-iteration/float(decay_timescale))
            #sses[iteration/take_stock] = sse
        sse_per_col_pair = sse/choose(L,2)
        print iteration, sse_per_col_pair, ws[0]['A','A']
        print "motif_ic:", motif_ic(cur_motif)
        if iteration > 0 and sse_per_col_pair < stop_crit:
            print "breaking:", sse, sse_per_col_pair
            break
        iteration += 1
    return ws

        
def score(model, site):
    return sum(wi[(b1, b2)] for wi, (b1,b2) in zip(model, choose2(site)))

def random_pairwise_model(L, sigma=1):
    return [{(b1,b2):random.gauss(0,sigma) for b1 in "ACGT" for b2 in "ACGT"}
            for _ in range(int(choose(L,2)))]

def compute_Z(model):
    k = len(model)
    L = int(1 + sqrt(1+8*k)/2)
    return sum(exp(score(model, "".join(kmer))) for kmer in kmers(L))

def sample_model(model, iterations=50000,x0=None):
    k = len(model)
    L = int(1 + sqrt(1+8*k)/2)
    if x0 is None:
        x0 = random_site(L)
    chain = mh(lambda s:score(model,s),
               proposal=mutate_site,
               x0=random_site(L),
               use_log=True, iterations=iterations)
    return chain

def spoof_motif(model, N):
    k = len(model)
    L = int(1 + sqrt(1+8*k)/2)
    return [sample_model(model, iterations=10*L)[-1] for i in range(N)]
