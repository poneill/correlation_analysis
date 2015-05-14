from utils import h,choose,log2

def hamming_evolution(L):
    ps = [1] + [0]*(L)
    dt = 0.001
    t = 0
    tf = 5
    hist = []
    hs = []
    while t < tf:
        psp = ps[:]
        hist.append(ps[:])
        for i in range(L+1):
            bolus = ps[i]*dt
            if 0 <= i - 1 <= L:
                psp[i] -= i*bolus
                psp[i-1] += i*bolus
            if 0 <= i + 1 <= L:
                psp[i] -= 3*(L-i)*bolus
                psp[i+1] += 3*(L-i)*bolus
        ps = psp[:]
        this_h = -sum(p*(log2(p/num_seqs_at(L,i)) if p > 0 else 0) for i,p in enumerate(ps))
        pred_h = 2*L*(1-exp(-10*t))
        print t,sum(ps),this_h,pred_h
        hs.append(this_h)
        t += dt
    return hist,hs

def num_seqs_at(L,i):
    return choose(L,i)*(3**i)
