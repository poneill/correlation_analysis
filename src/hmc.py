import numpy as np
from simplex_sampling_experiment import h_np,simplex_sample

def hmc(qs0,ps0,dhdq,dhdp,dt,steps):
    qs = qs0.copy()
    print qs
    #ps = np.random.normal(0,1,len(qs))
    ps = ps0.copy()
    qs_history = []
    for step in range(steps):
        ps_half = ps - dt/2.0 * dhdq(qs)
        qs_new = qs + dt*ps_half
        ps_new = ps_half - dt/2 * dhdq(qs_new)
        ps = ps_new.copy()
        qs = qs_new.copy()
        qs_history.append(qs)
        print qs
    return qs_history


def hmc_example(dt=1,steps=100):
    """
    Do Hamiltonian Monte Carlo for 1d particle in quadratic potential:

    H(q,p) = U(q) + K(p)
           = mgh + 1/2*m*v^2
           = m(g(q^2)) + 1/2*(p/m)^2
           = q^2 + 1/2p^2,
    in units where m = g = 1
    """
    dhdq = lambda qs:2*qs
    dhdp = lambda ps:ps
    qs0 = np.array([0])
    ps0 = np.array([1])
    return hmc(qs0,ps0,dhdq,dhdp,dt=dt,steps=steps)
 
def hmc_example2(dt=1,steps=100):
    M = 10000
    K = 4
    des_ent = 1
    dhdp = lambda qs:M*(2*(h_np(qs) - des_ent)*(1+np.log(qs)) + (np.sum(q)-1))
    dhdq = lambda ps:ps
    qs0 = np.array(simplex_sample(K))
    ps0 = np.random.normal(0,1,K)
    return hmc(qs0,ps0,dhdq,dhdp,dt=dt,steps=steps)
