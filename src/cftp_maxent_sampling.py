from utils import subst, motif_ic

def maxent_site_cftp(N, beta):
    top_site = ["A"]*N
    bottom_site = ["T"]*N
    def mutate_site(site, (ri, rdir)):
        i = "ACGT".index(site[ri])
        ip = max(0, min(3, i + rdir))
        bp = "ACGT"[ip]
        return subst(site,[bp],ri)
    iterations = 1
    trajs = [[top_site],[bottom_site]]
    rs = [(random.randrange(N),random.choice([-1,1]),random.random())
          for i in range(iterations)]
    converged = False
    def prob(site):
        return exp(-beta*motif_ic(site))
    while not converged:
        for ri, rdir, r in rs:
            for traj in trajs:
                x = traj[-1]
                xp = mutate_site(x,(ri, rdir))
                if log(r) < prob(xp) - prob(x):
                    x = xp
                traj.append(x)
        if trajs[0][-1] == trajs[-1][-1]:
            converged = True
        iterations *= 2
        #print iterations,[traj[-1] for traj in trajs]
        rs = [(random.randrange(N),random.choice([-1,1]),random.random())
              for i in range(iterations)] + rs
    assert all(map(lambda traj:traj[-1] == trajs[0][-1],trajs))
    return trajs[0][-1]
    
