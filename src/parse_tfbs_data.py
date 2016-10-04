from collections import defaultdict
from utils import motif_ic
class Organism():
    pass

def make_tfbs_object():
    with open("/home/pat/TFBS_data/tfbs_data_merged.tsv") as f:
        lines = [line.strip().split("\t") for line in f.readlines()[1:]]
    raw_dict = defaultdict(list)
    for line in lines:
        protein_id = line[2]
        site = line[7]
        raw_dict[protein_id].append(site)
    tfbss = Organism()
    setattr(tfbss,"tfs",[])
    total = 0
    insuf_site_rejected = 0
    diff_len_rejected = 0
    ic_rejected = 0
    for prt_id,sites in raw_dict.items():
        total += 1
        if len(sites) < 10:
            print "not enough sites for:",prt_id
            insuf_site_rejected += 1
            continue
        elif len(set(map(len,sites))) > 1:
            print "sites of different lengths in:",prt_id
            diff_len_rejected += 1
            continue
        elif motif_ic(sites) < 5:
            print "insufficient IC:",prt_id
            ic_rejected += 1
            continue
        else:
            print "appending:",prt_id
            if len(sites[0]) == 53: # deal with this case by hand
                tfbss.tfs.append(prt_id)
                trunc_sites = [site[15:38] for site in sites]
                setattr(tfbss,prt_id,trunc_sites)
            else:
                tfbss.tfs.append(prt_id)
                setattr(tfbss,prt_id,sites)
    print "total:",total
    print "insufficient sites:", insuf_site_rejected
    print "diff_len_rejected:", diff_len_rejected
    print "ic_rejected:", ic_rejected
    print "accepted:",len(tfbss.tfs)
    print "total:", len(tfbss.tfs) + (insuf_site_rejected +
                                      diff_len_rejected + ic_rejected)
    return tfbss

tfbss = make_tfbs_object()
tfdf = tfbss
