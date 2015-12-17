from collections import defaultdict

class Organism():
    pass

def make_tfbs_object():
    with open("../../TFBS_data/tfbs_data_merged.tsv") as f:
        lines = [line.strip().split("\t") for line in f.readlines()[1:]]
    raw_dict = defaultdict(list)
    for line in lines:
        protein_id = line[2]
        site = line[7]
        raw_dict[protein_id].append(site)
    tfbss = Organism()
    setattr(tfbss,"tfs",[])
    for prt_id,sites in raw_dict.items():
        if len(sites) < 10:
            print "not enough sites for:",prt_id
            continue
        elif len(set(map(len,sites))) > 1:
            print "sites of different lengths in:",prt_id
            continue
        else:
            print "appending:",prt_id
            tfbss.tfs.append(prt_id)
            setattr(tfbss,prt_id,sites)
    return tfbss

tfbss = make_tfbs_object()
tfdf = tfbss
