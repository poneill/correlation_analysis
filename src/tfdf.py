from parse_merged_data import tfdf

def extract_tfdf_sites(genome,tf):
    sites = tfdf[tfdf['genome_accession'] == genome][tfdf['TF']==tf]['site_sequence']
    # convert to list matrix form,remove nans
    bio_motif_all_lens = [site for site in sites if type(site) is str] 
    modal_length = mode(map(len,bio_motif_all_lens))
    bio_motif = filter(lambda x:len(x)==modal_length,bio_motif_all_lens)
    if len(bio_motif) != len(bio_motif_all_lens):
        print "removed", len(bio_motif_all_lens) - len(bio_motif),"of",
    return bio_motif

def extract_motif_object_from_tfdf():
    obj = Organism()
    genomes = set(tfdf['genome_accession'])
    setattr(obj,"tfs",[])
    for genome in genomes:
        tfs = set(tfdf[tfdf['genome_accession'] == genome]['TF'])
        for tf in tfs:
            print tf,genome
            if not type(tf) is str:
                continue
            tf_name = genome + "_" + tf
            sites = extract_tfdf_sites(genome,tf)
            if len(sites) >= 10:
                setattr(obj,tf_name,sites)
                obj.tfs.append(tf_name)
    return obj

tfdf = extract_motif_object_from_tfdf()
