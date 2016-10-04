import numpy as np

def adaptive_mh(f, x0, iterations=50000, verbose=False):
    """do adaptive mh according to
    http://dept.stat.lsa.umich.edu/~yvesa/afmp.pdf, p.3."""
    d = len(x0)
    
