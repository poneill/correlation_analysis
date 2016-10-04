"""Analyze a match-mismatch model under multiplicative fitness"""

from math import exp, log

def fitness(sigma, L, rho, G):
    """find occupancy of binder of length L, sigma kbt per match, rho mismatches, on a genome of length G"""
    Z = G*((3+exp(sigma))/4.0)**L
    m = L - rho
    return exp(m*sigma)/Z
