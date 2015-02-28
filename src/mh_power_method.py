import numpy as np
import random
from utils import sorted_indices,rslice,simplex_sample,inverse_cdf_sample

def random_stochastic_matrix(K):
    return np.matrix([simplex_sample(K) for j in range(K)])

def random_rm_matrix(K,mu):
    fitnesses = np.random.random(K)
    Fs = np.diag(fitnesses)
    A = np.random.random((K,K))
    Mu = (A + A.transpose())/2 # arbitrary symmetric matrix
    #Mu = np.matrix([[mu if j!=i else 1-(K-1)*mu for j in range(K)] for i in range(K)])
    W = Fs.dot(Mu)
    return W

def random_nice_rm_matrix(K,mu):
    fitnesses = np.random.random(K)
    Fs = np.diag(fitnesses)
    Mu = np.matrix([[mu if j!=i else 1-(K-1)*mu for j in range(K)] for i in range(K)])
    W = Fs.dot(Mu)
    return W
    
def random_simplex_vector(K):
    return np.array(simplex_sample(K))

def random_symmetric_matrix(K):
    A = np.random.random((K,K))
    Mu = (A + A.transpose())/2 # arbitrary symmetric matrix
    return Mu
    
def largest_eigenvector(A,iterations=100):
    K = len(A)
    v = random_simplex_vector(K)
    for iteration in xrange(iterations):
        v = v.dot(A)
        v /= np.sum(v)
    return np.array(v.tolist())

def random_walk_eigenvector(A,iterations=50000):
    K = len(A)
    history = np.zeros(K)
    state = random.randrange(K)
    for _ in xrange(iterations):
        history[state] += 1
        state = inverse_cdf_sample(range(K),(A[state,:]).tolist()[0])
    return history/np.sum(history)

def mh_eigenvector(A,iterations=50000):
    K = len(A)
    history = np.zeros(K)
    i = random.randrange(K)
    for _ in xrange(iterations):
        history[i] += 1
        j = random.randrange(K)
        ar = A[i,j]/A[j,i]
        if random.random() < ar:
            i = j
    return history/np.sum(history)
    
# A = random_stochastic_matrix(3)
# evals,evecs = np.linalg.eig(A.transpose())
# js = sorted_indices(evals)[::-1]
# evals,evecs = rslice(evals,js),rslice(evecs,js)
# assert evals[0] == max(evals)
# lamb = evals[0]
# v = evecs[0]
