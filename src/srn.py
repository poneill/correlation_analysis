"""
Implement the model given in simple_fitness_model with SRN + gillespie sampling
"""
N = 10
mu = 0.01

def fitness(n):
    """count number of 1s in binary representation of n"""
    return bin(n).count('1')
    
def gillespie(tf=10):
    pop = np.zeros(2**10)
    t = 0
    while t < tf:
        base_rates = [i*(fitness(i)+delta) for i in pop]
        pass
        
