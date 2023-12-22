from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import pdb


r = 0.6 # reproduction rate

# carrying capacity K :
K0, sigma_k = 2, 0.2

# strength of competition between individuals with trait values x1 and x2
sigma_c = 0.5

# total competition experienced by a single individual (approximate provided in document)
n_bins = 40 # number of classes

# directional selection constant
beta = 0.5

# mutation rate and std
mu, sigma_m = 0.3, 0.5


class Individual:
    def __init__(self, parent: Individual = None):
        if parent == None:
            self.X = 0
            self.Y = 0
        else:
            self.X = parent.X
            self.Y = parent.Y
            if np.random.rand() < mu:
                self.X = np.random.normal(parent.X, sigma_m)
                self.Y = np.random.normal(parent.Y, sigma_m)
                
            
        
    def K(self): 
        return K0*np.exp(-self.X**2/(2*sigma_k**2))
    
    def alpha(self, other: Individual): 
        return np.exp(-(self.X-other.X)**2/(2*sigma_c**2))
        
    def C(self, pop):
        f, edges = np.histogram(pop, n_bins)
        centers = [(edges[i]+edges[i+1])/2 for i in range(len(edges)-1)]
        return sum([self.alpha(self.X,centers[i])*f[i] for i in centers])
        
    def delta(self, pop: list[Individual]):
        Yb = np.mean([other.Y for other in pop])
        return self.C(pop)/self.K() - beta*(self.Y - Yb)
    
    def W(self, pop: list[Individual]):
        return np.exp(r*(1-self.delta(pop)))


if __name__ == '__main__':
    
    # checking that Individual intialisation works fine
    A = Individual()
    B = Individual(A)
    
    
    #pdb.set_trace()


