from __future__ import annotations
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tqdm
import pdb
import sys




r = 1 # reproduction rate

# carrying capacity K :
K0, sigma_k = 1000, 1

# strength of competition between individuals with trait values x1 and x2
sigma_c = 1

# total competition experienced by a single individual (approximate provided in document)
n_bins = 40 # number of classes

# directional selection constant
beta = 0

# mutation rate and std
mu, sigma_m = 0.01, 0.1

plt_settings = {}
plt_settings["mean_plots"] = False
plt_settings["XY_heatmap"] = True

x0, y0 = 0,0

class Individual:
    def __init__(self, parent: Individual = None):
        if parent == None:
            self.X = x0
            self.Y = y0
        else:
            self.X = parent.X
            self.Y = parent.Y
            if np.random.rand() < mu:
                self.X = np.random.normal(parent.X, sigma_m)
                self.Y = np.random.normal(parent.Y, sigma_m)
    
    def __str__(self):
        return f'({self.X}, {self.Y})'
        
    def K(self): 
        return K0*np.exp(-self.X**2/(2*sigma_k**2))
    
    def alpha(self, other: Individual): 
        if type(other) != Individual: 
            return np.exp(-(self.X - other)/(2*sigma_c**2))
        return np.exp(-(self.X-other.X)**2/(2*sigma_c**2))
        
    def C(self, pop: list[Individual]):
        f, edges = np.histogram([ind.X for ind in pop], n_bins)
        centers = [(edges[i]+edges[i+1])/2 for i in range(len(f))]
        return sum([self.alpha(centers[i])*f[i] for i in range(len(centers))])
        
    def delta(self, pop: list[Individual]):
        Yb = np.mean([other.Y for other in pop])
        delta = self.C(pop)/self.K() - beta*(self.Y - Yb)
        return delta * (delta > 0)
    
    def W(self, pop: list[Individual]):
        return np.exp(r*(1-self.delta(pop)))
    
    def make_offspring(self, pop: list[Individual]) -> list[Individual]:
        return [Individual(self) for _ in range(np.random.poisson(self.W(pop)))]


if __name__ == '__main__':
    
    if len(sys.argv) > 1:
        sweep = sys.argv[1]
        if sweep == "x0_sk":
            x0 = float(sys.argv[2])
            sigma_k = float(sys.argv[3])
            print(f'x0={x0}, sk={sigma_k}')
        if sweep == "x0_sc":
            x0 = float(sys.argv[2])
            sigma_c = float(sys.argv[3])
            print(f'x0={x0}, sc={sigma_c}')
        if sweep == "sk_sc":
            sigma_k = float(sys.argv[2])
            sigma_c = float(sys.argv[3])
            print(f'sk={sigma_k}, sc={sigma_c}')
        
        
    # checking that Individual intialisation works fine
    A = Individual()
    # B = Individual(A)
    
    n_gen = 100
    
    init_pop = [Individual() for _ in range(K0)]
    # run simulation
    
    # data for generation 0
    sim_data = {"Xf":[[ind.X for ind in init_pop]], "Yf":[[ind.Y for ind in init_pop]]}
    
    pop = init_pop
    prog_bar = tqdm.trange(n_gen)
    for n in prog_bar:
        # generate offspring this generation
        offspring = []; list(map(offspring.extend, [ind.make_offspring(pop) for ind in pop]))
        pop = offspring
        # pop = np.random.choice(offspring, K0, replace=False)
        sim_data["Xf"].append([ind.X for ind in pop]), sim_data["Yf"].append([ind.Y for ind in pop])
    
    if len(sim_data["Xf"][-1]) == 0: print("Everyone died horribly.")
    else: print("Survived.")
    
    if plt_settings["mean_plots"]:
        fig, (axX,axN) = plt.subplots(1,2)
        fig.set_figheight(6), fig.set_figwidth(12)
        axY = axX.twinx()
        
        time_scale = range(n_gen+1)
        axX.plot(time_scale, [np.mean(X_values) for X_values in sim_data["Xf"]], color=(1,0,0))
        axY.plot(time_scale, [np.mean(Y_values) for Y_values in sim_data["Yf"]], color=(0,0,1))
        axN.plot(time_scale, [len(Y_values) for Y_values in sim_data["Yf"]], color=(0,0,0), linestyle=':')
        axX.tick_params(axis='y', colors=(1,0,0))
        axY.tick_params(axis='y', colors=(0,0,1))
        axN.tick_params(axis='y', colors=(0,0,0))
        plt.show()
    if plt_settings["XY_heatmap"]:
        fig, (axX,axY, axN) = plt.subplots(1,3)
        fig.set_figheight(6), fig.set_figwidth(12)
        
        x_range = (np.min([min(Xf) for Xf in sim_data["Xf"]]), np.max([max(Xf) for Xf in sim_data["Xf"]]))
        y_range = (np.min([min(Yf) for Yf in sim_data["Yf"]]), np.max([max(Yf) for Yf in sim_data["Yf"]]))
        sns.heatmap(np.array([np.histogram(Xf, n_bins, range=x_range)[0] for Xf in sim_data["Xf"]]), ax=axX, cmap='RdBu_r')
        sns.heatmap(np.array([np.histogram(Yf, n_bins, range=y_range)[0] for Yf in sim_data["Yf"]]), ax=axY, cmap='RdBu_r')
        axN.imshow(np.array([[len(Xf)] for Xf in sim_data["Xf"]]))
        axN.set_xticklabels([])
        axN.set_xticks([])
        axX.set_title("X histogram through time"), axY.set_title("Y histogram through time")
        axN.set_title("pop_density")
        axX.set_xticklabels(np.round(np.linspace(*x_range, len(axX.get_xticks())),2))
        axY.set_xticklabels(np.round(np.linspace(*y_range, len(axY.get_xticks())),2))
        plt.savefig(f"dump/{sweep}_sweep_{x0}_{sigma_k}.png", dpi=300, format='png')
    
    
    
    
    
    #pdb.set_trace()


