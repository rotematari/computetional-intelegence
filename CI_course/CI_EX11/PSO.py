import numpy as np
import matplotlib.pyplot as plt


 




class PSO():
    def __init__(self, live_plot = False):
        # Hyperparameters of the algorithm
        self.c1 = 0.1 # cognitive acceleration coefficient
        self.c2 = 0.1 # social acceleration coefficient
        self.w = 0.8 # inertia weight
        
        self.live_plot = live_plot # whether to plot the results in real-time
        np.random.seed(100) # set random seed for reproducibility

    def define_problem(self, func = None, n_particles = 20, bounds = []):
        # Define the optimization problem
        self.n_particles = n_particles # number of particles in the swarm
        self.ndim = bounds.shape[1] # dimensionality of the problem

        self.func = func # objective function to minimize
        
        # Initialize particles randomly within the bounds
        self.X = np.random.rand(self.n_particles, self.ndim) * (bounds[1,:]-bounds[0,:]) + bounds[0,:]
        
        # Initialize particle velocities randomly
        self.V = np.random.randn(self.n_particles, self.ndim) * 0.1
        
        self.bounds = bounds

        # Initialize personal best positions as first generation
        self.pbest = self.X.copy()
        
        # Evaluate objective function for personal best positions
        self.pbest_obj = self.func(self.X)
        
        # Initialize global best position as the best personal best position
        self.gbest = self.pbest[self.pbest_obj.argmin(), :].copy()
        
        # Evaluate objective function for global best position
        self.gbest_obj = self.pbest_obj.min()

    def update(self):
        # Update particle positions and velocities
        r1, r2 = np.random.rand(2) # generate random numbers for stochastic update
        V = self.w * self.V + self.c1*r1*(self.pbest - self.X) + self.c2*r2*(self.gbest.reshape(1,-1)-self.X)
        
        X_temp = self.X + self.V
        
        for i in range(self.n_particles): # Check for constraints
            if np.all(X_temp[i,:] > self.bounds[0,:]) and np.all(X_temp[i,:] < self.bounds[1,:]):
                # Update particle position if it is within the bounds
                self.X[i,:] = X_temp[i,:].copy()
                
        obj = self.func(self.X)
        
        # Update personal best positions and objective values
        self.pbest[(self.pbest_obj >= obj), :] = self.X[(self.pbest_obj >= obj), :]
        self.pbest_obj = np.array([self.pbest_obj, obj]).min(axis=0)
        
        # Update global best position and objective value
        self.gbest = self.pbest[self.pbest_obj.argmin(), :]
        self.gbest_obj = self.pbest_obj.min()

    def run(self, max_iter = 1000):
        # Run the optimization algorithm for a specified number of iterations
        self.G = [] # list to store the global best objective value at each iteration
        
        for i in range(max_iter):
            # Update particle positions and velocities
            self.update()
            
            print(f'Iteration {i}, best fit so far: {np.round(self.gbest_obj, 5)}')
            
            # Store global best objective value at this iteration
            self.G.append(self.gbest_obj.copy())
            
            if self.live_plot:
                # Plot results in real-time if live_plot is True
                self.plot(iter = i)

        return self.gbest, self.gbest_obj
    
    def plot(self, iter = None):
        plt.figure(0)
        
        if self.live_plot:
            plt.clf()
            
        plt.plot(np.arange(len(self.G)), self.G)
        
        plt.xlabel('Iterations')
        plt.ylabel('Fit')
        
        if self.live_plot:
            plt.title(f'Best global fit: {np.round(self.gbest_obj, 5)}')
            
            if iter > 1:
                plt.xlim([0, iter])
                
            plt.pause(1e-1)
            
        else:
            plt.show()




if __name__ == "__main__":
    def f(z):
        x = z[:, 0]
        y = z[:, 1]
        return x**2 + (y+1)**2 - 5*np.cos(1.5*x+1.5) - 3*np.cos(2*y-1.5)

    bounds = np.array([[-5, -5], [5, 5]])

    P = PSO(live_plot = True)
    P.define_problem(func = f, n_particles = 10, bounds = bounds)
    P.run(max_iter = 100)