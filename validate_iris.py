

import numpy as np
import matplotlib.pyplot as plt


from mpl_toolkits.mplot3d import Axes3D
from matplotlib           import cm
from matplotlib.ticker    import LinearLocator, FormatStrFormatter


def plot_hist( signal, noise, nbins, xmin, xmax, name ):
  fig =plt.figure()
  bins = np.linspace(xmin, xmax, nbins)
  plt.hist(signal, bins, alpha=0.5)
  plt.hist(noise, bins, alpha=0.5)

  plt.xlabel('nn output')
  plt.ylabel('counter')
  plt.title('Neural Network output')
  plt.grid(True)
 
  #plt.show()
  plt.savefig(name)
  plt.close()

def plot_fitness( fit, name ):
  epoch = np.arange(0, len(fit), 1)
  fig =plt.figure()
  plt.plot(epoch, fit)

  plt.xlabel('Epoch')
  plt.ylabel('Fitness')
  plt.title('Best fitness')
  plt.grid(True)
  plt.savefig(name)
  #plt.show()
  plt.close()


def plot_scatter( class_1, class_2 ):

  fig =plt.figure()
  plt.scatter(class_1[:,0],class_1[:,1],c = 'b', alpha=0.5)  
  plt.xlabel('X')
  plt.ylabel('Y')
  plt.title('Class 1')
  plt.grid(True)
  plt.savefig('iris_class_1.pdf')
  #plt.show()
  plt.close()


  fig =plt.figure()
  plt.scatter(class_2[:,0],class_2[:,1],c = 'r', alpha=0.5)  
  plt.xlabel('X')
  plt.ylabel('Y')
  plt.title('Class 2')
  plt.grid(True)
  plt.savefig('iris_class_2.pdf')
  plt.close()




#******************** PSO Solver ********************
from PSOSolver     import neural_mse_fitness, minor_error, pso, Particle, neural_sp_fitness
from NeuralNetwork import NeuralNetwork
from sklearn import datasets

iris       = datasets.load_iris()
class_1   = iris.data[0:50, :2].T  # we only take the first two features.
class_2   = iris.data[50:100, :2].T  # we only take the first two features.

plot_scatter( class_1.T, class_2.T)

nparticles = 100
neural     = NeuralNetwork([2,2,1],['tansig','tansig'])
particles = list()
for i in range(nparticles):
  particles.append( Particle( neural.get_dim(), -10.0, 10.0) )

fitness_args = [class_1, class_2, neural, 0]

out_sgn = neural(class_1)
out_bkg = neural(class_2)

plot_hist( out_sgn, out_bkg, 100,-1,1, 'iris_nnoutput_before.pdf')

#PSO Trainer
gbest, fitness_evo = pso.solver( particles, neural_mse_fitness ,minor_error, 
                                             max_generations = 100,
                                             inertia        = 0.79,
                                             screenshot     = True,
                                             show           = 1,
                                             maxvel         = 10,
                                             fitness_args   = fitness_args)


#Lets make the neural network analysis
neural.set_position( gbest )
out_sgn = neural(class_1)
out_bkg = neural(class_2)

plot_hist( out_sgn, out_bkg, 100,-1,1, 'iris_nnoutput_after.pdf')


#screenshot = pso.get_screenshot()
#last = len(screenshot)-1
#plot_evolution( Ackley_fitness, screenshot[0], 'ackley_generation_0.pdf')
#plot_evolution( Ackley_fitness, screenshot[1], 'ackley_generation_1.pdf')
#plot_evolution( Ackley_fitness, screenshot[10], 'ackley_generation_10.pdf')
#plot_evolution( Ackley_fitness, screenshot[last], 'ackley_generation_1000.pdf')
plot_fitness(fitness_evo,'iris_best_fitness.pdf')







