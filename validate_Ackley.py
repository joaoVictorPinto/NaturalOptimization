

import numpy as np
import matplotlib.pyplot as plt


from mpl_toolkits.mplot3d import Axes3D
from matplotlib           import cm
from matplotlib.ticker    import LinearLocator, FormatStrFormatter


def plot_surface_3d( func , name ):
  fig = plt.figure()
  ax = fig.gca(projection='3d')
  x = y  = np.arange(-5, 5, 0.25)
  X, Y = np.meshgrid(x, y)
  zs = np.array([func(np.array([x,y])) for x,y in zip(np.ravel(X), np.ravel(Y))])
  zs = np.array([func(np.array([x,y])) for x,y in zip(np.ravel(X), np.ravel(Y))])
  Z = zs.reshape(X.shape)
  surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
  ax.set_zlim(0, 12)
  ax.zaxis.set_major_locator(LinearLocator(10))
  ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
  fig.colorbar(surf, shrink=0.5, aspect=5)
  #plt.show()
  plt.savefig(name)
  plt.close()

  
def plot_evolution( func , solutions, name):
  fig = plt.figure()
  ax = fig.gca(projection='3d')
  x = y  = np.arange(-5, 5, 0.25)
  X, Y = np.meshgrid(x, y)
  zs = np.array([func(np.array([x,y])) for x,y in zip(np.ravel(X), np.ravel(Y))])
  zs = np.array([func(np.array([x,y])) for x,y in zip(np.ravel(X), np.ravel(Y))])
  Z = zs.reshape(X.shape)
  surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
  ax.set_zlim(0, 12)
  ax.zaxis.set_major_locator(LinearLocator(10))
  ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
  fig.colorbar(surf, shrink=0.5, aspect=5)

  #plot the particles
  for solution in solutions:
    ax.scatter(solution[0], solution[1], func(solution)); 

  #plt.show()
  plt.savefig(name)
  plt.close()

def plot_fitness( fit, name ):
  epoch = np.arange(0, len(fit), 1)
  plt.plot(epoch, fit)

  plt.xlabel('Epoch')
  plt.ylabel('Fitness')
  plt.title('Best fitness')
  plt.grid(True)
  plt.savefig(name)
  #plt.show()
  plt.close()



#******************** PSO Solver ********************

from PSOSolver import Ackley_fitness, minor_error, pso, Particle
nparticles = 100

particles = list()
for i in range(nparticles):
  particles.append( Particle( 2, -10, 10 ) )


gbest, fitness_evo = pso.solver( particles, Ackley_fitness ,minor_error, 
                                             max_generations = 1000,
                                             inertia        = 3,
                                             screenshot     = True,
                                             show           = 1,
                                             maxvel         = 10)

screenshot = pso.get_screenshot()
last = len(screenshot)-1
plot_evolution( Ackley_fitness, screenshot[0], 'ackley_generation_0.pdf')
plot_evolution( Ackley_fitness, screenshot[1], 'ackley_generation_1.pdf')
plot_evolution( Ackley_fitness, screenshot[10], 'ackley_generation_10.pdf')
plot_evolution( Ackley_fitness, screenshot[last], 'ackley_generation_1000.pdf')
plot_fitness(fitness_evo[1::],'best_fitness.pdf')







