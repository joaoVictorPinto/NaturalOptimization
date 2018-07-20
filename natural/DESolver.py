
__all__ = ['de']

from Logger import Logger, retrieve_kw
import sys, random, os
from copy import deepcopy, copy
import numpy as np




class DESolver(Logger):

  def __init__(self, **kw):
    Logger.__init__(self, **kw)


  # crossover method for xt and donor
  def crossover(self, feature_a, feature_b, recombination_rate ):
    trial = list()
    for idx in range( len(feature_a) ):

      crossover = random.random()
      if crossover > recombination_rate:
        trial.append( feature_a[idx] ) # append target feature
      else:
        trial.append( feature_b[idx] ) # append donor feature

    return np.array(trial)


  # select a set of individual from population
  def candidates( self, population, current_candidate, n_candidates ):
    candidates_idx = range(0, len(population));  candidates_idx.remove(current_candidate)
    return [population[idx] for idx in  random.sample( candidates_idx, n_candidates)]



  def solver(self, population, cost_func, answer, **kw):

    recombination_rate = retrieve_kw( kw, 'recombination_rate', 0.9 )
    mutate_factor      = retrieve_kw( kw, 'mudate_factor'     , 0.5 )
    min_error          = retrieve_kw( kw, 'min_error'         , 1e-8)
    maxfes             = retrieve_kw( kw, 'maxfes'            , 1000)
  
    
    fitness_evolution = []
    best_individual = None
    #fitness_avg_evolution = []
    gerror = sys.float_info.max # Best error found
    best_fitness = sys.float_info.max # Best error found
    generation  = 0; nmaxfes = 0

    while nmaxfes < maxfes:

      generation_fitness = []; 

      # calculate the fitness for all individual
      for idx, individual in enumerate(population):
        
        # retrieve three random candidates from the list
        candidates = self.candidates( population, idx, 3)
        x0 = candidates[0]; x1 = candidates[1]; x2 = candidates[2]
        
        # mutate simple method
        x_donor = deepcopy( individual )
        # DE/rand/1 mutation method
        x_donor.feature = x0.feature + mutate_factor * (x1.feature - x2.feature)

        # crossover
        x_trial = deepcopy( individual )
        x_trial.feature = self.crossover( individual.feature, x_donor.feature, recombination_rate)        
        x_trial.fitness = cost_func.fitness( x_trial.feature )
        individual.fitness = cost_func.fitness( individual.feature )
        nmaxfes+=2

        # selection
        if x_trial.fitness < individual.fitness:
          population[idx] = x_trial
          generation_fitness.append( x_trial.fitness )
        else:
          generation_fitness.append( individual.fitness )

      # calculation stat evolution of the current generation
      
      best_fitness = min(generation_fitness)
      #best_avg_fitness = sum(generation_fitness)/ float(len(population))
      best_individual = deepcopy( population[ generation_fitness.index(best_fitness) ] )
      # hold some evololution values
      fitness_evolution.append( best_fitness )
      #fitness_avg_evolution.append( best_fitness )
      # calculate the generation error between the best fit and the expected answer
      gerror = abs(answer - best_fitness)
    
      self._logger.info( 'Generation %d, best_fit = %1.2f, error = %1.8f, maxFES = %d', generation, best_fitness, gerror, nmaxfes)
      generation+=1

      if gerror < min_error:
        self._logger.info('Stop loop because the global error is < than min_error parameter.')
        break
    # end of while loop
  
    return best_individual, fitness_evolution





de = DESolver()


# test DE solver
dim = 10
population = [ Individual(dim, -100, 100) for _ in range(1000)]
from Prob import CEC2014
cost_function = CEC2014( dim = dim, prob = 1 ) 

de.solver( population, cost_function, 100 , maxfes = dim*10000)





