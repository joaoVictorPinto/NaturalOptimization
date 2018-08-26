
__all__ = ['jade', 'JADESolver']

from Logger import Logger, retrieve_kw
import sys, random, os
from copy import deepcopy, copy
import numpy as np

def Lehmer_mean( v ):
  return sum(v*v)/sum(v) if sum(v)>0.0 else 0.0

def mean( v ):
  return sum(v)/float(len(v)) if sum(v)>0.0 else 0.0


class JADESolver(Logger):

  def __init__(self, **kw):
    Logger.__init__(self, **kw)


  # crossover method for xt and donor
  def crossover(self, feature_a, feature_b, CR_i ):
    trial = list()
    # one sort for each dimesion of the feature
    i_rand = random.randint(0,len(feature_a)-1)
    for idx in range( len(feature_a) ):
      crossover = random.random()
      if i_rand == idx or crossover < CR_i:
        trial.append( feature_b[idx] ) # append donor feature
      else:
        trial.append( feature_a[idx] ) # append target feature

    return np.array(trial)


  # select a set of individual from population
  def candidates( self, population, current_candidates, n_candidates ):
    candidates_idx = range(0, len(population)); 
    candidates_idx = [idx for idx in candidates_idx if idx not in current_candidates]
    return [population[idx] for idx in  random.sample( candidates_idx, n_candidates)]



  def solver(self, population, cost_func, answer, **kw):

    p                  = retrieve_kw( kw, 'p'                 , max(0.05, 3/float(len(population))) )
    c                  = retrieve_kw( kw, 'c'                 , 0.1 )
    mu_cr              = retrieve_kw( kw, 'mu_cr'             , 0.5 )
    mu_f               = retrieve_kw( kw, 'mu_f'              , 0.6 )

    min_error          = retrieve_kw( kw, 'min_error'         , 1e-8)
    maxfes             = retrieve_kw( kw, 'maxfes'            , 1000)
  
    
    gfitness_evol = []; gerror_evol = []
    gbest_individual = None
    gerror = sys.float_info.max # Best error found
    gbest_fitness = sys.float_info.max # Best error found
    
    generation  = 0; nmaxfes = 0

    while nmaxfes < maxfes:
 
      # current generation candidates
      gen_fitness = []
      best_individual = None
      best_individual_idx = None
      best_fitness = sys.float_info.max


      for individual in population:
        nmaxfes+=1
        individual.fitness = cost_func.fitness( individual.feature )[0]
        gen_fitness.append(individual.fitness)
      

      bestp_fitness = sys.float_info.max; bestp_individual = None
      for individual in self.candidates( population, [], int(p*len(population))):
        if individual.fitness < bestp_fitness:
          bestp_fitness=individual.fitness;  bestp_individual=deepcopy(individual)


      one_third = int(len(population)/3.); two_third = len(population)-one_third
      Fi_one_third = 1.2 * np.random.randn( one_third )
      Fi_two_third = 0.1 * np.random.randn( two_third ) + mu_f
      
      # generate 
      S_cr = []; S_f = []
      F  = np.concatenate( (Fi_one_third, Fi_two_third) ); np.random.shuffle(F)
      CR = 0.1 * np.random.randn(len(population)) + mu_cr


      # truncate parameters
      CR[CR>1.0]=1.0; CR[CR<=0.0]=0.05
      F[F>1.2]=1.2; F[F<=0.0]=0.05

      gen_fitness = []
      # calculate the fitness for all individual
      for idx, individual in enumerate(population):
        
        # retrieve three random candidates from the list except the target and best 
        # individual from the current generation
        candidates = self.candidates( population, [idx, best_individual_idx], 2)
        x0 = candidates[0]; x1 = candidates[1]
        
        # mutate simple method
        x_donor = deepcopy( individual )
        # DE/current-best/1 mutation method
        x_donor.feature = individual.feature + F[idx] * (bestp_individual.feature - individual.feature) + F[idx] * (x0.feature - x1.feature)

        # crossover
        x_trial = deepcopy( individual )
        x_trial.feature = self.crossover( individual.feature, x_donor.feature, CR[idx])        
        x_trial.fitness = cost_func.fitness( x_trial.feature )[0]
        nmaxfes+=1

        # selection
        if x_trial.fitness < individual.fitness:
          population[idx] = x_trial
          gen_fitness.append( x_trial.fitness )
          S_f.append( F[idx] ); S_cr.append( CR[idx] )
        else:
          gen_fitness.append( individual.fitness )

      # calculation stat evolution of the current generation
     
      # Update mu_cr and mu_f
      mu_cr = (1-c)*mu_cr + c * mean( np.array(S_cr) )
      mu_f  = (1-c)*mu_f  + c * Lehmer_mean( np.array(S_f) )
      
      # get the best individual since we mutate all candidates
      best_individual = population[ gen_fitness.index(min(gen_fitness)) ] 

      if not gbest_individual:
        gbest_individual = deepcopy( best_individual )
        gbest_fitness = gbest_individual.fitness
      elif best_individual.fitness < gbest_individual.fitness:
        gbest_individual = deepcopy( best_individual )
        gbest_fitness = best_individual.fitness


      # hold some evololution values
      gfitness_evol.append( gbest_individual.fitness )
      # calculate the generation error between the best fit and the expected answer
      gerror = abs(answer - gbest_fitness)
      gerror_evol.append(gerror)
    
      self._logger.info( 'Generation %d, best_fit = %1.2f, error = %1.8f, maxFES = %d', generation, gbest_fitness, gerror, nmaxfes)
      generation+=1

      if gerror < min_error:
        self._logger.warning('Stop loop because the global error is < than min_error parameter.')
        break
    # end of while loop
  
    return best_individual, gfitness_evol, gerror_evol



jade = JADESolver()







