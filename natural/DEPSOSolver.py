
__all__ = ['depso', 'DEPSOSolver']

from Logger import Logger, retrieve_kw
import sys, random, os
from copy import deepcopy, copy
import numpy as np




class DEPSOSolver(Logger):

  def __init__(self, **kw):
    Logger.__init__(self, **kw)


  # select a set of individual from population
  def candidates( self, population, current_candidate, n_candidates ):
    candidates_idx = range(0, len(population));  candidates_idx.remove(current_candidate)
    return [population[idx] for idx in  random.sample( candidates_idx, n_candidates)]



  def solver(self, particles, cost_func, answer, **kw):

    recombination_rate = retrieve_kw( kw, 'recombination_rate', 0.9     )
    C1                 = retrieve_kw( kw, 'C1'                , 2.05    )
    C2                 = retrieve_kw( kw, 'C2'                , 2.05    )
    W                  = retrieve_kw( kw, 'W'                 , 0.7298  ) 
    min_error          = retrieve_kw( kw, 'min_error'         , 1e-8    )
    maxfes             = retrieve_kw( kw, 'maxFES'            , 1000    )
    del kw 

    gfitness_evol = []; gerror_evol = []
    gbest = None
    gerror = sys.float_info.max # Best error found
    gfitness = sys.float_info.max # Best error found

    for pi in particles:
      pi.fitness =cost_func.fitness( pi.position )
      perror = abs(answer-pi.fitness)
      pi.gbest = pi.position; pi.gfitness=pi.fitness
      if perror < gerror:
        gbest = copy( pi ); gerror=perror; gfitness=pi.fitness


    
  
    generation  = 0; nmaxfes = 0
    while nmaxfes < maxfes:

      gen_fitness = []; 
      # calculate the fitness for all individual
      for pi in particles:
        pi.velocity = W * pi.velocity + C1 * random.random() * (pi.gbest       - pi.position) + \
                                        C2 * random.random() * (gbest.position - pi.position)
        pi.position += pi.velocity
        pfitness =cost_func.fitness( pi.position )
        nmaxfes+=1
        perror = abs(answer-pfitness)
        if perror < abs(answer-pi.fitness):
          pi.pfitness = pfitness; pi.pbest = pi.position
        if perror < gerror:
          gpbest = copy( pi ); gerror=perror; gfitness=pfitness

      
      for idx, pi in enumerate(particles):

        # retrieve three random candidates from the list
        candidates = self.candidates( particles, idx, 2)
        p0 = candidates[0]; p1 = candidates[1];
        
        # mutate simple method
        p_trial = deepcopy( pi )
        # Crossover
        i_rand = random.randint(0,len(p_trial.velocity)-1)
        for i in range(len(p_trial.velocity)):
          crossover = random.random()
          if crossover <= recombination_rate or i == i_rand:
            p_trial.position[i] = gbest.position[i] + (p0.pbest[i]-p1.pbest[i])/2.0
            
        p_trial.fitness = cost_func.fitness( p_trial.position )
        nmaxfes+=1

        # selection
        if abs(answer-p_trial.fitness) < abs(answer-pi.fitness):
          particles[idx] = p_trial
          gen_fitness.append( p_trial.fitness )
        else:
          gen_fitness.append( pi.fitness )


      # calculation stat evolution of the current generation      
      best_fitness = min(gen_fitness)
      #best_avg_fitness = sum(generation_fitness)/ float(len(population))
      gbest_individual = deepcopy( particles[ gen_fitness.index(best_fitness) ] )
      # hold some evololution values
      gfitness_evol.append( best_fitness )
      #fitness_avg_evolution.append( best_fitness )
      # calculate the generation error between the best fit and the expected answer
      gerror = abs(answer - best_fitness)
      gerror_evol.append(gerror)

      self._logger.info( 'Generation %d, best_fit = %1.2f, error = %1.8f, maxFES = %d', generation, best_fitness, gerror, nmaxfes)
      generation+=1

      if gerror < min_error:
        self._logger.warning('Stop loop because the global error is < than min_error parameter.')
        break
    # end of while loop
  
    return gbest_individual, gfitness_evol, gerror_evol





depso = DEPSOSolver()

#************************** Main *****************************
#
#
#from Population import Particle
#particles = list()
#D=10
#for i in range(20):
#  particles.append( Particle( D, -500, 500) )
#
#from Prob import CEC2014, Arkley
#cost_function = CEC2014( dim = D, prob = 2 )
##cost_function = Arkley()
#answer=200
#
#gbest, f_evolution, error_evolution = depso.solver( particles, 
#                                 cost_function ,
#                                 answer, 
#                                 maxFES=100000
#                                )
#
#





