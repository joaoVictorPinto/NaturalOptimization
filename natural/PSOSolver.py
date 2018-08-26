
__all__ = ['pso', 'PSOSolver']

from Logger import Logger, LoggingLevel, EnumStringification, retrieve_kw, NotSet
import numpy as np
import math
import sys
import copy



class PSOSolver(Logger):
  

  def __init__(self, **kw):
    Logger.__init__(self,**kw)
    

  def calculate(self, p, gbest):
    r1 = np.random.rand()
    r2 = np.random.rand()
    return ( (self.w  * p.velocity) + \
             (self.c1 * r1 * (p.pbest - p.position) ) + \
             (self.c2 * r2 * (gbest   - p.position)) )  


  def solver( self, particles, cost_function, answer, **kw ):
 
    self.w       = kw.pop('inertia'     , 0.7298  )
    self.c1      = kw.pop('cognitive'   , 2.05    )
    self.c2      = kw.pop('social'      , 2.05    )
    maxFES       = kw.pop('maxFES'      , 1000    )


    gbest    = None #Best global particle
    gfitness = sys.float_info.max #Best error found
    gerror = sys.float_info.max #Best error found
    fitness_evolution = list()
    error_evolution = list()
    generation  = 0
  
    for p in particles:    
      f = cost_function.fitness(p.position) 
      # Local error
      current_error = abs(answer - f)
      p.pfitness = f; p.pbest  = copy.copy(p.position)
      # is the current particle better than the global particle position
      if current_error < gerror: 
        gfitness = f
        gbest  = copy.copy(p.position)
        gerror = current_error


    #Loop
    generation=0; cmaxFES=0;
    while gerror>1e-8:

      #Status display
      self._logger.info('Generation = %d, best fit = %.10f, error = %.10f (maxFEX=%d)' ,generation, gfitness,gerror, cmaxFES)

      
      #Loop over particles
      for p in particles:    
        
        #Calculate velocity
        p.velocity = self.calculate(p,gbest)        
        #Update position
        p.position += p.velocity

        #Apply fitness
        cmaxFES+=1
        f = cost_function.fitness(p.position) 
        p.fitness = f
        # Local error
        current_error = abs(answer - f)
       
        # is the currenct interation better than the local position
        if current_error < abs(answer-p.pfitness):
          p.pfitness = f
          p.pbest  = p.position
        
        # is the current particle better than the global particle position
        if current_error < gerror: 
          gfitness = f
          gbest  = copy.copy(p.position)
          gerror = current_error

      fitness_evolution.append(gfitness)
      error_evolution.append(gerror)

      if cmaxFES>maxFES:
        self._logger.warning('Number of Max FES reached. Stop pso solver...')
        return gbest, fitness_evolution, error_evolution


      #end particles
      generation+=1
    #end epochs

    #self._logger.info( 'gbest position: ', gbest )
    return gbest, fitness_evolution, error_evolution


pso = PSOSolver()

#************************** Main *****************************
#
#
#from Population import Particle
#particles = list()
#D=10
#for i in range(100):
#  particles.append( Particle( D, -100, 100) )
#from Prob import CEC2014, Arkley
#cost_function = CEC2014( dim = D, prob = 6 )
##cost_function = Arkley()
#answer=600
#
#gbest, f_evolution = pso.solver( particles, 
#                                 cost_function ,
#                                 answer, 
#                                 cognitive= 1.4962, 
#                                 social=1.4962, 
#                                 inertia=0.7298,
#                                 maxFES=100000
#                                )






