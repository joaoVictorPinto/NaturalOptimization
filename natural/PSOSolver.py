
__all__ = ['pso']

from Logger import Logger, LoggingLevel, EnumStringification, retrieve_kw, NotSet
import numpy as np
import math
import sys
import copy



class PSOSolver(Logger):
  
  _screenshot = list()

  def __init__(self, **kw):
    Logger.__init__(self,**kw)
    

  def calculate(self, p, gbest):
    r1 = np.random.random()
    r2 = np.random.random()
    return ( (self.w  * p.velocity) + \
             (self.c1 * r1 * (p.pbest - p.position) ) + \
             (self.c2 * r2 * (gbest   - p.position)) )  


  def solver( self, solutions, cost_function, is_the_best, **kw ):
 
    target       = kw.pop('target'      , 0.0  )
    self.w       = kw.pop('inertia'     , 0.729)
    self.c1      = kw.pop('cognitive'   , 2    )
    self.c2      = kw.pop('social'      , 2    )
    maxvel       = kw.pop('maxVel'      , 10   )
    maxFES       = kw.pop('maxFES'      , 1000 )
    #minx         = kw.pop('minx'        , -1   )
    #maxx         = kw.pop('maxx'        , 1    )
    doScreenShot = kw.pop('screenshot'  , False)
    show         = kw.pop('show'        , 10   )
    max_generations = kw.pop('max_generations', 1000000)

    gbest    = None #Best global particle
    gfitness = None #Best error found
    gerror = sys.float_info.max #Best error found
    fitness_evolution = list()
    generation  = 0
    pscreenshot = list()


    #Initialize all structures
    for p in solutions:
      f = cost_function.fitness(p.position)
      error = abs(target - f)
      if error < abs(target-p.pfitness):
        p.pfitness = f
        p.pbest  = copy.copy(p.position)

      if not gfitness:
        gfitness = f; gbest = copy.copy(p.position)
      elif  error < abs(target-gfitness): 
        gfitness = f
        gbest  = copy.copy(p.position)
      
      #screenshot for the first time
      if doScreenShot:
        pscreenshot.append(copy.copy(p.position))
      
    if doScreenShot:  self._screenshot.append(pscreenshot)
    #screenshot trigger ctrl by show
    trigger=False

    #Loop
    generation=0; cmaxFES=0;
    while gerror>1e-8:

      #Status display
      if generation % show == 0 and generation > 1:
        self._logger.info('Generation = %d, best fit = %.10f, error = %.10f (maxFEX=%d)' ,generation, gfitness,gerror, cmaxFES)
        if doScreenShot:  trigger=True;
      #Screenshot
      if trigger:  pscreenshot = list()

      fitness_evolution.append(gfitness)
      #Loop over particles
      for p in solutions:    
        #Apply fitness
        cmaxFES+=1
        f = cost_function.fitness(p.position) 
        error = abs(target - f)

        if error < abs(target-p.pfitness):
          p.pfitness = f
          p.pbest  = copy.copy(p.position)
        
        if error < abs(target-gfitness): 
          gfitness = f
          gbest  = copy.copy(p.position)
          gerror = error

        #Calculate velocity
        p.velocity = self.calculate(p,gbest)
        
        #Limits for velocity
        #p.velocity[p.velocity >    maxvel] =    maxvel
        #p.velocity[p.velocity < -1*maxvel] = -1*maxvel

        #Update position
        p.position += p.velocity
        #Limits for position
        bounds = p.bounds()
        p.position[p.position > bounds[1]] = bounds[1]
        p.position[p.position < bounds[0]] = bounds[0]

        if trigger:  pscreenshot.append(copy.copy(p.position))
        
        if cmaxFES>maxFES:
          self._logger.warning('Number of Max FES reached. Stop pso solver...')
          return gbest, fitness_evolution

      #end particles
      if trigger:  self._screenshot.append(pscreenshot)
      trigger=False
      generation+=1
    #end epochs

    #self._logger.info( 'gbest position: ', gbest )
    return gbest, fitness_evolution

  def get_screenshot(self):
    return self._screenshot

pso = PSOSolver()

#************************** Main *****************************


#from Particle import Particle
#particles = list()
#for i in range(100):
#  particles.append( Particle( 10, -100, 100 ) )
#from Prob import CEC2014
#cost_function = CEC2014( dim = 10, prob = 6 ) 
#
#
#minor_error = None
#gbest, f_evolution = pso.solver( particles, 
#                                 cost_function ,
#                                 minor_error, 
#                                 screenshoot = True, 
#                                 show=1,
#                                 cognitive=0.9, 
#                                 social=0.4, 
#                                 inertia=.95,
#                                 target=600., 
#                                 maxvel=0*4*100., 
#                                 maxFES=1000000
#                                )






