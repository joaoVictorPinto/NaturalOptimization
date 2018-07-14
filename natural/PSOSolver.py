
__all__ = ['Particle', 'pso']

from Logger import Logger, LoggingLevel, EnumStringification, retrieve_kw, NotSet
import numpy as np
import math
import sys
import copy


# helper function to display a progress bar
def progressbar(it, prefix="", size=60):
    count = len(it)
    def _show(_i):
        x = int(size*_i/count)
        sys.stdout.write("%s[%s%s] %i/%i\r" % (prefix, "█"*x, "."*(size-x), _i, count))
        sys.stdout.flush()
    _show(0)
    for i, item in enumerate(it):
        yield item
        _show(i+1)
    sys.stdout.write("\n")
    sys.stdout.flush()



def minor_error(error, best_error):  
  return error < best_error



class Particle:

  def __init__(self, dim, minx, maxx):
    self.dim      = dim
    self.velocity = np.zeros(dim)
    #self.velocity = ((maxx-minx)*np.random.random(dim) + minx)
    self.position = ((maxx-minx)*np.random.random(dim) + minx)
    self.pbest    = np.zeros(dim)
    self.pfitness = sys.float_info.max



class PSOSolver:
  
  _screenshot = list()

  def __init__(self):
    pass

  def calculate(self, p, gbest):
    r1 = np.random.random()
    r2 = np.random.random()
    return ( (self.w  * p.velocity) + \
             (self.c1 * r1 * (p.pbest - p.position) ) + \
             (self.c2 * r2 * (gbest   - p.position)) )  


  def solver( self, solutions, fitness, is_the_best, **kw ):
    
    self.w       = kw.pop('inertia'     , 0.729)
    self.c1      = kw.pop('cognitive'   , 2    )
    self.c2      = kw.pop('social'      , 2    )
    maxvel       = kw.pop('maxVel'      , 10   )
    #minx         = kw.pop('minx'        , -1   )
    #maxx         = kw.pop('maxx'        , 1    )
    doScreenShot = kw.pop('screenshot'  , False)
    show         = kw.pop('show'        , 10   )
    fitness_args = kw.pop('fitness_args', None )
    max_generations = kw.pop('max_generations', 1000)

    gbest    = None #Best global particle
    gfitness = sys.float_info.max #Best error found
    fitness_evolution = list()
    generation  = 0
    pscreenshot = list()


    #Initialize all structures
    for p in solutions:
      f = fitness(p.position, fitness_args)
      if is_the_best( f, p.pfitness):
        p.pfitness = f
        p.pbest  = copy.copy(p.position)
      if is_the_best( f, gfitness ):
        gfitness = f
        gbest  = copy.copy(p.position)
      #screenshot for the first time
      if doScreenShot:
        pscreenshot.append(copy.copy(p.position))
      
    if doScreenShot:  self._screenshot.append(pscreenshot)
    #screenshot trigger ctrl by show
    trigger=False

    #Loop
    while generation < max_generations:

      #Status display
      if generation % show == 0 and generation > 1:
        print ('Generation = %d, best fit = %.10f')%(generation, gfitness)
        if doScreenShot:  trigger=True;
      #Screenshot
      if trigger:  pscreenshot = list()

      fitness_evolution.append(gfitness)
      #Loop over particles
      for p in solutions:    
        
        #Apply fitness
        f = fitness(p.position, fitness_args)

        if is_the_best( f, p.pfitness):
          p.pfitness = f
          p.pbest  = copy.copy(p.position)
        if is_the_best( f, gfitness ):
          gfitness = f
          gbest  = copy.copy(p.position)
  
        #Calculate velocity
        p.velocity = self.calculate(p,gbest)
        
        #Limits for velocity
        p.velocity[p.velocity >    maxvel] =    maxvel
        p.velocity[p.velocity < -1*maxvel] = -1*maxvel

        #Update position
        p.position += p.velocity
        
        #Limits for position
        #p.position[p.position > maxx] = -0.1*maxx
        #p.position[p.position < minx] = -0.1*minx

        if trigger:  pscreenshot.append(copy.copy(p.position))

      #end particles
      if trigger:  self._screenshot.append(pscreenshot)
      trigger=False
      generation+=1
    #end epochs

    print 'gbest position: ', gbest
    return gbest, fitness_evolution

  def get_screenshot(self):
    return self._screenshot

pso = PSOSolver()

#************************** Main *****************************
#from PSOSolver import Ackley_fitness, minor_error, pso
#particles = list()
#for i in range(100):
#  particles.append( Particle( 2, -10, 10 ) )
#pso.solver( particles, Ackley_fitness ,minor_error, max_generations = 2000, screenshoot = True)

