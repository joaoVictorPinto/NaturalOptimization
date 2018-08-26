
__all__ = ['Individual', 'Particle']

import numpy as np
import random, sys


class Individual(object):

  def __init__(self, dim, minx, maxx):
    self._bounds    = (minx, maxx)
    self._feature   = np.array([random.uniform(minx,maxx) for _ in range(dim)])
    self._velocity  = np.array([random.uniform(minx,maxx) for _ in range(dim)])
    self._fitness   = sys.float_info.max 

  @property
  def bounds(self):
    return self._bounds

  @property
  def fitness(self):
    return self._fitness

  @fitness.setter
  def fitness(self,v):
    self._fitness=v

  @property
  def feature(self):
    return self._feature
   
  @feature.setter
  def feature(self,v):
    self._feature= self.ensure_bounds(v)
  
  def ensure_bounds(self,v):
    # upper bound 
    v[v>self._bounds[1]] = self._bounds[1]
    # down bound
    v[v<self._bounds[0]] = self._bounds[0]
    return v




class Particle(object):

  def __init__(self, dim, minx, maxx):
    import random
    self.dim      = dim
    self.bounds   = (minx, maxx)
    self.velocity = np.zeros(dim)
    self.position = np.array( [random.uniform(minx,maxx) for _ in range(dim)] )
    self.fitness = sys.float_info.max

    # best local position
    self.pbest    = np.zeros(dim)
    self.pfitness = sys.float_info.max


  @property
  def position(self):
    return self._position
   
  @position.setter
  def position(self,v):
    self._position= self.ensure_bounds(v)
  

  def ensure_bounds(self,v):
    # upper bound 
    v[v>self.bounds[1]] = self.bounds[1]
    # down bound
    v[v<self.bounds[0]] = self.bounds[0]
    return v












