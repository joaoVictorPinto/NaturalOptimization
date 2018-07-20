


__all__ = ['Arkley', 'CEC2014']

from Logger import retrieve_kw
import numpy as np
import math

class Arkley(object):

  def __init__(self, **kw):
    pass

  def fitness( self, position ):
    first = np.power(position,2).sum()
    second = np.cos( 2*math.pi*position ).sum()
    n = float(len(position))
    return -20.0 * np.exp( -0.2* np.sqrt(first/n) ) - np.exp(second/n) + 20 + math.e



# Take from: https://github.com/esa/pagmo2/blob/master/include/pagmo/problems/cec2014.hpp
class CEC2014(object):

  def __init__( self, **kw ):
    import pygmo as pg
    dim     = retrieve_kw( kw, 'dim', 2 )
    prob = retrieve_kw( kw, 'prob', 1)
    self._cec2014_core = pg.problem(pg.cec2014( prob, dim ))

  def fitness( self, position ):
    return self._cec2014_core.fitness( position )



