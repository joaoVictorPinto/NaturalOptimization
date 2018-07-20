
__all__ = ['Individual', 'Particle']


class Individual(object):

  def __init__(self, dim, minx, maxx):
    self._bounds   = (minx, maxx)
    self._feature  = np.array([random.uniform(minx,maxx) for _ in range(dim)])
    self._fitness  = sys.float_info.max 
  
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


class Particle:

  def __init__(self, dim, minx, maxx):
    self.dim      = dim
    self.velocity = np.zeros(dim)
    self.position = np.zeros(dim)
    self.pbest    = np.zeros(dim)
    self.pfitness = sys.float_info.max
    self.__random_init(dim,minx,maxx)
    self._bounds  = (minx, maxx)

  # Initialize the particle position in the random space
  def __random_init(self, dim, minx, maxx):
    self.position = ((maxx-minx)*np.random.random(dim) + minx)

  def bounds(self):
    return self._bounds


