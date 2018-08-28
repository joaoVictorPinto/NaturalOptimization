

def SuccessRate( runs_error, min_error=1e-8):
  occurrences = 0
  for error in runs_error:
    if error <= min_error:
      occurrences+=1
  return occurrences/float(len(runs_error))

from natural import *
from copy import deepcopy
import numpy as np
import pickle

# configurations for work1
nruns = 51
probs = [1,2,6,7,9,14]
answer= [100,200,600,700,900,1400]
alg_objects = [de, jade, pso, depso, cmaes]
alg_names = ['DE','JADE','PSO','DEPSO','CMAES']

# Loop for D=10 and DE algorithm
dim = 30

for alg_idx, alg_obj in enumerate(alg_objects):

  fsummary = {'F1':None,'F2':None,'F6':None,'F7':None,'F9':None,'F14':None}
  for idx, prob in enumerate(probs):
  
    rsummary = {'error':[],'individual':[], 'error_evol':[]}
  
    for jdx in range(nruns):
      cost_function = CEC2014( dim = dim, prob = prob )

      #individual, fitness_evol, error_evol = de.solver(population,cost_function,answer[idx], maxfes = 10000*dim)
      
      if type(alg_obj) is CMAESSolver: 
        individual, fitness_evol, error_evol = alg_obj.solver(dim,cost_function,answer[idx], maxfes = 10000*dim)
      elif type(alg_obj) is PSOSolver:
        particles = [ Particle(dim, -100, 100) for _ in range(20)]
        individual, fitness_evol, error_evol = alg_obj.solver(particles,cost_function,answer[idx], maxfes = 10000*dim)
      elif type(alg_obj) is DEPSOSolver:
        particles = [ Particle(dim, -100, 100) for _ in range(20)]
        individual, fitness_evol, error_evol = alg_obj.solver(particles,cost_function,answer[idx], maxfes = 10000*dim)
      else:
        population = [ Individual(dim, -100, 100) for _ in range(20)]
        individual, fitness_evol, error_evol = alg_obj.solver(population,cost_function,answer[idx], maxfes = 10000*dim)

      rsummary['individual'].append(individual)
      rsummary['error'].append(error_evol[-1])
      rsummary['error_evol'].append(error_evol)
  
    # get all stat values from all runs
    psummary = {}
    psummary['error_best'] = min(rsummary['error'])
    psummary['error_worst'] = max(rsummary['error'])
    psummary['error_median'] = np.median(rsummary['error'])
    psummary['error_mean'] = np.mean(rsummary['error'])
    psummary['error_std'] = np.std(rsummary['error'])
    psummary['success_rate'] = SuccessRate( rsummary['error'] )
    psummary['runs'] = rsummary
    fsummary['F%d'%prob] = deepcopy(psummary)
  
  
  #f = open('JADE_30D_summary.pic','wb')
  f = open('%s_%dD_summary.pic' % (alg_names[alg_idx],dim),'wb')
  pickle.dump(fsummary,f)




