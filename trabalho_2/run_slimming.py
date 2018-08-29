
from natural import Logger
mainLogger = Logger()
mainLogger = mainLogger.getModuleLogger()
import pickle
import numpy as np

def FixArrayFormat( l ):
  x = list()
  for ll in l:
    try:
      x.append(ll[0])
    except:
      x.append(ll)
  return x 


files = [
          'DE_10D_summary.pic',
          'DE_30D_summary.pic',
          'JADE_10D_summary.pic',
          'JADE_30D_summary.pic',
          'PSO_10D_summary.pic',
          'PSO_30D_summary.pic',
          'DEPSO_10D_summary.pic',
          'DEPSO_30D_summary.pic',
          'CMAES_10D_summary.pic',
          'CMAES_30D_summary.pic',
        ]

probs = ['F1','F2','F6','F7','F9','F14']

solves = {}

for idx, fname in enumerate(files):
  
  k = pickle.load(open(fname,'r'))
  key = fname.replace('_summary.pic','')
  plots = {}
  for prob in probs:

    best_idx = FixArrayFormat(k[prob]['runs']['error']).index( k[prob]['error_best']  )
    worst_idx =FixArrayFormat(k[prob]['runs']['error']).index( k[prob]['error_worst'] )

    print k[prob]['runs']['individual'][best_idx]
    plots[prob] = { 
                    'best_evol'     : FixArrayFormat(k[prob]['runs']['error_evol'][best_idx]),
                    'worst_evol'    : FixArrayFormat(k[prob]['runs']['error_evol'][worst_idx]),
                    'champion'      : k[prob]['runs']['individual'][best_idx] ,
                    'loser'         : k[prob]['runs']['individual'][worst_idx] ,
                    'individuals'   : k[prob]['runs']['individual'],
                    'errors'        : FixArrayFormat(k[prob]['runs']['error']),
                    'error_mean'    : k[prob]['error_mean'],
                    'error_std'     : k[prob]['error_std'],
                    'error_median'  : k[prob]['error_median'],
                    'success_rate'  : k[prob]['success_rate'],
                  }
  solves[key] = plots

summary = open('solves_summary.pic','wb')
pickle.dump(solves,summary)



  








