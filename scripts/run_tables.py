
from natural import Logger
mainLogger = Logger()
mainLogger = mainLogger.getModuleLogger()
import pickle
import numpy as np


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


output_names = ['DE_statistics_10.txt', 
                'DE_statistics_30.txt', 
                'JADE_statistics_10.txt', 
                'JADE_statistics_30.txt',
                'PSO_statistics_10.txt', 
                'PSO_statistics_30.txt', 
                'DEPSO_statistics_10.txt',
                'DEPSO_statistics_30.txt',
                'CMAES_statistics_10.txt',
                'CMAES_statistics_30.txt',
                ]

title = [' DE (10D) ', 
         ' DE (30D) ', 
         ' JADE (10D) ', 
         ' JADE (30D) ',
         ' PSO (10D) ',
         ' PSO (30D) ',
         ' DEPSO (10D) ',
         ' DEPSO (30D) ',
         ' CMAES (10D) ',
         ' CMAES (30D) ',
         
         ]
probs = ['F1','F2','F6','F7','F9','F14']


f_stroutput = '| {0:<8} | {1:<17} | {2:<17} | {3:<17} | {4:<17} | {5:<17} | {6:<15} |'.format(
    'Funtion',
    'Best',
    'Worst',
    'Median',
    'Mean',
    'Std',
    'Success Rate')


percentage = [0.0, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


### Create main table
for idx, fname in enumerate(files):

  foutput = open(output_names[idx],'w')
  f = pickle.load(open(fname,'r'))
  top_stroutput = '+{:-^128}+'.format(title[idx])
  botton_stroutput = '+{:-^128}+'.format('-')
  mainLogger.info( top_stroutput )
  mainLogger.info( f_stroutput )
  foutput.write(top_stroutput+'\n')
  
  for key in probs:
    obj = f[key]
    stroutput = '| {0:<8} | {1:<17} | {2:<17} | {3:<17} | {4:<17} | {5:<17} | {6:<15} |'.format(
                '%s'%key,
                '%1.4f'%obj['error_best'], 
                '%1.4f'%obj['error_worst'],
                '%1.4f'%obj['error_median'],
                '%1.4f'%obj['error_mean'],
                '%1.4f'%obj['error_std'],
                '%1.4f'%obj['success_rate'])
    mainLogger.info(stroutput)
    foutput.write(stroutput+'\n')
  

    pfile = open('test.txt','w')
    for p in percentage:
      stroutput = '| {0:<15} |'.format('Erro para FES=%1.2f*MaxFES'%p)
      for jdx in range(51):
        error_evol = obj['runs']['error_evol'][jdx]
        stroutput+=' {0:<17} |'.format( error_evol[ int(p*(len(error_evol)-1))])
      stroutput+=' {0:<17} |\n'.format(np.mean(error_evol))
    pfile.write(stroutput)
    pfile.close()

  
  mainLogger.info( botton_stroutput )
  foutput.write(botton_stroutput)
  foutput.close()








