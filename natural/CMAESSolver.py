
__all__ = ['cmaes', 'CMAESSolver']

from Logger import Logger, LoggingLevel, EnumStringification, retrieve_kw, NotSet
import numpy as np
import sys
import random

from math import sqrt, exp, expm1, floor, log



class CMAESSolver(Logger):

  def __init__(self, **kw):
    Logger.__init__(self,**kw)
 
  def solver(self, N, cost_function, answer, **kw):
    
    while True:
      try:
        gbest, fitness_evol, error_evol = self._solver(N,cost_function,answer,**kw)
        break
      except:
        pass
    return gbest, fitness_evol, error_evol

  def _solver(self, N, cost_function, answer, **kw):   

      gfitness_evol = []; gerror_evol = []
      stopeval             = retrieve_kw( kw, 'maxfes'            , 10000)


      #Container for coordinates called xmean
      xmean=[]                                  #Objective variables (coordinates)initial point
      for i in range(int(N)):
          xmean.append([random.uniform(0,1)]);

      xmean=np.asarray(xmean,dtype=np.float_)
      
      sigma=0.5                              #Coordinate wise standard deviation
      stopfitness = 1e-8                    #stop if fitness < stopfitness (minimization)
  
      #Strategy parameter setting: Selection
      lambd=int(4.0+floor(3.0*log(N)))                #population size, offspring number
      mu=(lambd/2)                           #number of parents/points for recombination
      weights=np.array([])
      for j in range(mu):
          weights=np.append(weights,log(mu+0.5)-log(j+1))#muXone array for weighted recombination
      #mu = floor(mu)
      mu = int(mu)
      weights=weights/sum(weights)
      mueff=sum(weights)**2/sum(weights**2)
      weights_array=weights.reshape([mu,1])  
  
      #Strategy parameter setting: Adaptation
      cc=(4+mueff/N)/(N+4 + 2*mueff/N)      #time constant for cumulation for C
      cs=(mueff+2)/(N+mueff+5)              #t-const for cumulation for sigma control 
      c1=2/((N+1.3)**2+mueff)               #learning rate for rank-one update of C
      cmu=min(1-c1,2*(mueff-2+1/mueff)/((N+2)**2+mueff)) #and for rank-mu update
      damps =1 +2*max(0,sqrt((mueff-1)/(N+1))-1)+cs  #damping for sigma, usually close to 1
  
      #Initialize dynamic (internal) strategy parameters and constants
      pc=np.zeros((N,1),dtype=np.float_)                      #evolution path for C  
      ps=np.zeros((N,1),dtype=np.float_)                      #evolution path for sigma 
      D=np.ones((N,1),dtype=np.float_)                        #diagonal D defines the scaling  
      B =np.eye(N,dtype=np.float_)                #B defines the coordinate system
      D_sqd=D**2           
      diag_D_sqd=(np.diag(D_sqd[:,0]))       #Generate diagonal matrix with D_sqd in the diagonal, the rest are zeros 
      trans_B=B.transpose()                 #Tranpose matrix of B 
      C=np.dot(np.dot(B,diag_D_sqd),trans_B)#Covariance matrix C
      nega_D=D**(-1)                        #Elementwise to power -1 
      diag_D_neg=(np.diag(nega_D[:,0]))     #Generate diagonal matrix with nega_D in diagonal, rest zeros
      invsqrtC =np.dot(np.dot(B,diag_D_neg),trans_B)#C^-1/2 
      eigeneval = 0                         #track update of B and D
      chiN=N**0.5*(1-1/(4*N)+1/(21*N**2))   #expectation of ||N(0,I)|| == norm(randn(N,1)) 
 

      generation=0
      counteval=0
      arx=[]

      while counteval<stopeval:
          #Generate and evaluate lamba offspring
          generation+=1
          itera=1
          arfitness=[]
          for l in range(lambd):
              offspring=[]                  #Create a container for the offspring 
              offspring=xmean+np.dot(sigma*B,(D*np.random.standard_normal((N,1)))) #m + sig * Normal(0,C)
              if itera==1:
                  arx=offspring
              else:
                  arx=np.hstack((arx,offspring))
              arfitness.append(cost_function.fitness(offspring.reshape((N,))))          #EVALUATE OBJ FUNCTION
              counteval=counteval+1
              itera=itera+1   
          
          #Sort by fitness and compute weighted mean into xmean
          ordered=[]
          ordered=sorted(enumerate(arfitness), key=lambda x: x[1])    #minimization, list with (Index,Fitness) elements
          xold=xmean
          best_off_indexes=[]
          for i in range(int(mu)):
              best_off_indexes.append(ordered[i][0]) #List of best indexes of mu offspring
          recomb=np.zeros([N,mu])
          cont=0
          for index in best_off_indexes:
              recomb[:,cont]=arx[:,index]
              cont=cont+1
          xmean=np.dot(recomb,weights_array)
  
          #Cumulation: update evolution paths
          ps=(1-cs)*ps+sqrt(cs*(2-cs)*mueff)*np.dot(invsqrtC,(xmean-xold))*(1/sigma)
          hsig=np.linalg.norm(ps)/sqrt(1-(1-cs)**(2*counteval/lambd))/chiN < 1.4 + 2/(N+1)
          if hsig==True:
              hsig=1
          else:
              hsig=0
          pc=(1-cc)*pc+hsig*sqrt(cc*(2-cc)*mueff)*(xmean-xold)/sigma

          #Adapt covariance matrix C
          artmp=(1/sigma)*(recomb-np.tile(xold,(1,mu)))
          weights_diag=np.diag(weights_array[:,0])
          C=(1-c1-cmu)*C+c1*(np.dot(pc,pc.transpose())+((1-hsig)*cc*(2-cc)*C))+cmu*np.dot(artmp,np.dot(weights_diag,artmp.transpose()))
  
          #Adapt step size sigma
          sigma=sigma*exp((cs/damps)*(np.linalg.norm(ps)/chiN - 1))
  
          #Decomposition of C into B*diag(D.^2)*B' (diagonalization)
          if counteval-eigeneval > lambd/(c1+cmu)/N/10:
              eigeneval=counteval                    #to achieve O(N^2)
              C=np.triu(C)+np.triu(C,1).transpose()  #enforce symmetry
              B=np.linalg.eig(C)[1]                  #eigen decomposition, B==normalized eigenvectors
              diag_D=np.linalg.eig(C)[0]
              diag_D=diag_D.reshape([N,1])
              D=diag_D**0.5                       #D is a vector of standard deviations now
              D_inv=D**(-1)
              D_inv=np.diag(D_inv[:,0])
              invsqrtC=np.dot(B,np.dot(D_inv,B.transpose()))
         

          gfitness_evol.append( ordered[0][1] )
          gerror_evol.append( abs(answer-ordered[0][1]) )
          self._logger.info( 'Generation %d, best_fit = %1.2f, error = %1.8f, maxFES = %d', generation, ordered[0][1],\
                  abs(answer-ordered[0][1]), counteval)
          
          #Break, if fitness is good enough or condition exceeds 1e14, better termination methods are advisable           
          del arfitness
          if  abs(answer - ordered[0][1]) <= stopfitness or max(D)>1e7*min(D):
            self._logger.info('Stop interation since error is lower than min_error')
            break
          #Return best point at last iteration

      xmin=arx[:,best_off_indexes[0]]
      return xmin, gfitness_evol, gerror_evol





cmaes = CMAESSolver()

#************************** Main *****************************
#D=30
#from Prob import CEC2014, Arkley
#cost_function = CEC2014( dim = D, prob = 2 )
#answer=200
#cmaes.solver( D, cost_function ,answer) 




