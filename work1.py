# Trabalho de CPE 726 - Topicos Especiais em Inferencia em Grafos
# Author: Joao Victor da Fonseca Pinto

from simulator import Simulator ,SimulationMethod
from simulator import Node, Graph, LoggingLevel



prior_U = [0.3, 0.7] # F or T
prior_V = [0.4, 0.6]
prior_R = [0.5, 0.5]
prior_X = [[[0.3,0.7],[0.2,0.8]],[[0.7,0.3],[0.6,0.4]]]
prior_Z = [[0.3,0.7],[0.2,0.8]]
prior_W = [[[0.2,0.8],[0.4,0.6]],[[0.6,0.4],[0.7,0.3]]]


U = Node('U', prior_U)
V = Node('V', prior_V)
R = Node('R', prior_R)
X = Node('X', prior_X, parents = [U,V])
Z = Node('Z', prior_Z, parents = [X]  )
W = Node('W', prior_W, parents = [X,R])

# Use this method to set an evidence.
U.set_evidence(1) # can be 0 or 1
R.set_evidence(0) # can be 0 or 1


net = Graph([U,V,X,R,Z,W])

MC = Simulator( net, 20000, method = SimulationMethod.LikelyhoodWeighting , level = LoggingLevel.INFO )
#MC = Simulator( net, 20000, method = SimulationMethod.LikelyhoodWeighting , level = LoggingLevel.DEBUG )
#MC = Simulator( net, 20000, method = SimulationMethod.LogicalSampling , level = LoggingLevel.INFO )
MC.initialize()
MC.execute()
# output all probs for all nodes
MC.finalize()





