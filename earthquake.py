# Author: Joao Victor da Fonseca Pinto

from simulator import Simulator ,SimulationMethod
from simulator import Node, Graph, LoggingLevel


prior_B = [0.3, 0.7]
prior_E = [0.4, 0.6]
prior_P = [0.5, 0.5]
prior_A = [[[0.3, 0.7], [0.2, 0.8]], [[0.7, 0.3], [0.6, 0.4]]]
prior_J = [[[0.2, 0.8], [0.4, 0.6]], [[0.6, 0.4], [0.7, 0.3]]]
prior_M = [[0.3, 0.7], [0.2, 0.8]]


B = Node('B', prior_B)
E = Node('E', prior_E)
P = Node('P', prior_P)
A = Node('A', prior_A, parents = [B, E])
J = Node('J', prior_J, parents = [P, A])
M = Node('M', prior_M, parents = [A]   )

# Use this method to set an evidence.
#B.set_evidence(0) # can be 0 or 1


net = Graph([B, E, P, A, J, M])

#MC = Simulator( rede, 20000, method = SimulationMethod.LikelyhoodWeighting , level = LoggingLevel.DEBUG )
MC = Simulator( net, 20000, method = SimulationMethod.LogicalSampling , level = LoggingLevel.INFO )
MC.initialize()
MC.execute()
# output all probs for all nodes
MC.finalize()





