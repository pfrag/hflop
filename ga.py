import time
import sys
import getopt
import json
from operator import itemgetter, attrgetter
import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.operators.sampling.rnd import IntegerRandomSampling

class HFLOPGA(ElementwiseProblem):
  """ Genetic algorithm to solve HFLOP."""
  
  def __init__(self, configuration):
    self.configuration = configuration
    N = len(configuration["workload"])
    M = len(configuration["capacities"]) 
    super().__init__(n_var= N*M + M, n_obj=1, n_ieq_constr=3*M+N+1, xl=0.0, xu=1.0)

  def _evaluate(self, x, out, *args, **kwargs):
    # num devices
    N = len(self.configuration["workload"])

    # num edge hosts
    M = len(self.configuration["capacities"]) 

    R = self.configuration["capacities"] # vector with M elements
    C_d = self.configuration["device_costs"] # NxM matrix
    C_e = self.configuration["edge_costs"] # vector with M elements
    D = self.configuration["workload"] # vector with N elements
    l = self.configuration["local_to_global_ratio"] # how many local aggregation rounds per global aggregation round
    T = self.configuration["participation_ratio"]*N # minimum number of participating devices

    # Each element of x looks like: x11...x1m x21... x2m ... xn1..xnm y1 y2 ...
    # x has #solutions elements

    # vectorize cost matrix and create one cost vector for everything
    Cvec = np.concatenate((np.reshape(C_d, N*M), np.array(C_e)))

    outF = []
    outG = []

    # calculate objective function value   
    obj = np.dot(Cvec, x)
    outF.append(obj)
          
    # calculate constraints
    for j in range(0, M): # constraints foreach edge host (j)
    
      # constraints 
      s = 0
      d = 0
      ii = 0 # helper counter
      for i in range(j, M*N, M):
        s += x[i] # Sum over xij for fixed j
        d += x[i]*D[ii] # Same, to calculate total demand to edge host j
        ii += 1
      outG.append(x[M*N+j] - s) # Constraint 1: y_j - sum(xij) <= 0
      outG.append(s - N*x[M*N+j]) # Constraint 2: sum(xij) - bigM * yj <= 0, setting bigM = N
      outG.append(d - R[j]) # Constraint 3: demand - capacity <= 0
   
    for i in range(0, N*M, M): # constraints for each edge device (i)
      s = 0
      for j in range(i, i + M):
        s += x[j]
      outG.append(s - 1) # Constraint 4: sum(xij) - 1 <= 0
    
    s = 0
    for i in range(0, N*M):
      s += x[i]
    outG.append(T - s) # Constraint 5: T - sum(xij) <= 0

    out["F"] = outF
    out["G"] = outG

def export_solution(configuration, solution, solution_time=None):
  N = len(configuration["workload"])
  M = len(configuration["capacities"]) 
  R = configuration["capacities"]
  D = configuration["workload"]

  assignment = {}
  edge_hosts = []

  X = np.reshape(solution.X[0:N*M], (N,M))
    
  # for each edge node, show devices assigned
  for j in range(0, M):
    host = {"id": j}
    used_capacity = 0
    available_capacity = R[j]
    associated_devices = []

    for i in range(0,N):
      if X[i][j] == 1:
        associated_devices.append(i)
        used_capacity += D[i]
    host["associated_devices"] = associated_devices
    host["available_capacity"] = available_capacity
    host["used_capacity"] = used_capacity
    edge_hosts.append(host)
  assignment["edge_hosts"] = edge_hosts
  assignment["objective"] = solution.F[0]
  assignment["execution_time"] = solution_time
  
  return assignment


def hflop_ga(configuration):
  h = HFLOPGA(configuration)
  
  # solve the problem
  method = GA(pop_size=100,
              sampling=IntegerRandomSampling(),
              crossover=SBX(prob=1.0, eta=3.0, vtype=float, repair=RoundingRepair()),
              mutation=PM(prob=1.0, eta=3.0, vtype=float, repair=RoundingRepair()),
              eliminate_duplicates=True,
              )

  start = time.time()
  res = minimize(h,
                 method,
                 termination=('n_gen', 500),
                 seed=int(time.time()),
                 save_history=True,
                 verbose=True,
                 )
  end = time.time()
  
  return export_solution(configuration, res, end-start)


if __name__ == '__main__':    
  # open configuration file
  myopts, args = getopt.getopt(sys.argv[1:], "c:")
  configfile = None
  for o, a in myopts:
    if o == "-c":
      configfile = a

  if configfile == None:
    print("Missing configuration file. Usage: python greedy.py -c /path/to/configfile")
    exit(1)

  try:
    with open(configfile) as fp:
      configuration = json.load(fp)
  except:
    print("Error loading scenario file. Possibly malformed...")

  assignment = hflop_ga(configuration)
  
  # output optimal solution
  print(json.dumps(assignment, indent=2))
  
