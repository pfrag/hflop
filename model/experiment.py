# Various example experiments. Uncomment the one you want to run, and configure the parameters accordingly.

from generate import *
from hflop import *
from greedy import *
from knapsack import *
import numpy as np
import scipy.stats as stats

iterations = 10
participation_ratio = 0.75
local_to_global_ratio = 4
MAXDEMAND = 10 # per client inference workload (rps)


# Fixed M, varying N
"""
M = 10
for N in [10, 100, 1000, 10000, 100000]:
  runtime = 0
  for i in range(0, iterations):
    MAXCAP = 10*N*MAXDEMAND/M # to "make sure" there's enough capacity
    scenario = generate(N, M, int(time.time()), participation_ratio, local_to_global_ratio, MAXCAP, MAXDEMAND)
    solution = hflop(scenario)
    runtime += solution["execution_time"]
  print(N, float(runtime)/iterations)
"""

#####################

# fixed N, varying M
"""
N = 10
MAXCAP = 10*MAXDEMAND*N
for M in [10, 100, 1000, 10000, 100000]:
  runtime = 0
  for i in range(0, iterations):
    #MAXCAP = 10*N*MAXDEMAND/M # to "make sure" there's enough capacity
    scenario = generate(N, M, int(time.time()), participation_ratio, local_to_global_ratio, MAXCAP, MAXDEMAND)
    solution = hflop(scenario)
    runtime += solution["execution_time"]
  print(M, float(runtime)/iterations)
"""

#####################

# fixed device/edge ratio, varying #devices
"""
device_edge_ratio = 10

for N in [10, 100, 1000, 10000, 100000]:
  M = device_edge_ratio*N
  runtime = 0
  for i in range(0, iterations):
    MAXCAP = 10*N*MAXDEMAND/M # to "make sure" there's enough capacity
    scenario = generate(N, M, int(time.time()), participation_ratio, local_to_global_ratio, MAXCAP, MAXDEMAND)
    solution = hflop(scenario)
    runtime += solution["execution_time"]
  print(N, float(runtime)/iterations)
"""

####################

# compare greedy vs. hflop for a fixed scenario
# run many times to check optimality gap
"""
M = 100
N = 10000
MAXCAP = 10000
MAXDEMAND = 10
iterations = 10
print("Optimal\tHeuristic\toptimality gap\tExec time diff")
for i in range(0, iterations):
  # generate topology
  scenario = generate(N, M, int(time.time()), participation_ratio, local_to_global_ratio, MAXCAP, MAXDEMAND)  
  
  # run algorithms for this topology & compare results
  opt = hflop(scenario)
  heur = greedy_cheapest_first(scenario, prune_devices=True)
  strO = ""
  if opt is None:
    strO = "-"
  if heur is None:
    strO += "*"
  if opt is not None and heur is not None:
    gap = heur["objective"] - opt["objective"]
    tdiff = opt["execution_time"] - heur["execution_time"]
    strO = str(opt["objective"]) + "\t" + str(heur["objective"]) + "\t\t" + str(gap) + "\t" + str(opt["execution_time"]) + " : " + str(heur["execution_time"])
  print(strO)

"""

# Running time experiments

iterations = 10
T = 0.7 # FL participation ratio
l = 4 # ratio of local to global rounds
zero_cost_lan = False # if true, each client has a single edge loc. for which it has zero cost and cost is 1 unit towards the rest. Otherwise, cost is drawn randomly
min_conn_cost = 0 # minimum communication cost 
max_conn_cost = 10 # maximum communication cost
MAXDEMAND = 20
verbose = False # print more info about solutions

# M N avg ci-min  ci-max
M = [100]
N = [10000]
for m in M:
  for n in N:
    samples = []
    for i in range(0, iterations):
      MAXCAP = 10*n*MAXDEMAND/m # to "make sure" there's enough capacity
      scenario = generate(n, m, int(time.time()), participation_ratio = T, local_to_global_ratio = l, max_edge_node_capacity = MAXCAP, max_device_demand = MAXDEMAND, zero_cost_lan = False, min_conn_cost = min_conn_cost, max_conn_cost = max_conn_cost)
      solution = hflop(scenario, model_version = 2)
      samples.append(solution["execution_time"])
      print(solution["execution_time"])
    # get avg and confidence intervals 
    if iterations < 30:
      CI = stats.t.interval(confidence=0.95, df=len(samples)-1, loc=np.mean(samples), scale=stats.sem(samples)) 
    else: 
      CI = stats.norm.interval(confidence=0.95, loc=np.mean(samples), scale=stats.sem(samples)) 
    print(m, n, np.mean(samples), CI[0], CI[1])


