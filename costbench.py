import random
import time 
import json 
import getopt
import sys
from hflop import *
from hflop_uncapacitated import *
import solve_clustered
from generate import *
import numpy as np
import scipy.stats as stats

def get_hfl_cost(configuration, solution, model_size, global_rounds_to_converge, unit_costs=False):
  """ Calculate the cost of an HFL solution. """
  C_d = configuration["device_costs"] # NxM matrix
  C_e = configuration["edge_costs"] # vector with M elements
  l = configuration["local_to_global_ratio"] # how many local aggregation rounds per global aggregation round

  cost = 0
  for e in solution["edge_hosts"]:
    # edge host costs
    if unit_costs:
      c = min(1, C_e[e["id"]])
    else:
      c = C_e[e["id"]]
    cost += 2 * c * model_size * global_rounds_to_converge
    
    for d in e["associated_devices"]:
      # device costs
      if unit_costs:
        c = min(1, C_d[d][e["id"]])
      else:
        c = C_d[d][e["id"]]
      cost += 2 * c * model_size * l * global_rounds_to_converge
  
  return cost
  

def get_fl_cost(configuration, model_size, global_rounds_to_converge, unit_costs = False):
  """ Calculate the cost of an FL configuration, assuming that all clients are recruited (ignores T) and that all transmissions cost.
  
  The way this is calculated takes place under the assumption that the lowest cost edge server of the device acts as its proxy, therefore
  the cost per device is that of the edge host
  """
  # num devices
  N = len(configuration["workload"])

  # num edge hosts
  M = len(configuration["capacities"]) 
  C_d = configuration["device_costs"] 
  C_e = configuration["edge_costs"] # vector with M elements
  
  cost = 0
  for i in range(0, N):
    # for each device, find its lowest cost edge host
    eid = C_d[i].index(min(C_d[i]))
    
    if unit_costs:
      c = min(1,C_e[eid])
    else:
      c = C_e[eid]
    cost += 2 * c * model_size * global_rounds_to_converge
  return cost

def run_benchmark(num_clients, iterations = 10, edge_node_spec=10, relative=False, unit_costs = False):
  # fix topology size
  if relative:
    # edge_node_spec in this case represents the ratio of devices/edge nodes
    M = int(num_clients/edge_node_spec)
  else:    
    M = edge_node_spec
     
  N = num_clients
  T = 1
  zero_cost_lan = True
  min_conn_cost = 1 # minimum communication cost 
  max_conn_cost = 1 # maximum communication cost
  MAXDEMAND = 40
  local_to_global_ratio = 2
  
  # how many (global) aggregation rounds it took for model training to converge -- based on our experiments
  hfl_global_rounds_to_converge = 10
  hflops_global_rounds_to_converge = 10
  fl_global_rounds_to_converge = 20
  samples = []
  
  # serialized model size (GRU)
  #model_size = 0.593
  model_size = 1.15
  for i in range(0, iterations):
    MAXCAP = int(2*N*MAXDEMAND/M) # to "make sure" there's enough capacity, but still favor the uncapacitated HFLOP
    configuration = generate(N, M, int(time.time()*1000), participation_ratio = T, local_to_global_ratio = local_to_global_ratio, max_edge_node_capacity = MAXCAP, max_device_demand = MAXDEMAND, zero_cost_lan = zero_cost_lan, min_conn_cost = min_conn_cost, max_conn_cost = max_conn_cost)

    solution_hfl = hflop_uncapacitated(configuration)
    solution_hflop = hflop(configuration, model_version=2)

    if solution_hflop is None:
      # infeasible - skip sample
      #print("Infeasible solution")
      continue
    cost_hfl = get_hfl_cost(configuration, solution_hfl, model_size, hfl_global_rounds_to_converge, unit_costs=unit_costs)
    cost_hflop = get_hfl_cost(configuration, solution_hflop, model_size, hflops_global_rounds_to_converge, unit_costs=unit_costs)
    cost_fl = get_fl_cost(configuration, model_size, fl_global_rounds_to_converge, unit_costs=unit_costs)
    samples.append([cost_hfl, cost_hflop, cost_fl, (cost_fl-cost_hfl)/cost_fl*100, (cost_fl-cost_hflop)/cost_fl*100])
  
  return samples

def run_clustered_topology(iterations, infile, capacities, unit_costs=True):
    # Clustered topology experiment
    # parse cluster information
    device_indexes, hosts, pairs = solve_clustered.get_dev_edge_pairs(infile)

    N = len(device_indexes)
    M = len(capacities)
    
    T = 1
    local_to_global_ratio = 2
    
    # how many (global) aggregation rounds it took for model training to converge -- based on our experiments
    hfl_global_rounds_to_converge = 10
    hflops_global_rounds_to_converge = 10
    fl_global_rounds_to_converge = 20
    samples = []
    
    # serialized model size (GRU)
    #model_size = 0.593
    model_size = 1.15
    for i in range(0, iterations):
      # generate a configuration appropriate for solving
      configuration = solve_clustered.generate(device_indexes, hosts, pairs, int(time.time()*1000), 1.0, len(capacities), capacities, True, 1, 1)

      # solve
      solution_hfl = hflop_uncapacitated(configuration)
      solution_hflop = hflop(configuration, model_version=2)

      if solution_hflop is None:
        # infeasible - skip sample
        #print("Infeasible solution")
        continue

      cost_hfl = get_hfl_cost(configuration, solution_hfl, model_size, hfl_global_rounds_to_converge, unit_costs=unit_costs)
      cost_hflop = get_hfl_cost(configuration, solution_hflop, model_size, hflops_global_rounds_to_converge, unit_costs=unit_costs)
      cost_fl = get_fl_cost(configuration, model_size, fl_global_rounds_to_converge, unit_costs=unit_costs)
      samples.append([cost_hfl, cost_hflop, cost_fl, (cost_fl-cost_hfl)/cost_fl*100, (cost_fl-cost_hflop)/cost_fl*100])
    
    return samples


if __name__ == '__main__':
  # Some experimens are shown below. Uncomment the one(s) you want to run
  
  # fixed number of edge hosts, varying number of clients
  """
  M = 10
  iterations = 10
  for N in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    samples = run_benchmark(N, iterations, edge_node_spec=M, relative=False)
    samples = np.array(samples)
    rows, cols = samples.shape
    # for each column, print mean and confidence interval
    print(N, end="\t")
    for i in range(0, cols):
      data = samples[:, i]
      # get avg and confidence intervals 
      if stats.sem(data) == 0:
        CI = (np.mean(data), np.mean(data))
      else:
        if rows < 30:
          CI = stats.t.interval(confidence=0.95, df=len(data)-1, loc=np.mean(data), scale=stats.sem(data)) 
        else: 
          CI = stats.norm.interval(confidence=0.95, loc=np.mean(data), scale=stats.sem(data)) 
      print(np.mean(data), CI[0], CI[1], end="\t")
    print()
  """  

  # Cost vs edge node density
  """
  M = [2, 4, 8, 16, 32, 64] # devices per edge host
  N = 200
  iterations = 50
  for m in M:
    samples = run_benchmark(N, iterations, edge_node_spec=m, relative=True, unit_costs=True)
    samples = np.array(samples)
    rows, cols = samples.shape
    # for each column, print mean and confidence interval
    print(1/m, end="\t")
    for i in range(0, cols):
      data = samples[:, i]
      # get avg and confidence intervals 
      if stats.sem(data) == 0:
        CI = (np.mean(data), np.mean(data))
      else:
        if rows < 30:
          CI = stats.t.interval(confidence=0.95, df=len(data)-1, loc=np.mean(data), scale=stats.sem(data)) 
        else: 
          CI = stats.norm.interval(confidence=0.95, loc=np.mean(data), scale=stats.sem(data)) 
      print(np.mean(data), CI[0], CI[1], end="\t")
    print(str(1) + ":" + str(m))
    
  """
  
  # Fixed clustered topology
  
  # file with cluster data. should be just columns, not including any heading
  INFILE = "./clusters.dat"
    
  # fixed edge server capacities
  capacities = [40, 120, 40, 120] 
  iterations = 100
  samples = run_clustered_topology(iterations, INFILE, capacities)
  samples = np.array(samples)
  rows, cols = samples.shape
  # print mean and confidence interval
  print("Clustered [4/20]:", end="\t")
  for i in range(0, cols):
    data = samples[:, i]
    # get avg and confidence intervals 
    if stats.sem(data) == 0:
      CI = (np.mean(data), np.mean(data))
    else:
      if rows < 30:
        CI = stats.t.interval(confidence=0.95, df=len(data)-1, loc=np.mean(data), scale=stats.sem(data)) 
      else: 
        CI = stats.norm.interval(confidence=0.95, loc=np.mean(data), scale=stats.sem(data)) 
    print(np.mean(data), CI[0], CI[1], end="\t")
  print()
  
