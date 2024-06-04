from random import *
import time 
import json 
import getopt
import sys
import random

MAXCAP = 100 # max edge node capacity (rps)
MAXDEMAND = 10 # per client inference workload (rps)
ZERO_COST_LAN = False
MAX_CONN_COST = 10 # max connectivity cost
MIN_CONN_COST = 0 # min connectivity cost

def generate(N, M, seed = None, participation_ratio = 1.0, local_to_global_ratio = 4, max_edge_node_capacity = MAXCAP, max_device_demand = MAXDEMAND, zero_cost_lan = ZERO_COST_LAN, min_conn_cost = MIN_CONN_COST, max_conn_cost = MAX_CONN_COST):
  """ Generate a topology with N clients & M edge nodes."""
  # seed PRNG
  if not seed:
    seed = int(time.time())
  random.seed(seed)

  # Capacity vector R: includes the network capacity for each pair of hosts. This is not assumed symmetric.
  # Cost matrix C_d: c_ij encodes the cost/bit to tx/rx between device i and edge node j
  # Cost vector C_e: c_j denotes the cost/bit to tx/rx between the cloud and edge node j
  R = [0]*M
  C_d = [1]*N
  C_e = [1]*M
  D = [0]*N
  
  for j in range(0, M):
    R[j] = randint(0, max_edge_node_capacity) # random capacity
    #C_e[j] = 1 # for now, all transmissions between edge-cloud cost a unit
    C_e[j] = randint(0, max_conn_cost) # random costs
    
  for i in range(0, N):
    if zero_cost_lan: 
      # Each device is associated with a single edge node (0 communication cost).
      # Unit costs to all other edge nodes
      device_costs = [1]*M
      device_costs[randint(0,M-1)] = 0
      C_d[i] = device_costs
    else:
      C_d[i] = [randint(0, max_conn_cost) for m in range(0,M)]    
    # each client has a random inference workload
    D[i] = randint(0, max_device_demand)
    
  # create scenario
  scenario = {"seed": seed, "capacities": R, "workload": D, "device_costs": C_d, "edge_costs": C_e, "participation_ratio": participation_ratio, "local_to_global_ratio": local_to_global_ratio}
  return scenario

if __name__ == "__main__":
  # Get configuration from command line. If a seed is not provided, the current time is used
  # e: number of edge nodes
  # d: number of devices
  # s: seed
  # o: output path
  N = None # devices
  M = None # edge nodes
  seed = None
  outpath = None
  participation_rate = None
  local_to_global_ratio = None
  max_edge_node_capacity = None
  max_device_demand = None
 
  myopts, args = getopt.getopt(sys.argv[1:], "e:d:s:o:p:l:C:D:")
  for o, a in myopts:
    if o == "-e":
      M = int(a)
    if o == "-d":
      N = int(a)
    if o == "-s":
      seed = int(a)
    if o == "-o":
      outpath = a
    if o == "-p":
      participation_ratio = float(a)
    if o == "-l":
      local_to_global_ratio = int(a)
    if o == "-C":
      max_edge_node_capacity = int(a)
    if o == "-D":
      max_device_demand = int(a)
      
  if not N or not M:
    print("Missing paramenter.")
    sys.exit(1)

  scenario = generate(N, M, seed, participation_ratio, local_to_global_ratio, max_edge_node_capacity, max_device_demand)
  if outpath:
    fp = open(outpath, "w+")
    json.dump(scenario, fp, indent=2)
    fp.close()
  else:
    print(json.dumps(scenario, indent=2))

