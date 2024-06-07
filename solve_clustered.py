import random
import time 
import json 
import getopt
import sys
from hflop import *
import copy


# file with cluster data. should be just columns, not including any heading
INFILE = "./clusters.dat"

# where to output a solution to
OUTFILE = "./assignment.json"

# where to output the corresponding generated configuration
CONFIGURATION = "./config.json"

# maximum capacity of an edge server
EDGECAP = 80 # 10 rps, 8 cpu cores
#capacities = [EDGECAP]*4

# fixed edge server capacities
capacities = [40, 120, 40, 120] 

# default minimum-maximum communication cost
MIN_CONN_COST = 1
MAX_CONN_COST = 1

def get_dev_edge_pairs(infile):
  """Parses infile and returns a list of device IDs, a dict with hosts and the devices in their cluster, and pairs of device-cluster associations."""
  pairs = {}
  device_indexes = []
  hosts = {}
  with open(infile) as f:
    for line in f:
      tokens = line.split()
      pairs[tokens[0]] = int(tokens[7])
      device_indexes.append(tokens[0])
      if tokens[7] not in hosts:
        hosts[tokens[7]] = []
      hosts[tokens[7]].append(tokens[0])
      
  return device_indexes, hosts, pairs

def translate_assignment(configuration, assignment, device_indexes):
  translated = copy.deepcopy(assignment)
  
  for e in translated["edge_hosts"]:
    associated_devices = []
    for d in e["associated_devices"]:
      associated_devices.append({"id": device_indexes[d], "workload": configuration["workload"][d]})
    e["associated_devices"] = associated_devices

  return translated

def generate(device_indexes, hosts, pairs, seed = None, participation_ratio = 1.0, local_to_global_ratio = 4, edge_node_capacities = capacities, zero_cost_lan = True, min_conn_cost = MIN_CONN_COST, max_conn_cost = MAX_CONN_COST):
  # seed PRNG
  if not seed:
    seed = int(time.time())
  random.seed(seed)

  N = len(device_indexes)
  M = len(hosts)
  
  max_device_demand = int(1.7*sum(edge_node_capacities)/N)
  
  # Capacity vector R: includes the network capacity for each pair of hosts. This is not assumed symmetric.
  # Cost matrix C_d: c_ij encodes the cost/bit to tx/rx between device i and edge node j
  # Cost vector C_e: c_j denotes the cost/bit to tx/rx between the cloud and edge node j
  R = capacities
  C_d = [1]*N
  C_e = [1]*M
  D = [0]*N
  
  for j in range(0, M):
    C_e[j] = random.randint(min_conn_cost, max_conn_cost) # random costs
    
  for i in range(0, len(device_indexes)):
    if zero_cost_lan:
      # Each device is associated with a single edge node (0 communication cost).
      # Unit costs to all other edge nodes
      device_costs = [1]*M
      device_costs[pairs[device_indexes[i]]] = 0   # e.g. device_indexes[0] = 771667, pairs["771667"] = 0
      C_d[i] = device_costs
    else:
      C_d[i] = [random.randint(min_conn_cost, max_conn_cost) for m in range(0,M)]    
    # each client has a random inference workload
    D[i] = random.randint(0, max_device_demand)
    
  # create scenario
  scenario = {"seed": seed, "capacities": R, "workload": D, "device_costs": C_d, "edge_costs": C_e, "participation_ratio": participation_ratio, "local_to_global_ratio": local_to_global_ratio}
  return scenario

# parse cluster information
device_indexes, hosts, pairs = get_dev_edge_pairs(INFILE)

# generate a configuration appropriate for solving
configuration = generate(device_indexes, hosts, pairs, int(time.time()), 1.0, 4, capacities, True, 1, 1)

# solve and generate an assignment
assignment = hflop(configuration, model_version = 2)

if assignment is None:
  print("No feasible solution found.")
else:
  # format the solution so that it includes the original device IDs as well as workloads
  translated = translate_assignment(configuration, assignment, device_indexes)

# write output to file
fp = open(OUTFILE, "w+")
json.dump(translated, fp, indent=2)
fp.close()

# write configuration to file
fp = open(CONFIGURATION, "w+")
json.dump(configuration, fp, indent=2)
fp.close()

print(json.dumps(translated, indent=2))

