import time
import sys
import getopt
import json
import math
from operator import itemgetter, attrgetter
from greedy import *
from knapsack import *
from hflop import *

def _get_edge_host_cost(configuration, e):
  """ Returns the cost that edge node e with its associated devices contributes to a solution.
  
  e is assumed to be a dict including fields "id" (integer) and "associated_devices" (list of integers)
  """
  C_d = configuration["device_costs"]
  C_e = configuration["edge_costs"]
  l = configuration["local_to_global_ratio"]

  cost = 0
  for i in e["associated_devices"]:
    cost += l*C_d[i][e["id"]]
  cost += C_e[e["id"]]
  return cost

def _check_edge_host_feasibility(configuration, e):
  """ Check the feasibility of including edge host e in a solution.
  
  e is assumed to be a dict including fields "id" (integer) and "associated_devices" (list of integers).
  This function checks if the capacity constraint of e is violated by this assignment.
  """
  R = configuration["capacities"]
  D = configuration["workload"]

  load = 0
  for d in e["associated_devices"]:
    load += D[d]
  if load <= R[e["id"]]:
    return True
  else:
    return False

def _num_associated_devices(solution):
  num = 0
  for e in solution["edge_hosts"]:
    num += len(e["associated_devices"])
  return num

def _count_associated_devices(N, solution):
  num = 0
  for i in range(0, N):
    for e in solution["edge_hosts"]:
      if i in e["associated_devices"]:
        num += 1
        break      
  return num

def local_search(configuration):
  """ Solves HFLOP using a local search-like algorithm.

  This works as follows:
  * Phase 1: Find a feasible solution using a greedy algorithm
  
  * Phase 2: Iteratively merge edge hosts
    - Pick any two edge hosts
    - Solve a knapsack problem for each host for all associated devices of the two hosts, and pick the best solution if that reduces the obj. function value
    - Repeat until only two edge hosts are left, or if no improvement can be made after evaluating all current edge host pairs for merging
    
  * Phase 3: Prune excess clients, i.e., clients that are not necessary given the FL participation constraint
  """
  
  ##########################################
  # Configuration and variable setup
  ##########################################
  start = time.time()
  
  # num devices
  N = len(configuration["workload"])

  # num edge hosts
  M = len(configuration["capacities"]) 

  R = configuration["capacities"] # vector with M elements
  C_d = configuration["device_costs"] # NxM matrix
  C_e = configuration["edge_costs"] # vector with M elements
  D = configuration["workload"] # vector with N elements
  l = configuration["local_to_global_ratio"] # how many local aggregation rounds per global aggregation round
  T = int(configuration["participation_ratio"]*N) # minimum number of participating devices

  # Phase 1: Get a feasible solution
  S = greedy_cheapest_first(configuration, prune_devices = False)

  # Phase 2: Loop through pairs of edge hosts and try to merge them
  # 1. Pick the 1st pair
  # 2. create a set of all associated devices
  # 3. Run 2 knapsacks, one for each edge nodes
  # 4. Check if the two solutions are feasible
  # 5. Check if removing the 2 edge nodes and replacing them with the best knapsack result improves cost.
  # If so, use the min-cost solution, update edge_node set, go back to 1. Otherwise, pick another pair.
  i = 0
  num_merge_steps = 0
  while len(S["edge_hosts"]) > 2 and i < len(S["edge_hosts"]):
    j = i
    while j < len(S["edge_hosts"]):
      if i == j:
        j += 1
        continue
      grouped_devices = S["edge_hosts"][i]["associated_devices"] + S["edge_hosts"][j]["associated_devices"]
      original_cost = _get_edge_host_cost(configuration, S["edge_hosts"][i]) + _get_edge_host_cost(configuration, S["edge_hosts"][j])
      kp1 = knapsack_danzig(configuration, S["edge_hosts"][i]["id"], grouped_devices)
      kp2 = knapsack_danzig(configuration, S["edge_hosts"][j]["id"], grouped_devices)
      cost1 =_get_edge_host_cost(configuration, kp1) 
      cost2 =_get_edge_host_cost(configuration, kp2)
      num_associated1 = _num_associated_devices(S) - len(grouped_devices) + len(kp1["associated_devices"])
      num_associated2 = _num_associated_devices(S) - len(grouped_devices) + len(kp2["associated_devices"])
      
      if not _check_edge_host_feasibility(configuration, kp1) or num_associated1 < T:
        cost1 = math.inf
      if not _check_edge_host_feasibility(configuration, kp2) or num_associated2 < T:
        cost2 = math.inf

      if cost1 < cost2 and cost1 < original_cost:
        id1 = S["edge_hosts"][i]["id"]
        id2 = S["edge_hosts"][j]["id"]
        S["edge_hosts"] = [e for e in S["edge_hosts"] if e["id"] != id1 and e["id"] != id2]
        S["edge_hosts"].append(kp1)
        i = 0
        j = 0
        num_merge_steps += 1
        break        
      elif cost2 < cost1 and cost2 < original_cost:
        id1 = S["edge_hosts"][i]["id"]
        id2 = S["edge_hosts"][j]["id"]
        S["edge_hosts"] = [e for e in S["edge_hosts"] if e["id"] != id1 and e["id"] != id2]
        S["edge_hosts"].append(kp2)
        i = 0
        j = 0
        num_merge_steps += 1
        break
      else:
        j += 1
        # do nothing, keep existing solution and move to the next pair
    i += 1  

  # Phase 3: Prune unnecessary devices
  # 1. Collect all device associations
  # 2. Sort associated devices by cost
  # 3. Start removing devices until population threshold is reached.
  devices = []
  for e in S["edge_hosts"]:
    for d in e["associated_devices"]:
      devices.append({"id": d, "demand": D[d], "associated": e["id"], "association_cost": C_d[d][e["id"]]})
  
  devices = sorted(devices, key=itemgetter('association_cost'), reverse=True)  
  used_devices = len(devices)
  
  for d in devices:
    if used_devices <= int(T):
      break
    for e in S["edge_hosts"]:
      if e["id"] == d["associated"]:
        e["associated_devices"].remove(d["id"])
        used_devices -= 1 # we check if after this the edge node becomes empty later
        
  end = time.time()
  
  # get obj function value & filter out unused edge hosts
  objective = 0
  used_edge_hosts = []
  for e in S["edge_hosts"]:
    if len(e["associated_devices"]) > 0:
      objective += C_e[e["id"]]
      for d in e["associated_devices"]:
        objective += l*C_d[d][e["id"]]
      used_edge_hosts.append(e)
    
  return {"edge_hosts": used_edge_hosts, "objective": objective, "execution_time": end - start, "num_merge_steps": num_merge_steps}


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

  # solve the problem in three ways
  opt = hflop(configuration) # optimal
  if not opt:
    print("No feasible solution")
    sys.exit(1)
  ls_ass = local_search(configuration) # local search heuristic
  greedy_ass =  greedy_cheapest_first(configuration, prune_devices = True) # greedy heuristic

  # output solutions
  C_d = configuration["device_costs"]
  D = configuration["workload"]
  R = configuration["capacities"]
  N = len(configuration["workload"])
  
  print("[OPT] Hosts used: ", len(opt["edge_hosts"]), "objective:", opt["objective"], "execution time:", opt["execution_time"])
  print("-------------")
  devs = 0 
  for e in opt["edge_hosts"]:
    devs += len(e["associated_devices"])
    load = sum( D[x] for x in e["associated_devices"])
    print("Host ", e["id"], ":", len(e["associated_devices"]), "load/capacity:", str(load) + "/" + str(R[e["id"]]))
  print("Devices recruited:", str(devs))

  print("##########")

  print("[LS] Hosts used: ", len(ls_ass["edge_hosts"]), "objective:", ls_ass["objective"], "execution time:", ls_ass["execution_time"], "merge steps:", ls_ass["num_merge_steps"])
  print("-------------")
  devs = 0
  for e in ls_ass["edge_hosts"]:
    load = sum( D[x] for x in e["associated_devices"])
    devs += len(e["associated_devices"])
    print("Host ", e["id"], ":", len(e["associated_devices"]), "load/capacity:", str(load) + "/" + str(R[e["id"]]))
  print("Devices recruited:", str(devs))
  
  print("##########")

  print("[GREEDY] Hosts used: ", len(greedy_ass["edge_hosts"]), "objective:", greedy_ass["objective"], "execution time:", greedy_ass["execution_time"])
  print("-------------")
  
  devs = 0 
  for e in greedy_ass["edge_hosts"]:
    load = sum( D[x] for x in e["associated_devices"])
    devs += len(e["associated_devices"])
    print("Host ", e["id"], ":", len(e["associated_devices"]), "load/capacity:", str(load) + "/" + str(R[e["id"]]))
  print("Devices recruited:", devs)

