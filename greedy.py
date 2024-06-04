import time
import sys
import getopt
import json
from operator import itemgetter, attrgetter

def greedy_pack(configuration):
  """ Solves the HFLOP instance specified by configuration using a FFD-like heuristic.

  Tries to find a feasibly solution as follows:
  - Sort edge hosts in decreasing order of capacity
  - For each host, sort devices by some criterion (demand * cost) and pack as many as can fit
  - Check if # associated devices > T, otherwise return null (infeasible)
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
  T = configuration["participation_ratio"]*N # minimum number of participating devices

  edge_hosts = []
  devices = []
  for j in range(0, M):
    edge_hosts.append({"id": j, "cost": C_e[j], "available_capacity": R[j], "associated_devices": [], "used_capacity": 0})
  
  for i in range(0, N):
    devices.append({"id": i, "demand": D[i]})
        

  # Sort edge nodes by capacity. (MlogM)
  edge_hosts = sorted(edge_hosts, key=itemgetter('available_capacity'), reverse=True)
  
  # keep track of how many devs have been assigned
  used_devices = 0

  # Pick the highest capacity node j and fill up with devices, and keep going until constraints are met
  done = False
  for e in edge_hosts:
    # Sort devices by workload*(cost to j). Prioritize cheap and low-demand devices
    for d in devices:
      #d["score"] = (1+d["demand"])*(1+C_d[d["id"]][e["id"]])
      #d["score"] = C_d[d["id"]][e["id"]]
      d["score"] = ["demand"]
      d["associated"] = False
    devices = sorted(devices, key=itemgetter("score"), reverse=True)

    # Assign as many users as can fit from the cheapest to the most expensive until capacity filled up. If a device does not fit, skip it.
    for d in devices:
      if d["demand"] <= e["available_capacity"] - e["used_capacity"]: # if device can fit
        e["associated_devices"].append(d["id"]) # associate device with edge node 
        e["used_capacity"] += d["demand"] # keep track of used capacity
        d["associated"] = True
        used_devices += 1
        #if used_devices >= T: # Stop when enough devices have been recruited
        #  done = True
    devices = [d for d in devices if not d["associated"]]

    #if done:
    #    break
  end = time.time()

  if used_devices < T:
    return None
  else: 
    # get obj function value & filter out unused edge hosts
    objective = 0
    used_edge_hosts = []
    for e in edge_hosts:
      if len(e["associated_devices"]) > 0:
        objective += C_e[e["id"]]
        for d in e["associated_devices"]:
          objective += l*C_d[d][e["id"]]
        used_edge_hosts.append(e)
    
    return {"edge_hosts": used_edge_hosts, "objective": objective, "execution_time": end - start}

#######################
#######################

def greedy_cheapest_first(configuration, prune_devices = False):
  """ Solves the HFLOP instance specified by configuration using a heuristic that first considers device connectivity costs.

  Tries to find a feasibly solution as follows:
  - For each device i:
    * Sort edge hosts by cost (asc) 
    * Associate device with the first host in sorted list where device can fit
  - Check if # associated devices > T, otherwise return null (infeasible)
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
  T = configuration["participation_ratio"]*N # minimum number of participating devices

  edge_hosts = []
  devices = []
  for j in range(0, M):
    edge_hosts.append({"id": j, "cost": C_e[j], "available_capacity": R[j], "associated_devices": [], "used_capacity": 0, "score": 0})
  
  for i in range(0, N):
    devices.append({"id": i, "demand": D[i], "associated": None})
        

  # keep track of how many devs have been assigned
  used_devices = 0

  for d in devices:
    d["associated"] = False

    # Sort edge nodes from cheapest to most expensive
    for e in edge_hosts:
      e["score"] = C_d[d["id"]][e["id"]]
    edge_hosts = sorted(edge_hosts, key=itemgetter('score'))

    for e in edge_hosts:
      if d["demand"] <= e["available_capacity"] - e["used_capacity"]: # if device can fit
        e["associated_devices"].append(d["id"]) # associate device with edge node 
        e["used_capacity"] += d["demand"] # keep track of used capacity
        d["associated"] = e["id"]
        d["association_cost"] = C_d[d["id"]][e["id"]]
        used_devices += 1
        break

  # prune devices that are not necessary:
  # Sort devices by cost to their associated edge host and start removing until threshold
  if prune_devices:
    associated_devices = list(filter(lambda d: d["associated"] is not None, devices))
    associated_devices = sorted(associated_devices, key=itemgetter('association_cost'), reverse=True)  
    for d in associated_devices:
      if used_devices <= int(T):
        break
      for e in edge_hosts:
        if e["id"] == d["associated"]:
          e["associated_devices"].remove(d["id"])
          used_devices -= 1

  # get obj function value & filter out unused edge hosts
  objective = 0
  used_edge_hosts = []
  for e in edge_hosts:
    if len(e["associated_devices"]) > 0:
      objective += C_e[e["id"]]
      for d in e["associated_devices"]:
        objective += l*C_d[d][e["id"]]
      used_edge_hosts.append(e)
  
  end = time.time()
  
  return {"edge_hosts": used_edge_hosts, "objective": objective, "execution_time": end - start}

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

  # solve the problem
  assignment1 = greedy_pack(configuration)
  assignment2 = greedy_cheapest_first(configuration)
  
  # output solution
#  print(json.dumps(assignment1, indent=2))
  print("Hosts used: ", len(assignment1["edge_hosts"]), "objective:", assignment1["objective"])
  print("-------------")
  for e in assignment1["edge_hosts"]:
    print("Host ", e["id"], ":", len(e["associated_devices"]))
  print("##########")
#  print(json.dumps(assignment2, indent=2))
  print("Hosts used: ", len(assignment2["edge_hosts"]), "objective:", assignment2["objective"])
  print("-------------")
  for e in assignment2["edge_hosts"]:
    print("Host ", e["id"], ":", len(e["associated_devices"]))
  
