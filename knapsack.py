import time
import sys
import getopt
import json
from operator import itemgetter, attrgetter
from knapsack import *

def knapsack_dp(configuration, sack, items):
  """ Dynamic programming algorithm (exact). [TODO] 
  """
  pass

def knapsack_fptas(configuration, sack, items):
  """ FPTAS for the knapsack problem. [TODO]
  """
  pass
  
def knapsack_danzig(configuration, sack, items):
  """ Run Danzig's greedy approx. algorithm (sack: edge host id, items: list of device ids)
  
  The objective is to maximize a utility value that is inversely proportional to the device cost to 
  the specific host. The weight of a device is its workload, and the capacity is the edge hosts
  request processing capacity. We solve the problem as follows:
  - sort devices by decreasing value/weight ratios
  - place devices in the knapsack in that order until no more devices can fit.
  - compare the obj value to that of a set that includes only the k+1 item, and keep the max
  """
  R = configuration["capacities"] # vector with M elements
  C_d = configuration["device_costs"] # NxM matrix
  C_e = configuration["edge_costs"] # vector with M elements
  D = configuration["workload"] # vector with N elements
  l = configuration["local_to_global_ratio"] # how many local aggregation rounds per global aggregation round
  
  # sort devices by value/weight, where value = 1/(cost + 1), weight = workload
  devices = []
  for d in items:
    devices.append({
      "id": d, 
      "score": 1.0/(C_d[d][sack] + 1)/(D[d]+1),
      "demand": D[d],
      "associated": None
    })
  devices = sorted(devices, key=itemgetter("score"), reverse=True)
  
  load = 0 # current load
  value = 0 # value of obj func (we want to max it)
  W = R[sack] # knapsack capacity
  k = 0
  edge_host = {"id": sack, "cost": C_e[sack], "available_capacity": R[sack], "associated_devices": [], "used_capacity": 0}
  # pack devices into knapsack
  while True:
    if k < len(devices) and devices[k]["demand"] + load <= W:
      edge_host["associated_devices"].append(devices[k]["id"])
      edge_host["used_capacity"] += devices[k]["demand"]
      devices[k]["associated"] = sack
      devices[k]["association_cost"] = C_d[devices[k]["id"]][sack]
      load += devices[k]["demand"]
      value += 1.0/(C_d[devices[k]["id"]][sack] + 1)
      k += 1
    else:
      break
    
  if k < len(devices):
    # check devices[k+1], maybe it's better
    if 1.0/(C_d[devices[k]["id"]][sack] + 1) > value:
      edge_host["associated_devices"] = [devices[k]["id"]]
    
  return edge_host


