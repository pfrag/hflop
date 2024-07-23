from generate import *
from hflop import *
from greedy import *
from knapsack import *
from localsearch import *

################################################################
# generate a configuration
################################################################

N = 500 # clients
M = 50 # edge nodes
seed = int(time.time())
T = 0.7 # FL participation ratio
l = 4 # ratio of local to global rounds
maxcap = 4500 # maximum edge host capacity
maxdemand = 50 # maximum device demand
zero_cost_lan = False # if true, each client has a single edge loc. for which it has zero cost and cost is 1 unit towards the rest. Otherwise, cost is drawn randomly
min_conn_cost = 0 # minimum communication cost 
max_conn_cost = 10 # maximum communication cost

verbose = False # print more info about solutions

# generate problem instance
configuration = generate(N, M, seed, participation_ratio = T, local_to_global_ratio = l, max_edge_node_capacity = maxcap, max_device_demand = maxdemand, zero_cost_lan = zero_cost_lan, min_conn_cost = min_conn_cost, max_conn_cost = max_conn_cost)

################################################################
# solve the problem in three ways
################################################################

opt = hflop(configuration) # optimal
if not opt:
  print("No feasible solution")
  sys.exit(1)
ls_ass = local_search(configuration) # local search heuristic
greedy_ass =  greedy_cheapest_first(configuration, prune_devices = True) # greedy heuristic

################################################################
# output solutions
################################################################

C_d = configuration["device_costs"]
D = configuration["workload"]
R = configuration["capacities"]
N = len(configuration["workload"])

print("[OPT] Hosts used: ", len(opt["edge_hosts"]), "objective:", opt["objective"], "execution time:", opt["execution_time"])
if verbose:
  print("-------------")
  devs = 0 
  for e in opt["edge_hosts"]:
    devs += len(e["associated_devices"])
    load = sum( D[x] for x in e["associated_devices"])
    print("Host ", e["id"], ":", len(e["associated_devices"]), "load/capacity:", str(load) + "/" + str(R[e["id"]]))
  print("Devices recruited:", str(devs))
  print("##########")

print("[LS] Hosts used: ", len(ls_ass["edge_hosts"]), "objective:", ls_ass["objective"], "execution time:", ls_ass["execution_time"], "merge steps:", ls_ass["num_merge_steps"])
if verbose:
  print("-------------")
  devs = 0
  for e in ls_ass["edge_hosts"]:
    load = sum( D[x] for x in e["associated_devices"])
    devs += len(e["associated_devices"])
    print("Host ", e["id"], ":", len(e["associated_devices"]), "load/capacity:", str(load) + "/" + str(R[e["id"]]))
  print("Devices recruited:", str(devs))
  print("##########")

print("[GREEDY] Hosts used: ", len(greedy_ass["edge_hosts"]), "objective:", greedy_ass["objective"], "execution time:", greedy_ass["execution_time"])

if verbose:
  print("-------------")
  devs = 0 
  for e in greedy_ass["edge_hosts"]:
    load = sum( D[x] for x in e["associated_devices"])
    devs += len(e["associated_devices"])
    print("Host ", e["id"], ":", len(e["associated_devices"]), "load/capacity:", str(load) + "/" + str(R[e["id"]]))
  print("Devices recruited:", devs)

