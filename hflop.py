import cplex
import time
import sys
import getopt
import json
from operator import itemgetter, attrgetter

def hflop(configuration):
  """ Solves the HFLOP instance specified by configuration.
  
  This function uses the CPLEX python api to invoke the solver and returns the
  solution (assingment of devices to edge aggregators) and some information
  about it.
  """
  
  ##########################################
  # Configuration and variable setup
  ##########################################
  
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
  
  
  # x_{i,j}, y_j variable names
  varnames = [] # variable names 
  vartypes = "" # variable types (basically all are binary)
  pairs = [] # variable-constant pairs that will be summed up as terms of the objective function
  
  # generate intuitive variable names
  start = time.time()

  for j in range(1, M+1):
    varnames.append("y-" + str(j)) # edge node j is selected
    vartypes += "B" # binary variable
    
  for i in range(1, N+1): #for each device i
    for j in range(1, M+1): # for each node j, create a variable x_ij (device j associated with edge node j)
      varnames.append("x-" + str(i) + "-" + str(j))
      vartypes += "B" # binary variables
      
  # generate coefficient for each variable
  for j in range(1, M+1):
      pairs.append(["y-" + str(j), C_e[j-1]])

  for i in range(1, N+1):
    for j in range(1, M+1):
      pairs.append(["x-" + str(i) + "-" + str(j), C_d[i-1][j-1]*l])
  end = time.time()
  #print("Generating var names took: " + "%.5f" % (end - start) + "s")

  ###################################
  # CPLEX setup
  ###################################
  
  c = cplex.Cplex()
  c.set_results_stream(None)
  c.set_log_stream(None)
  c.set_problem_name("HFLOP")

  # minimization problem
  c.objective.set_sense(c.objective.sense.minimize)

  # create variables
  start = time.time()
  c.variables.add(names = varnames, types=vartypes)
  end = time.time()
  #print("Adding vars took: " + "%.5f" % (end - start) + "s")

  c.objective.set_linear(pairs)

  # map variable name to its index
  name2idx = { n : j for j, n in enumerate(c.variables.get_names()) }

  ##########################################
  # Create constraints
  ##########################################
  
  start = time.time()
  
  # constraints to remove the need for indicator function in the objective
  # 1. need to make sure a node w/o associated clients is not used
  # 2. need to make sure a node is used if it has associated clients
  # (a pair of constraints per edge host)

  # 1) sum(x_ij) - y_J >= 0
  for j in range(1, M+1): # 1 constraint for each edge host
    #senses = ""
    rhs = []
    lhs = [] 

    vars = []
    coefs = [1]*N

    for i in range(1, N+1): # for each device
      vars.append(name2idx["x-" + str(i) + "-" + str(j)])
    vars.append(name2idx["y-" + str(j)])
    coefs.append(-1)
    lhs.append((vars, coefs))
    rhs.append(0)
    senses = "G"
    c.linear_constraints.add(
                         lin_expr=lhs,
                         senses=senses,
                         rhs=rhs
                         )

  # 2) sum(x_ij) - M*y_j <= 0
  bigM = N + 1
  for j in range(1, M+1): # 1 constraint for each edge host
    rhs = []
    lhs = []

    vars = []
    coefs = [1]*N

    for i in range(1, N+1): # for each device
      vars.append(name2idx["x-" + str(i) + "-" + str(j)])
    vars.append(name2idx["y-" + str(j)])
    coefs.append(-bigM)
    lhs.append((vars, coefs))
    rhs.append(0)
    senses = "L"
    c.linear_constraints.add(
                         lin_expr=lhs,
                         senses=senses,
                         rhs=rhs
                         )
                         
  # Create capacity constraints
  senses = ""
  for j in range(1, M+1):
    vars = []
    coefs = []
    lhs = []
    rhs = []
    for i in range(1, N+1):
      vars.append(name2idx["x-" + str(i) + "-" + str(j)])
      coefs.append(D[i-1])
    lhs.append((vars, coefs))
    rhs.append(R[j-1])
    senses = "L"
    c.linear_constraints.add(
                   lin_expr=lhs,
                   senses=senses,
                   rhs=rhs
                   )

  # Verify that each device gets assigned to at most one edge node
  senses = ""
  rhs = []
  lhs = []
  for i in range(1, N+1): # for each device i 
    vars = []
    coef = [1]*M
    for j in range(1, M+1): # for each host j
      vars.append(name2idx["x-" + str(i) + "-" + str(j)])
    lhs.append(cplex.SparsePair(ind=vars, val=coef))
    rhs.append(1)
    senses += "L"
  c.linear_constraints.add(
                         lin_expr=lhs,
                         senses=senses,
                         rhs=rhs
                         )

  # verify that at least T clients get assigned to participate in FL
  senses = ""
  rhs = []
  lhs = []
  coef = [1]*M*N
  vars = []
  for i in range(1, N+1): # for each device i 
    for j in range(1, M+1): # for each host j
      vars.append(name2idx["x-" + str(i) + "-" + str(j)])
  lhs.append(cplex.SparsePair(ind=vars, val=coef))
  rhs.append(T)
  senses += "G"
  c.linear_constraints.add(
                         lin_expr=lhs,
                         senses=senses,
                         rhs=rhs
                         )
  
  end = time.time()
  #print("Adding constraints took: " + "%.5f" % (end - start) + "s")

  ################################
  # Invoke the solver
  ################################
  
  start = time.time()
  c.solve()
  end = time.time()
  sol = c.solution
  
  #print("Objective function value: " + str(sol.get_objective_value()))
  #print("Elapsed time: " + "%.5f" % (end - start) + "s")
  
  return export_solution(configuration, sol, end - start)

def export_solution(configuration, solution, solution_time=None):
  """ Exports a solution to an HFLOP instance with the given configuration as a python dictionary."""
  N = len(configuration["workload"])
  M = len(configuration["capacities"]) 
  R = configuration["capacities"]
  D = configuration["workload"]

  assignment = {}
  edge_hosts = []
    
  # for each edge node, show devices assigned
  for j in range(1, M+1):
    host = {"id": j-1}
    used_capacity = 0
    available_capacity = R[j-1]
    associated_devices = []
    vars = []
    for i in range(1, N+1):
      vars.append("x-" + str(i) + "-" + str(j))
    values = solution.get_values(vars)

    for k in range(len(values)):
      if int(values[k]) == 1:
        associated_devices.append(k)
        used_capacity += D[k-1]
    host["associated_devices"] = associated_devices
    host["available_capacity"] = available_capacity
    host["used_capacity"] = used_capacity
    edge_hosts.append(host)
  assignment["edge_hosts"] = edge_hosts
  assignment["objective"] = solution.get_objective_value()
  assignment["execution_time"] = solution_time
  
  return assignment
  
if __name__ == '__main__':    
  # open configuration file
  myopts, args = getopt.getopt(sys.argv[1:], "c:")
  configfile = None
  for o, a in myopts:
    if o == "-c":
      configfile = a

  if configfile == None:
    print("Missing configuration file. Usage: python vassign.py -c /path/to/configfile")
    exit(1)

  try:
    with open(configfile) as fp:
      configuration = json.load(fp)
  except:
    print("Error loading scenario file. Possibly malformed...")

  # solve the problem
  assignment = hflop(configuration)
  
  # output optimal solution
  print(json.dumps(assignment, indent=2))
  
