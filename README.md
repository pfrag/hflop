# hflop - Hierarchical Federated Learning Orchestration

## Description 
Solves a version of the Hierarchical Federated Learning Orchestration Problem (HFLOP).

The purpose is to select a minimum (communication) cost assignment of FL devices to aggregators deployed at edge hosts with specific inference request processing capacities. 

## Requirements
- ILOG CPLEX Optimization Studio (the academic edition is available here after registration: https://www.ibm.com/academic/topic/data-science)
- Python 3

## Setup
- Setup a python 3 virtual environment:
```
virtualenv -p python3 venv
. ./venv/bin/activate
```

- Install CPLEX. You will be prompted to run the following commnand to install the python API: `python /opt/ibm/ILOG/CPLEX_Studio2211/python/setup.py install`
Run this command inside the above virtual environment.

## Running
First generate a configuration. Example: `python generate.py -e 2 -d 10 -p 1 -l 4 -o cfg.json`
where:
-e: Number of edge nodes
-d: Number of FL clients (devices)
-p: Device participation ratio
-l: Number of local aggregation rouds per global aggregator round
-o: output file

Then, execute like this: `python hflop.py -c cfg.json`. The output looks like this:
```
{
  "edge_hosts": [
    {
      "id": 0,
      "associated_devices": [
        0,
        3,
        4
      ],
      "available_capacity": 67,
      "used_capacity": 15
    },
    {
      "id": 1,
      "associated_devices": [
        1,
        2,
        5,
        6,
        7,
        8,
        9
      ],
      "available_capacity": 54,
      "used_capacity": 43
    }
  ],
  "objective": 2.0,
  "execution_time": 0.008423328399658203
}
```
The field `objective` represents the value of the objective function corresponding to the calculated optimal solution.

