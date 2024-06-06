#!/bin/bash
#run with bash run.sh

#name for wandb
project_name="Non Hierarchical Benchmark"
#global server
server_rounds=50
#number of client epochs
client_epochs=10
#not important for current experiments
number_failing_clients=0
failure_rounds=0
local_rounds=0
number_clients=20
#inference requests for each client
inference_request_rate=(100 100 100 100 100 100 100 100 100 100 100 100 100 100 100 100 100 100 100 100)
inference_processing_capacity=(100 100 100 100 100 100 100 100 100 100 100 100 100 100 100 100 100 100 100 100)
# Start global server first
python3  local_server.py --number_clients "$number_clients" --project "$project_name" --id "g0" --rounds "$server_rounds" --address "8080" --local_rounds "$local_rounds"&

#wait for server to start
sleep 10

# Start clients in Cluster A
for i in $(seq 0 $(($number_clients - 1)))
do
  python3 client.py --project "$project_name" --inference_request_rate "${inference_request_rate[$i]}" --inference_processing_capacity "${inference_processing_capacity[$i]}" --epochs "$client_epochs" --id "A$i" --server_address "8080" &
  sleep 1
done

# Wait for all background processes to complete
wait
