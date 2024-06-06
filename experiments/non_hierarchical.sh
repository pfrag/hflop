#!/bin/bash
#run with bash run.sh

#name for wandb
project_name="Non Hierarchical Benchmark"
#global server
server_rounds=50
#local aggregations
local_rounds=2
#number of client epochs
client_epochs=5
#not important for current experiments
number_failing_clients=0
failure_rounds=0

# Clients in each cluster
clients_in_cluster_A=5
clients_in_cluster_B=5
clients_in_cluster_C=5
clients_in_cluster_D=5
#inference requests for each client
inference_request_rate=(100 100 100 100 100 100 100 100 100 100 100 100 100 100 100 100 100 100 100 100)
#inference request capacity for each client
inference_processing_capacity=(100 100 100 100 100 100 100 100 100 100 100 100 100 100 100 100 100 100 100 100)

 Start global server first
python3  local_server.py --number_clients "0" --project "$project_name" --id "g0" --rounds "$server_rounds" --address "8080" --local_rounds "$local_rounds"&

#wait for server to start
sleep 5

# Start intermediate server for Cluster A
python3 local_server.py --number_clients "$clients_in_cluster_A" --project "$project_name" --id "l1" --rounds "$server_rounds" --address "8081" --local_rounds "$local_rounds"&

# Start intermediate server for Cluster B
python3 local_server.py --number_clients "$clients_in_cluster_B" --project "$project_name" --id "l2" --rounds "$server_rounds" --address "8082" --local_rounds "$local_rounds"&

# Start intermediate server for Cluster C
python3 local_server.py --number_clients "$clients_in_cluster_C" --project "$project_name" --id "l2" --rounds "$server_rounds" --address "8083" --local_rounds "$local_rounds"&

# Start intermediate server for Cluster D
python3 local_server.py --number_clients "$clients_in_cluster_D" --project "$project_name" --id "l2" --rounds "$server_rounds" --address "8084" --local_rounds "$local_rounds"&

# Wait for intermediate servers to start
sleep 5

# Start clients in Cluster A
for i in $(seq 0 $(($clients_in_cluster_A - 1)))
do
  python3 client.py --project "$project_name" --inference_request_rate "${inference_request_rate[$i]}" --inference_processing_capacity "${inference_processing_capacity[$i]}" --epochs "$client_epochs" --id "A$i" --server_address "8081" &
  sleep 1
done

# Start clients in Cluster B
total_clients=$((clients_in_cluster_A + clients_in_cluster_B-1))
for i in $(seq $clients_in_cluster_A $total_clients)
do
  python3 client.py --project "$project_name" --inference_request_rate "${inference_request_rate[$i]}" --inference_processing_capacity "${inference_processing_capacity[$i]}" --epochs "$client_epochs" --id "B$i" --server_address "8082" &
  sleep 1
done

# Start clients in Cluster B

for i in $(seq $(($total_clients+1))  $(($clients_in_cluster_C + $total_clients)))
do
  python3 client.py --project "$project_name" --inference_request_rate "${inference_request_rate[$i]}" --inference_processing_capacity "${inference_processing_capacity[$i]}" --epochs "$client_epochs" --id "C$i" --server_address "8083" &
  sleep 1
done

total_clients=$((total_clients + clients_in_cluster_C))

# Start clients in Cluster D
for i in $(seq $((total_clients +1))  $(($clients_in_cluster_D + $total_clients)))
do
  python3 client.py --project "$project_name" --inference_request_rate "${inference_request_rate[$i]}" --inference_processing_capacity "${inference_processing_capacity[$i]}" --epochs "$client_epochs" --id "D$i" --server_address "8084" &
  sleep 1
done

wait
