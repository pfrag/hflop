#!/bin/bash
#run with bash run.sh

#change the latency in the client.py + name of dir

#name for wandb
proj="HFLOP l0.0008 l0.01 speed1"
#global server
s_r=100
#local aggregations
l_r=2
#number of client epochs
c_ep=5
#not important for current experiments
number_failing_clients=0
failure_rounds=0

# Clients in each cluster
clients_in_cluster_A=4
clients_in_cluster_B=6
clients_in_cluster_C=4
clients_in_cluster_D=6

#inference requests for each client
ids=(771667 767495 767523 718064 767621 717571 760987 717492 769359 717491 769443 765099 717508 767350 772596 716968 717481 717461 716337 761604)

i_req_r=(2 13 3 20 23 7 26 23 14 22 15 8 6 0 12 16 2 20 8 3)
#inference request capacity for each client
i_p_c=(2 13 3 20 23 7 26 23 14 22 15 8 6 0 12 16 2 20 8 3)

#Start global server first
python3  local_server.py --n_c "4" --pr "$proj" --id "g0" --rounds "$s_r" --address "8080" --l_r "$l_r" --server_capacity "1"&
sleep 5

#Start local servers
python3  local_server.py --n_c "$clients_in_cluster_A" --pr "$proj" --id "l1" --rounds "$s_r" --address "8081" --l_r "$l_r" --server_capacity "1"&
python3  local_server.py --n_c "$clients_in_cluster_B" --pr "$proj" --id "l2" --rounds "$s_r" --address "8082" --l_r "$l_r" --server_capacity "1"&
python3  local_server.py --n_c "$clients_in_cluster_C" --pr "$proj" --id "l3" --rounds "$s_r" --address "8083" --l_r "$l_r" --server_capacity "1"&
python3  local_server.py --n_c "$clients_in_cluster_D" --pr "$proj" --id "l4" --rounds "$s_r" --address "8084" --l_r "$l_r" --server_capacity "1"&

##wait for server to start
sleep 5

# Start clients in Cluster A
for i in $(seq 0 $(($clients_in_cluster_A - 1)))
do
  python3 client.py --pr "$proj" --i_req_r "${i_req_r[$i]}" --i_p_c "${i_p_c[$i]}" --epochs "$c_ep" --id "A${ids[i]}" --server_address "8081" &
  sleep 1
done

# Start clients in Cluster B
total_clients=$((clients_in_cluster_A + clients_in_cluster_B-1))
for i in $(seq $clients_in_cluster_A $total_clients)
do
  python3 client.py --pr "$proj" --i_req_r "${i_req_r[$i]}" --i_p_c "${i_p_c[$i]}" --epochs "$c_ep" --id "B${ids[i]}" --server_address "8082" &
  sleep 1
done

# Start clients in Cluster B

for i in $(seq $(($total_clients+1))  $(($clients_in_cluster_C + $total_clients)))
do
  python3 client.py --pr "$proj" --i_req_r "${i_req_r[$i]}" --i_p_c "${i_p_c[$i]}" --epochs "$c_ep" --id "C${ids[i]}" --server_address "8083" &
  sleep 1
done

total_clients=$((total_clients + clients_in_cluster_C))

# Start clients in Cluster D
for i in $(seq $((total_clients +1))  $(($clients_in_cluster_D + $total_clients)))
do
  python3 client.py --pr "$proj" --i_req_r "${i_req_r[$i]}" --i_p_c "${i_p_c[$i]}" --epochs "$c_ep" --id "D${ids[i]}" --server_address "8084" &
  sleep 1
done

wait
