import argparse
import json
import threading
from collections import OrderedDict

import flwr as fl
import torch

from flwr.server.client_proxy import ClientProxy
from bottle import Bottle, request, response
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
from typing import Dict, List, Optional, Tuple, Union
import queue
import wandb
from utils2 import Net, test, load_data, inference, write_inference_results
import logging
from flwr.common.logger import log
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
import functools
import time

cloud_aggregation_lock = threading.Lock()
cloud_aggregation_lock.acquire()

continue_edge_round_lock = threading.Lock()
continue_edge_round_lock.acquire()

# https://github.com/adap/flower/blob/main/examples/simulation_pytorch/main.py
parser = argparse.ArgumentParser(description="Flower Simulation with PyTorch")
# For wandb
parser.add_argument("--number_clients", type=int, default=64)  # clients in the training
parser.add_argument("--project", type=str, default="")
# For the flower client
parser.add_argument("--id", type=str, default=0)
parser.add_argument("--rounds", type=int, default=0)
parser.add_argument("--address", type=int, default=0)
parser.add_argument("--local_rounds", type=int, default=0)
args = parser.parse_args()

current_edge_round = 0
next_cloud_round = args.local_rounds
model = Net(num_sensors=1, num_hidden_units=128, num_layers=2, t=12, dropout=0)
app = Bottle()
result_queue = queue.Queue()

lock = threading.Lock()


class CustomFedAvgGlobalServer(fl.server.strategy.FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        while result_queue.qsize() != 0:
            time.sleep(2)  # Wait for 2 seconds before checking the queue again
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        parameters_aggregated = ndarrays_to_parameters(aggregate(weights_results))
        metrics_aggregated = {}
        return parameters_aggregated, metrics_aggregated


class CustomFedAvg(fl.server.strategy.FedAvg):
    def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        # Wait until the queue is empty
        while result_queue.qsize() != 0:
            time.sleep(2)  # Wait for 2 second before checking the queue again
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Convert results
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        parameters_aggregated = ndarrays_to_parameters(aggregate(weights_results))

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(logging.WARNING, "No fit_metrics_aggregation_fn provided")

        # custom added code to support Hierarchical FL
        global edge_num_examples
        global current_edge_round
        global next_cloud_round
        params_dict = zip(model.state_dict().keys(), parameters_to_ndarrays(parameters_aggregated))
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

        edge_num_examples = 0
        for _, fit_res in results:
            edge_num_examples = edge_num_examples + fit_res.num_examples

        current_edge_round = current_edge_round + 1

        print("ServerThread - Finished edge aggregation: " + str(current_edge_round))

        if current_edge_round == next_cloud_round:
            next_cloud_round += args.local_rounds

            print("ServerThread - Releasing cloud aggregation lock...")
            cloud_aggregation_lock.release()

            print("ServerThread - Waiting for continue edge round lock...")
            continue_edge_round_lock.acquire()

            print("ServerThread - Continuing edge round...")

        return parameters_aggregated, metrics_aggregated


class EdgeServerClient(fl.client.NumPyClient):
    def __init__(self):
        self.device = "cuda:1"
        self.id = args.id
        self.total_epochs = 0

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        train_loader, test_loader = load_data(self.id, 16, 12, self.total_epochs)
        print("CLIENT server, training")
        self.set_parameters(parameters)
        # train(model, train_loader, self.epochs, self.device)
        global current_edge_round

        print("ClientThread - Waiting for cloud aggregation lock...")
        cloud_aggregation_lock.acquire()

        print("ClientThread - Sending parameters to cloud aggregation: " + str(current_edge_round))
        return self.get_parameters(config={}), len(train_loader.dataset), {}

    def evaluate(self, parameters, config):
        train_loader, test_loader = load_data(self.id, 16, 12, self.total_epochs)
        self.set_parameters(parameters)
        loss = test(model, test_loader, self.device)
        wandb.log({"loss": loss, "epochs": self.total_epochs})
        self.total_epochs += 1
        print("ClientThread - Releasing continue edge round lock...")
        continue_edge_round_lock.release()
        return float(loss), len(test_loader.dataset), {"loss": float(loss)}


class clientThread(threading.Thread):
    def __init__(self, threadID, name, counter):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.counter = counter

    def run(self):
        fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=EdgeServerClient())


class serverThread(threading.Thread):
    def __init__(self, threadID, name, counter):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.counter = counter

    def run(self):
        customStrategy = CustomFedAvg(
            fraction_fit=1,
            fraction_evaluate=1,
            min_available_clients=number_clients,
            min_fit_clients=number_clients,
            min_evaluate_clients=number_clients,
            accept_failures=True
        )

        # Start Flower server
        fl.server.start_server(
            server_address=f"127.0.0.1:{args.address}",  # different port
            config=fl.server.ServerConfig(num_rounds=args.rounds * args.local_rounds),
            strategy=customStrategy
        )


@app.post('/inference')
def inference_process(ID):
    "GET INFERENCE REQUEST"
    try:
        req_obj = request.body.read().decode('utf-8')
        result_queue.put(req_obj)
        response.status = 200
        return 'All done'
    except json.JSONDecodeError:
        response.status = 400
        return 'Invalid JSON'


# Function to consume results from the queue
def consume_results(ID):
    while True:
        try:
            req_data = result_queue.get()
            req_obj = json.loads(req_data)
            X_list = req_obj["X"]
            X = torch.tensor(X_list)
            y_list = req_obj["y"]
            y = torch.tensor(y_list)
            number_of_requests = req_obj["number_of_requests"]
            total_time = inference(X, y, model, "cpu", number_of_requests)
            # Write the result
            write_inference_results(ID, total_time)
            # Mark the task as done
            result_queue.task_done()
        except Exception as e:
            print("Error consuming result:", e)


def listen_to_route(port, ID):
    print(port)
    my_process_with_id = functools.partial(inference_process, ID)
    app.route('/inference', 'POST', my_process_with_id)
    app.run(host='127.0.0.1', port=port)


if __name__ == "__main__":
    number_clients = args.number_clients
    id = args.id
    is_global_server = id.startswith("g")
    server_position = 'local'
    if is_global_server:
        server_position = 'global'
    print(f"started {server_position} server-{id}")
    wandb.init(
        project=args.project,
        name=f"{server_position} server-{id}",
        mode="online",
    )
    if not is_global_server:
        thread1 = clientThread(1, "Client-Thread", 1)
        thread2 = serverThread(2, "Server-Thread", 2)
        route_thread = threading.Thread(target=listen_to_route, args=(str(1) + str(args.address)[1:], id,))
        route_thread.start()
        result_consumer_thread = threading.Thread(target=consume_results, args=(id,))
        result_consumer_thread.daemon = True
        result_consumer_thread.start()

        thread1.start()
        thread2.start()
        thread1.join()
        thread2.join()
        result_consumer_thread.join()
        route_thread.join()
    else:
        route_thread = threading.Thread(target=listen_to_route, args=(str(1) + str(args.address)[1:], id,))
        route_thread.start()
        result_consumer_thread = threading.Thread(target=consume_results, args=(id,))
        result_consumer_thread.daemon = True
        result_consumer_thread.start()
        global_strategy = CustomFedAvgGlobalServer(
            fraction_fit=1,
            fraction_evaluate=1,
            min_available_clients=number_clients,
            min_fit_clients=number_clients,
            min_evaluate_clients=number_clients,
            accept_failures=True
        )
        fl.server.start_server(
            server_address=f"127.0.0.1:{args.address}",
            config=fl.server.ServerConfig(num_rounds=args.rounds),
            strategy=global_strategy

        )
        result_consumer_thread.join()
        route_thread.join()
    print("connect")
