import argparse
from collections import OrderedDict
import random

import flwr as fl
import torch

import wandb
from utils2 import train, test, load_data, Net, inference, send_inference_requests_to_server, write_inference_results, \
    write_latency_results
import asyncio
import time

# https://github.com/adap/flower/blob/main/examples/simulation_pytorch/main.py
parser = argparse.ArgumentParser(description="Flower Simulation with PyTorch")
# For wandb
parser.add_argument("--pr", type=str, default="")
# For the flower client
parser.add_argument("--epochs", type=int, default=0)
parser.add_argument("--monitor", type=bool, default=False)
parser.add_argument("--id", type=str, default=0)
parser.add_argument("--server_address", type=str, default=8080)
parser.add_argument("--failure", type=int, default=0, required=False)
parser.add_argument("--i_req_r", type=int, default=0, required=False)
parser.add_argument("--i_p_c", type=int, default=0, required=False)
args = parser.parse_args()


class CifarClient(fl.client.NumPyClient):

    def __init__(self, epochs, monitor, failure, inference_request_rate, inference_processing_capacity, id,
                 server_address):
        device_nr = int(id[1:]) % 2
        self.device = torch.device(f"cuda:{device_nr}")
        self.model = Net(num_sensors=1, num_hidden_units=128, num_layers=2, t=12, dropout=0).to(self.device)
        self.epochs = epochs
        self.monitor = monitor
        self.failure = failure
        self.id = id
        self.latency = random.uniform(0.0008, 0.01)#0.05, 0.1)#0.0008, 0.01)  # latency
        if str(id[1:]) == "767621" or str(id[1:]) == "772596": #only for hier benchmark
            self.latency = self.latency + random.uniform(0.05, 0.1)  # latency
        write_latency_results(self.id, self.latency)
        print(self.latency)
        self.count = 0
        self.total_epochs = 0
        self.inference_request_rate = inference_request_rate
        self.server_address = str(1) + str(server_address)[1:]
        self.inference_processing_capacity = inference_processing_capacity

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters, net):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        logging_value = self.total_epochs  # con issue
        self.set_parameters(parameters, self.model)
        train_loader, test_loader = load_data(self.id, 16, 12, logging_value)
        loss = test(self.model, test_loader, self.device)
        (X, y) = list(test_loader)[-1]  # only use last batch for inference
        if (self.inference_request_rate >= self.inference_processing_capacity):
            asyncio.run(
                send_inference_requests_to_server(X, y, self.inference_request_rate, self.server_address, self.latency))
        else:
            lateny, calc_time = inference(X, y, self.model, "cpu", self.inference_request_rate, 0, 1)
            write_inference_results(lateny, calc_time)
        wandb.log({"global loss": loss, "epochs": logging_value})
        time.sleep(1)
        train(self.model, train_loader, self.epochs, self.device)
        loss = test(self.model, test_loader, self.device)
        time.sleep(1)
        wandb.log({"local loss": loss})
        self.total_epochs = self.total_epochs + 1
        return self.get_parameters(config={}), len(train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters, self.model)
        logging_value = self.total_epochs  # con issue
        train_loader, test_loader = load_data(self.id, 16, 12, logging_value)
        loss = test(self.model, test_loader, self.device)
        return float(loss), len(test_loader.dataset), {"loss": float(loss)}


if __name__ == "__main__":
    print("started client " + str(args.id))
    # https://docs.wandb.ai/guides/track/launch
    wandb.init(
        project=args.pr,
        name=f"client-{args.id}",
    )
    fl.client.start_numpy_client(server_address=f"127.0.0.1:{args.server_address}",
                                 client=CifarClient(args.epochs, args.monitor, args.failure,
                                                    args.i_req_r, args.i_p_c,
                                                    args.id, args.server_address))
