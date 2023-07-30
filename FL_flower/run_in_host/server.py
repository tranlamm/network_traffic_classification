from typing import List, Tuple

import pickle
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from flwr.common import FitIns

import flwr as fl
from flwr.common import Metrics

# Config
NUM_ROUNDS = 50
NUM_CLIENTS = 10
history_loss = []
history_acc = []

# Strategy
class CustomStrategy(fl.server.strategy.FedAvg):
    train_time = 30

    def aggregate_fit(
        self,
        server_round: int,
        results,
        failures,
    ):
        parameters_aggregated, metrics_aggregated = super().aggregate_fit(server_round, results, failures)
        return parameters_aggregated, metrics_aggregated

    def aggregate_evaluate(
        self,
        server_round: int,
        results,
        failures,
    ):
        loss_aggregated, metrics_aggregated = super().aggregate_evaluate(server_round, results, failures)
        history_loss.append(loss_aggregated)
        return loss_aggregated, metrics_aggregated

    def configure_fit(self, server_round: int, parameters, client_manager):
        config = {"train_time": self.train_time}
        fit_ins = FitIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )
        # Return client/config pairs
        return [(client, fit_ins) for client in clients]

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    acc = sum(accuracies) / sum(examples)
    # Aggregate and return custom metric (weighted average)
    return {"accuracy": acc}

# Create FedAvg strategy
strategy = CustomStrategy(
    fraction_fit=1,  # Sample 80% of available clients for training
    fraction_evaluate=1,  # Sample 80% of available clients for evaluation
    min_fit_clients=2,  # Never sample less than 8 clients for training
    min_evaluate_clients=2,  # Never sample less than 8 clients for evaluation
    min_available_clients=2,  # Wait until all 8 clients are available
    evaluate_metrics_aggregation_fn=weighted_average,
)

# Start Flower server
fl.server.start_server(
  server_address="0.0.0.0:9090",
  config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
  strategy=strategy,
)

# Declare storage file
this_dir = Path.cwd()
output_dir = this_dir / "flower_saved" / "30s"
if not output_dir.exists():
    output_dir.mkdir(parents=True)
    
loss = np.array(history_loss)
acc = np.array(history_acc)

# Lưu lại history vào file
history_loss_file = str(output_dir) + '/loss.pkl'
history_acc_file = str(output_dir) + '/acc.pkl'
with open(history_loss_file, 'wb') as f:
    pickle.dump(loss, f)
with open(history_acc_file, 'wb') as f:
    pickle.dump(acc, f)