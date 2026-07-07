import json
import statistics
from dataclasses import dataclass
from functools import cache

import torch
import numpy as np
from torch import Tensor

from utils.model import load_lenet5_from_safetensors
from utils.session import SessionResult, SESSIONS_DIR


@dataclass
class Measurement:
    acc_avg: float
    acc_std_dev: float
    secs_taken_avg: float
    secs_taken_std_dev: float
    loss_history_avgs: list[float]
    loss_history_std_devs: list[float]


def __retrieve_training_results(name: str) -> list[SessionResult]:
    """
    Retrieves the training results for a given training session name.
    """
    NAME_DIR = SESSIONS_DIR / name
    results = []

    for result_path in sorted(NAME_DIR.glob("*.json")):
        with open(result_path, "r") as f:
            data = json.load(f)
            results.append(SessionResult(**data))

    return results


@cache
def __build_datasets() -> tuple[Tensor, Tensor]:
    """
    Builds the testing datasets for measuring the trained model's accuracy.
    """
    mnist_dataset_dir = "datasets/mnist"
    x_test_path = f"{mnist_dataset_dir}/mnist_test_samples.bin"
    x_test = np.fromfile(x_test_path, dtype=np.float32).reshape(-1, 1, 28, 28)
    xt = torch.tensor(x_test)

    y_test_path = f"{mnist_dataset_dir}/mnist_test_labels.bin"
    y_test = np.fromfile(y_test_path, dtype=np.float32).reshape(-1, 10)
    yt = torch.tensor(y_test.argmax(axis=1), dtype=torch.long)

    return xt, yt


def __calculate_accuracies(results: list[SessionResult]) -> list[float]:
    """
    Calculates the model accuracies of all the measurements for the same model.
    """
    xt, yt = __build_datasets()
    accuracies = []

    for result in results:
        model = load_lenet5_from_safetensors(result.safetensors_path)

        with torch.no_grad():
            accuracy = (model(xt).argmax(dim=1) == yt).float().mean().item()
            accuracies.append(accuracy)

    return accuracies


def __calculate_times_taken(results: list[SessionResult]) -> list[float]:
    """
    Retrieves the seconds taken for every iteration of the same model.
    """
    return [result.secs_taken for result in results]


def __calculate_loss_histories(results: list[SessionResult]) -> list[list[float]]:
    """
    Retrieves and groups the loss histories of the various training sessions of the same model.
    """
    loss_histories = [result.loss_history for result in results]
    grouped_loss_histories = []
    i = 0

    while True:
        epoch_losses = []

        for lh in loss_histories:
            if i < len(lh):
                epoch_losses.append(lh[i])

        if not epoch_losses:
            break

        grouped_loss_histories.append(epoch_losses)
        i += 1

    return grouped_loss_histories


def __measure_training_results_for_name(name: str) -> Measurement:
    """
    For a given session name measures the relevant statistics.
    """
    results = __retrieve_training_results(name)

    if not results:
        raise ValueError("No training sessions were executed")

    accs = __calculate_accuracies(results)
    acc_avg = statistics.mean(accs)
    acc_std_dev = statistics.stdev(accs)

    secs_taken = __calculate_times_taken(results)
    secs_taken_avg = statistics.mean(secs_taken)
    secs_taken_std_dev = statistics.stdev(secs_taken)

    loss_histories = __calculate_loss_histories(results)
    loss_history_avgs = [statistics.mean(lh) for lh in loss_histories]
    loss_history_std_devs = [statistics.stdev(lh) for lh in loss_histories]

    return Measurement(
        acc_avg,
        acc_std_dev,
        secs_taken_avg,
        secs_taken_std_dev,
        loss_history_avgs,
        loss_history_std_devs,
    )


def measure_training_results() -> dict[str, Measurement]:
    """
    Calculates the statistics for all the training sessions in the sessions directory.
    """
    measurements = {}

    for entry in SESSIONS_DIR.iterdir():
        name = entry.name
        measurements[name] = __measure_training_results_for_name(name)

    return measurements


def available_results() -> bool:
    """
    Looks for past training data, if there is none returns `False`, or `True` otherwise.
    """
    return SESSIONS_DIR.exists()
