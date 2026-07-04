from pathlib import Path
from tarfile import TarInfo
import orchestra
import time
import json
from dataclasses import dataclass
from typing import Callable, Any

from orchestra import PyTrainingConfig, TrainedModel

from utils.model import build_lenet5_model_config

SESSIONS_DIR = "sessions"


@dataclass
class SessionResult:
    trained_model: TrainedModel
    ms_taken: float


def __timed(f: Callable[[], Any]) -> tuple[Any, float]:
    """
    Executes a closure and times how much time it took to finish.
    """
    start = time.time_ns()
    res = f()
    end = time.time_ns()
    return res, (end - start) / 1e-6


def __save_training_results(train_session_result: SessionResult, name: str, file_name: str):
    """
    Saves the results of a training session to disk.
    """
    NAME_DIR = f"{SESSIONS_DIR}/{name}"
    train_session_result_path = f"{NAME_DIR}/{file_name}.json"
    Path(train_session_result_path).parent.mkdir(parents=True, exist_ok=True)

    with open(train_session_result_path, "w") as f:
        json.dump(train_session_result, f)


def exec_training(name: str, training: PyTrainingConfig, times: int):
    """
    Executes training sessions saving the results to disk.
    """
    model = build_lenet5_model_config()

    for i in range(times):
        session = orchestra.orchestrate(model, training)
        trained_model, ms_taken = __timed(lambda: session.wait())
        train_session_result = SessionResult(trained_model, ms_taken)
        __save_training_results(train_session_result, name, str(i))


def should_train(confirm: bool) -> bool:
    """
    Looks for past training data, if there is none, returns `True` or `False` otherwise.
    """
    if not confirm:
        print("Confirmation flag is set to False")

    return confirm and not Path(SESSIONS_DIR).exists()
