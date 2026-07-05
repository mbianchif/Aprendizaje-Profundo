import orchestra
import time
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Any

from orchestra import PyTrainingConfig, TrainedModel

from utils.model import build_lenet5_model_config

SESSIONS_DIR = Path("sessions")


@dataclass
class SessionResult:
    secs_taken: float
    safetensors_path: str
    loss_history: list[float]


def __timed(f: Callable[[], Any]) -> tuple[Any, float]:
    """
    Executes a closure and times how much time it took to finish.
    """
    start = time.time_ns()
    res = f()
    end = time.time_ns()
    return res, round((end - start) / 1e9, 2)


def __save_training_results(trained_model: TrainedModel, secs_taken: float, name: str, file_name: str) -> None:
    """
    Saves the results of a training session to disk.
    """
    NAME_DIR = SESSIONS_DIR / name
    NAME_DIR.mkdir(parents=True, exist_ok=True)

    safetensors_path = f"{NAME_DIR}/{file_name}.safetensors"
    session_result_path = f"{NAME_DIR}/{file_name}.json"

    session_result = SessionResult(
        secs_taken=secs_taken,
        safetensors_path=safetensors_path,
        loss_history=trained_model.loss_history(),
    )

    trained_model.save_safetensors(safetensors_path)

    with open(session_result_path, "w") as f:
        json.dump(asdict(session_result), f, indent=4, sort_keys=True)


def exec_training(name: str, training_config: PyTrainingConfig, iteration: int) -> None:
    """
    Executes training sessions saving the results to disk.
    """
    print(f"Executing training session name={name} - iteration={iteration}")
    model = build_lenet5_model_config()
    session = orchestra.orchestrate(model, training_config)
    trained_model, secs_taken = __timed(lambda: session.wait())
    __save_training_results(trained_model, secs_taken, name, str(iteration))


def should_train() -> bool:
    """
    Looks for past training data, if there is none returns `True`, or `False` otherwise.
    """
    return not SESSIONS_DIR.exists()
