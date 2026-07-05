import random
from dataclasses import dataclass
from itertools import product
from typing import Self, Callable

from orchestra import PyTrainingConfig

from utils.docker import Docker
from utils.session import exec_training, should_train
from utils.training import (
    build_mnist_all_reduce_training_config,
    build_mnist_parameter_server_training_config,
)


@dataclass(frozen=True)
class SessionConfig:
    name: str
    nodes: int
    config_builder: Callable[[list[str], int], PyTrainingConfig]

    @classmethod
    def all_reduce(cls, nodes: int, offline_epochs: int) -> Self:
        """
        Builds a configuration for a training session using All Reduce as distributed training algorithm.
        """
        if nodes < 1:
            raise ValueError("The node count must be positive")

        return cls(
            name=f"all_reduce-{nodes}-{offline_epochs}",
            nodes=nodes,
            config_builder=lambda addrs, seed: build_mnist_all_reduce_training_config(
                addrs=addrs,
                offline_epochs=offline_epochs,
                seed=seed,
            ),
        )

    @classmethod
    def parameter_server(cls, nodes: int, offline_epochs: int) -> Self:
        """
        Builds a configuration for a training session using Parameter Server as distributed training algorithm.
        """
        if nodes < 2:
            raise ValueError("The node count must be greater than 1")

        nservers = nodes // 2

        return cls(
            name=f"parameter_server-{nodes}-{offline_epochs}",
            nodes=nodes,
            config_builder=lambda addrs, seed: build_mnist_parameter_server_training_config(
                addrs=addrs,
                nservers=nservers,
                offline_epochs=offline_epochs,
                seed=seed,
            ),
        )


INITIAL_SEED = 67

# Key system parameters to measure.
NODES = [6, 8, 10]
OFFLINE_EPOCHS = [0, 1, 2]
REPEATS = 1

SESSIONS = [
    factory(nodes, offline_epochs)
    for factory in (SessionConfig.all_reduce, SessionConfig.parameter_server)
    for nodes, offline_epochs in product(NODES, OFFLINE_EPOCHS)
]


def main() -> None:
    if not should_train():
        print("There are past measurements, delete the 'sessions' directory and retry")
        return

    rng = random.Random(INITIAL_SEED)
    docker = Docker(ono_project_root="~/fiuba/tpp")

    for session in SESSIONS:
        addrs = docker.compose_up(nodes=session.nodes, release=True)

        def training_config_factory(seed: int) -> PyTrainingConfig:
            return session.config_builder(addrs, seed)

        exec_training(session.name, training_config_factory, REPEATS, rng)

    docker.compose_down()


if __name__ == "__main__":
    main()
