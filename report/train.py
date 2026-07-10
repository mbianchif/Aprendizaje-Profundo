#!/usr/bin/env -S uv run

from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from itertools import product
from random import Random
from typing import Self, Callable

from orchestra import PyTrainingConfig

from utils.docker import Docker
from utils.session import exec_training
from utils.training import (
    build_mnist_all_reduce_training_config,
    build_mnist_parameter_server_training_config,
)


@dataclass(frozen=True)
class SessionConfig:
    name: str
    nodes: int
    iteration: int
    config_builder: Callable[[list[str], int], PyTrainingConfig]

    @classmethod
    def all_reduce(cls, nodes: int, offline_epochs: int, iteration: int) -> Self:
        """
        Builds a configuration for a training session using All Reduce as distributed training algorithm.
        """
        if nodes < 1:
            raise ValueError("The node count must be positive")

        return cls(
            name=f"all_reduce-{nodes}-{offline_epochs}",
            nodes=nodes,
            iteration=iteration,
            config_builder=lambda addrs, seed: build_mnist_all_reduce_training_config(
                addrs=addrs,
                offline_epochs=offline_epochs,
                seed=seed,
            ),
        )

    @classmethod
    def parameter_server(cls, nodes: int, offline_epochs: int, iteration: int) -> Self:
        """
        Builds a configuration for a training session using Parameter Server as distributed training algorithm.
        """
        if nodes < 2:
            raise ValueError("The node count must be greater than 1")

        return cls(
            name=f"parameter_server-{nodes}-{offline_epochs}",
            nodes=nodes,
            iteration=iteration,
            config_builder=lambda addrs, seed: build_mnist_parameter_server_training_config(
                addrs=addrs,
                nservers=nodes // 2,
                offline_epochs=offline_epochs,
                seed=seed,
            ),
        )


def parse_args() -> Namespace:
    """
    Parses the relevant command line arguments.
    """
    parser = ArgumentParser()

    parser.add_argument(
        "-n",
        "--nodes",
        type=int,
        nargs="+",
        help="The amount of nodes to test",
        required=True,
    )
    parser.add_argument(
        "-e",
        "--offline_epochs",
        type=int,
        nargs="+",
        help="The offline epochs to test",
        required=True,
    )
    parser.add_argument(
        "-r",
        "--repeats",
        type=int,
        help="The amount of repeats per session",
        required=True,
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        help="The starting seed for the rng",
        required=False,
        default=67,
    )
    parser.add_argument(
        "-a",
        "--algorithm",
        type=str,
        nargs="+",
        choices=["all_reduce", "parameter_server"],
        help="The distributed algorithm to train on",
        required=True,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    TOTAL_ITERATIONS = 1 + args.repeats

    SESSIONS = [
        factory(nodes, offline_epochs, iteration)
        for nodes, offline_epochs in product(args.nodes, args.offline_epochs)
        for factory in (getattr(SessionConfig, algorithm) for algorithm in args.algorithm)
        for iteration in range(TOTAL_ITERATIONS)
    ]

    rng = Random(args.seed)
    seeds = [rng.randint(0, 2**32 - 1) for _ in range(TOTAL_ITERATIONS)]
    docker = Docker(ono_project_root="~/fiuba/tpp")

    for session in SESSIONS:
        addrs = docker.compose_up(session.nodes, release=True)
        training_config = session.config_builder(addrs, seeds[session.iteration])
        exec_training(session.name, training_config, session.iteration)

    docker.compose_down()


if __name__ == "__main__":
    main()
