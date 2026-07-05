import orchestra
from orchestra import PyTrainingConfig
from orchestra.loss_fns import CrossEntropy
from orchestra.optimizers import Adam
from orchestra.store import BlockingStore
from orchestra.sync import BarrierSync
from orchestra.serializer import SparseSerializer

from utils.dataset import get_mnist_dataset


def __get_common_training_config_params() -> dict:
    """
    These are the common hyperparameters for every
    training session.

    The max epoch count and batch size will remain
    constant throughout the notebook.
    """
    return {
        "dataset": get_mnist_dataset(),
        "optimizer": Adam(
            lr=0.01,
            b1=0.9,
            b2=0.999,
            eps=1e-8,
        ),
        "loss_fn": CrossEntropy(),
        "max_epochs": 32,
        "batch_size": 64,
        "serializer": SparseSerializer(r=0.5),
    }


def build_mnist_parameter_server_training_config(
    addrs: list[str],
    nservers: int,
    offline_epochs: int,
) -> PyTrainingConfig:
    """
    Builds a training configuration for a Parameter Server setup
    in it's blocking variant.
    """
    return orchestra.parameter_server(
        addrs=addrs,
        nservers=nservers,
        sync=BarrierSync(),
        store=BlockingStore(),
        offline_epochs=offline_epochs,
        **__get_common_training_config_params(),
    )


def build_mnist_all_reduce_training_config(
    addrs: list[str],
    offline_epochs: int,
) -> PyTrainingConfig:
    """
    Builds a training configuration for an All Reduce setup.
    """
    return orchestra.all_reduce(
        addrs=addrs,
        offline_epochs=offline_epochs,
        **__get_common_training_config_params(),
    )
