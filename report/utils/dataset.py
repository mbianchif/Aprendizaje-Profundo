from os import path
import subprocess

from orchestra.datasets import LocalDataset


DOWNLOAD_DIR = "datasets"
MNIST_DATASET_DIR = f"{DOWNLOAD_DIR}/mnist"


def get_mnist_dataset() -> LocalDataset:
    """
    Downloads a copy of the MNIST dataset if not downloaded
    and returns a `LocalDataset` instance pointing to that
    dataset.
    """
    if not path.exists(MNIST_DATASET_DIR):
        TARGET_URL = "https://drive.google.com/uc?export=download&id=18kPPrx97omjSkVltx_INglQPvylsbhbI"
        ZIP_DOWNLOAD_PATH = f"{DOWNLOAD_DIR}/mnist.zip"

        subprocess.run(["mkdir", "-p", DOWNLOAD_DIR])
        subprocess.run(["curl", "-L", "-o", ZIP_DOWNLOAD_PATH, TARGET_URL])
        subprocess.run(["unzip", ZIP_DOWNLOAD_PATH, "-d", DOWNLOAD_DIR])

    return LocalDataset(
        samples_path=f"{MNIST_DATASET_DIR}/mnist_train_samples.bin",
        labels_path=f"{MNIST_DATASET_DIR}/mnist_train_labels.bin",
        x_size=28**2,
        y_size=10,
    )
