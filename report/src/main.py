from orchestra import Sequential
from orchestra.arch import Conv2d, Dense, MaxPooling
from orchestra.initialization import Xavier
from orchestra.activations import Tanh


def lenet5_model() -> Sequential:
    return Sequential(
        [
            Conv2d(
                input_dim=(1, 28, 28),
                kernel_dim=(6, 1, 5),
                stride=1,
                padding=2,
                init=Xavier(),
                act=Tanh()
            ),
            MaxPooling(),
            Conv2d(
            ),
            MaxPooling(),
            Dense(),
        ]
    )
