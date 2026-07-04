from orchestra import Sequential
from orchestra.activations import Softmax, Tanh
from orchestra.arch import Conv2d, Dense, MaxPooling
from orchestra.initialization import Xavier

from torch import nn
from safetensors.torch import load_file


class LeNet5Inference(nn.Module):
    """
    The LeNet5 architecture implemented in PyTorch.
    """

    def __init__(self):
        super(LeNet5Inference, self).__init__()

        self.layer_0 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2)
        self.layer_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer_2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        self.layer_3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.layer_4 = nn.Linear(16 * 5 * 5, 120)
        self.layer_5 = nn.Linear(120, 84)
        self.layer_6 = nn.Linear(84, 10)

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.tanh(self.layer_0(x))
        x = self.layer_1(x)
        x = self.tanh(self.layer_2(x))
        x = self.layer_3(x)
        x = self.flatten(x)
        x = self.tanh(self.layer_4(x))
        x = self.tanh(self.layer_5(x))
        x = self.softmax(self.layer_6(x))
        return x


def load_lenet5_from_safetensors(safetensors_path: str) -> nn.Module:
    """
    Instantiates the LeNet5 model and injects the pre-trained safetensors parameters.
    """
    state_dict = load_file(safetensors_path)
    pytorch_state_dict = {}

    for i in (0, 2, 4, 5, 6):
        w_key = f"layer_{i}.weight"
        w_tensor = state_dict[w_key]

        b_key = f"layer_{i}.bias"
        b_tensor = state_dict[b_key]

        if i in (4, 5, 6):
            # Transpose Linear layers [in, out] to PyTorch's [out, in].
            w_tensor = w_tensor.t()

        pytorch_state_dict[w_key] = w_tensor
        pytorch_state_dict[b_key] = b_tensor

    model = LeNet5Inference()
    model.load_state_dict(pytorch_state_dict, strict=True)
    model.eval()
    return model


def build_lenet5_model_config() -> Sequential:
    """
    Builds a model configuration with Lenet5's paper architecture.
    """
    return Sequential(
        [
            Conv2d(
                input_dim=(1, 28, 28),
                kernel_dim=(6, 1, 5),
                stride=1,
                padding=2,
                init=Xavier(),
                act_fn=Tanh(),
            ),
            MaxPooling(
                input_dim=(6, 28, 28),
                filter_size=2,
                stride=2,
                padding=0,
            ),
            Conv2d(
                input_dim=(6, 14, 14),
                kernel_dim=(16, 6, 5),
                stride=1,
                padding=0,
                init=Xavier(),
                act_fn=Tanh(),
            ),
            MaxPooling(
                input_dim=(16, 10, 10),
                filter_size=2,
                stride=2,
                padding=0,
            ),
            Dense(
                output_size=120,
                init=Xavier(),
                act_fn=Tanh(),
            ),
            Dense(
                output_size=84,
                init=Xavier(),
                act_fn=Tanh(),
            ),
            Dense(
                output_size=10,
                init=Xavier(),
                act_fn=Softmax(),
            ),
        ]
    )
