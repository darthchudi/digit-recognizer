import torch.nn as nn

class Network(nn.Module):
    def __init__(self, input_dim, output_dim, use_conv2d=True):
        super().__init__()

        if use_conv2d:
            self.model = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1, stride=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 2), stride=2),
                nn.Flatten(start_dim=1),
                nn.Linear(6272, 3136),
                nn.ReLU(),
                nn.Linear(3136, 784),
                nn.ReLU(),
                nn.Linear(784, output_dim),
            )
        else:
            self.model = nn.Sequential(
                nn.Flatten(start_dim=1),
                nn.Linear(input_dim, 784),
                nn.ReLU(),
                nn.Linear(784, 784),
                nn.ReLU(),
                nn.Linear(784, output_dim),
            )

    def forward(self, x):
        y = self.model(x)
        return y