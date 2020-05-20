import torch.nn as nn
import torch

class CenterLossNet(nn.Module):
    def __init__(self):
        super(CenterLossNet, self).__init__()

        self.layer = nn.Sequential(nn.Conv2d(1, 32, 5, 1, 2),  # 32*28*28
                                   nn.BatchNorm2d(32),
                                   nn.PReLU(),
                                   nn.Conv2d(32, 32, 5, 1, 2),  # 32*28*28
                                   nn.BatchNorm2d(32),
                                   nn.PReLU(),
                                   nn.MaxPool2d(2, 2),  # 32*14*14

                                   nn.Conv2d(32, 64, 5, 1, 2),  # 64*14*14
                                   nn.BatchNorm2d(64),
                                   nn.PReLU(),
                                   nn.Conv2d(64, 64, 5, 1, 2),  # 64*14*14
                                   nn.BatchNorm2d(64),
                                   nn.PReLU(),
                                   nn.MaxPool2d(2, 2),  # 64*7*7

                                   nn.Conv2d(64, 128, 5, 1, 2),  # 128*7*7
                                   nn.BatchNorm2d(128),
                                   nn.PReLU(),
                                   nn.Conv2d(128, 128, 5, 1, 2),  # 128*7*7
                                   nn.BatchNorm2d(128),
                                   nn.PReLU(),

                                   nn.MaxPool2d(2, 2),  # 128*3*3
                                   )
        self.features = nn.Linear(128*3*3, 2)
        self.output = nn.Linear(2, 10)

    def forward(self, x):
        x = self.layer(x)
        x = torch.reshape(x, [-1, 128*3*3])
        features = self.features(x)
        output = torch.log_softmax(self.output(features), dim=1)

        return features, output


if __name__ == "__main__":
    net = CenterLossNet()
    a = torch.randn((2, 1, 28, 28))
    b = net(a)
    print(b)
