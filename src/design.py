import torch as t
from torch.nn import Sequential, Conv2d, ReLU, MaxPool2d, Dropout, Linear, CrossEntropyLoss

class ConvNet(t.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = Sequential(Conv2d(1, 32, kernel_size=5, padding=2),
                                 ReLU(),
                                 MaxPool2d(kernel_size=2, stride=2)
                                )        
        self.layer2 = Sequential(Conv2d(32, 64, kernel_size=5, padding=2),
                                 ReLU(),
                                 MaxPool2d(kernel_size=2, stride=2)
                                )
        self.drop_out = Dropout()
        self.fc1 = Linear(7 * 7 * 64, 1000)
        self.fc2 = Linear(1000, 10)                                

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out
