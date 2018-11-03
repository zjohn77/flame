import argparse
import torch as t
from torchvision import datasets, transforms

class Net(t.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = t.nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = t.nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = t.nn.Dropout2d()
        self.fc1 = t.nn.Linear(320, 50)
        self.fc2 = t.nn.Linear(50, 10)

    def forward(self, x):
        x = t.nn.functional.relu(t.nn.functional.max_pool2d(self.conv1(x), 2))
        x = t.nn.functional.relu(t.nn.functional.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = t.nn.functional.relu(self.fc1(x))
        x = t.nn.functional.dropout(x, training=self.training)
        x = self.fc2(x)
        return t.nn.functional.log_softmax(x, dim=1)

def fit(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = t.nn.functional.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with t.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += t.nn.functional.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and t.cuda.is_available()

    t.manual_seed(args.seed)

    device = t.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    

    MEAN = 0.1307
    STANDARD_DEV = 0.3081
    TRANSF = transforms.Compose([transforms.ToTensor(), 
                                transforms.Normalize((MEAN,), 
                                                     (STANDARD_DEV,)
                                                    )
                                ])
    TRAIN_DATASET = datasets.MNIST(root = '../data', 
                                   train = True, 
                                   download = True,
                                   transform = TRANSF
                                  )
    TEST_DATASET = datasets.MNIST(root = '../data', 
                                  train = False, 
                                  download = False,
                                  transform = TRANSF
                                 )
    train_loader = t.utils.data.DataLoader(dataset = TRAIN_DATASET,
                                           batch_size = args.batch_size, 
                                           shuffle = True, 
                                           **kwargs
                                          )
    test_loader = t.utils.data.DataLoader(dataset = TEST_DATASET,
                                          batch_size = args.test_batch_size, 
                                          shuffle = True, 
                                          **kwargs
                                         )


    model = Net().to(device)
    optimizer = t.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        fit(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)

if __name__ == '__main__':
    main()