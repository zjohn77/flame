from data import train_loader, test_loader
from design import ConvNet
import torch as t
from torch.nn import CrossEntropyLoss

LEARN_RATE = 0.001
EPOCHS = 5

model = ConvNet()
criterion = CrossEntropyLoss()
optimizer = t.optim.Adam(model.parameters(), 
                         lr = LEARN_RATE)

def fit(model_, train_, loss_crit_, optimizer_):
    losses = []
    for i, (image, label) in enumerate(train_):
        optimizer_.zero_grad()
        # Forward.
        pred = model_(image)
        loss = loss_crit_(pred, label)
        losses.append(loss.item())        
        # Backward.
        loss.backward()
        optimizer_.step()
        
        print(i)

for epoch in range(EPOCHS):
    fit(model, train_loader, criterion, optimizer)