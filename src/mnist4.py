

# Test the model
model.eval()
with t.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = t.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Test Accuracy of the model on the 10000 test images: {} %'.format((correct / total) * 100))

def test(model_, test_, lossfunc_):
    model_.eval()
    test_loss = 0
    correct = 0
    with t.no_grad():
        for image, label in test_:
            pred = model_(image)
            test_loss += lossfunc_(pred, 
                                   label, 
                                   reduction = 'sum').item()

            pred = pred.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()


    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

