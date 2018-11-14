KERNEL_SIZE = 3
convnet = ConvNet(input_dim = 28,
                  kernel_size = KERNEL_SIZE,
                  stride = 1,
                  padding = int((KERNEL_SIZE - 1) / 2)
               )
fit(model = convnet, 
   training = train_loader, 
   optimizer = t.optim.Adam(convnet.parameters(), 
                           lr = 0.001
                           ),   ## lr: the learn rate
   loss_func = CrossEntropyLoss(),
   n_epochs = 2
   )
accuracy = Accuracy(convnet, holdout_loader)
print(f'{accuracy}')   