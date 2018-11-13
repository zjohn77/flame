from torch.nn import Module, Sequential, Conv2d, ReLU, MaxPool2d, Dropout, Linear

class ConvNet(Module):
    ''' '''
    def __init__(self, input_dim, kernel_size, stride, padding):
        ''' '''
        super().__init__()
        self.conv_layer_1 = Sequential(Conv2d(1, 
                                              16, 
                                              kernel_size, 
                                              stride, 
                                              padding
                                             ),
                                       ReLU(),
                                       MaxPool2d(2, 2)
                                      )        
        self.conv_layer_2 = Sequential(Conv2d(16, 
                                              32, 
                                              kernel_size, 
                                              stride, 
                                              padding
                                             ),
                                       ReLU(),
                                       MaxPool2d(2, 2)
                                      )    
        self.dropout = Dropout()
        self.full_connect_1 = Linear(7 * 7 * 32, 50)
        self.full_connect_2 = Linear(50, 10)                                

    def forward(self, _input):
        ''' '''
        out = self.conv_layer_1(_input)
        out = self.conv_layer_2(out)        
        out = out.reshape(len(out), -1)
        out = self.dropout(out)
        out = self.full_connect_1(out)
        out = self.full_connect_2(out)
        return out