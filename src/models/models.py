"""
Define ConvNet as a model to be imported by main.py in order to fit a convolutional network. 
"""
from torch.nn import Module, Sequential, Conv2d, ReLU, MaxPool2d, Dropout, Linear

class ConvNet(Module):
    '''Define network structure in the constructor; then specify the logic of the forward pass 
    in a method called forward.
    '''
    def __init__(self, input_dim, kernel_size, stride, padding):
        '''1. Convolve the single gray channel image into 16 and 32 channels respectively.
           2. Define how the amount of downsampling, spatial size of input, etc, map into the FC input dimension.
           3. Specify FC network with a hidden layer of width 50.
        '''
        super().__init__()
        
        ## 1. Convolve the image using 2 convolutional layers
        self.conv_layers = Sequential(Conv2d(1, 
                                             16, 
                                             kernel_size, 
                                             stride, 
                                             padding
                                            ),
                                      ReLU(),
                                      MaxPool2d(2, 2),
                                      ## layer 1

                                      Conv2d(16, 
                                             32, 
                                             kernel_size,
                                             stride, 
                                             padding
                                            ),
                                      ReLU(),
                                      MaxPool2d(2, 2)
                                      ## layer 2
                                     )
        self.dropout = Dropout()
        
        ## 2. Define FC input dimension
        def n_extracted_features(conv_layers, input_dim, last_out_channel):
            N_LAYERS = len(conv_layers) / 3
            POOLED_DIM = input_dim / 2**N_LAYERS
            return int(last_out_channel * POOLED_DIM**2)

        ## 3. Specify FC network
        self.N_EXTRACTED_FEATURES = n_extracted_features(self.conv_layers, input_dim, 32)
        self.fc_layers = Sequential(Linear(self.N_EXTRACTED_FEATURES, 50),
                                    Linear(50, 10)
                                   )

    def forward(self, _input):
        '''1. Extract features by convolving raw pixel into (7 by 7 by 32) feature tensors.
           2. Flatten the tensor; then feed into fully connected layers.
        '''
        _output = (self.conv_layers(_input)
                       .reshape(-1, self.N_EXTRACTED_FEATURES)
                  )
        _output = self.dropout(_output)
        return self.fc_layers(_output)