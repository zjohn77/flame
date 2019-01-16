"""
* Define network architecture in the ConvNet class.
* Specific to the application of text classification, Conv1d and MaxPool1d allow us to
extract 'concepts' from word sequences -- much like n-grams. The key ingredient is setting
channels (the colors in computer vision) to word embedding dimensions.
"""
from torch.nn import Module, Sequential, Conv1d, ReLU, MaxPool1d, Dropout, Linear

class ConvNet(Module):
	'''Define network structure in the constructor; then specify the logic 
	of the forward pass in a method called forward.
   1. Convolve the tensor into 8 and then 16 channels respectively to extract "concepts".
   2. Define how the amount of downsampling, spatial size of input, etc, 
      map into the fully connected (FC) input dimension.
   3. Specify fully connected network having 1 hidden layer with width=hidden_layer_nodes.
	'''
	def __init__(self, input_length, channels, kernel_size, stride, padding, 
                hidden_layer_nodes, output_layer_nodes):		
		super().__init__()
		
		## 1. Convolve the tensor using 2 convolutional layers
		self.conv_layers = Sequential(
         # The 1st Layer:
         Conv1d(channels, # 1 channel for each dimension of word embedding
                8, 
                kernel_size, 
                stride,
                padding
               ),
         ReLU(),
         MaxPool1d(2),

         # The 2nd Layer:
         Conv1d(8, 
                16, 
                kernel_size,
                stride, 
                padding
               ),
         ReLU(),
         MaxPool1d(2),
         
         # The 3rd Layer:
         Conv1d(16, 
                64, 
                kernel_size,
                stride, 
                padding
               ),
         ReLU(),
         MaxPool1d(2)
      )
		self.dropout = Dropout()
		
		## 2. Define fully connected input dimension
		def n_extracted_features(conv_layers, input_length, last_out_channel):
			'''Helper function for creating the attribute "self.N_EXTRACTED_FEATURES"
         that computes the numbers of nodes to feed into the flat, fully connected layer.
			'''
			N_LAYERS = len(conv_layers) / 3
			
			# validate input
			if input_length % 2**N_LAYERS != 0:
				raise Exception('input_length should evenly divide 2**N_LAYERS.')
			
			POOLED_DIM = input_length / 2**N_LAYERS
			
			return int(last_out_channel * POOLED_DIM)  # = the volume of the flattened tensor
                                                    # being fed forward to the FC layer

		## 3. Specify fully connected network using the above dimension calculations.
		self.N_EXTRACTED_FEATURES = n_extracted_features(self.conv_layers, 
																		 input_length, 
																		 64   # must = last Conv1d's out channels
																		)
		self.fc_layers = Sequential(Linear(self.N_EXTRACTED_FEATURES, hidden_layer_nodes),
											 Linear(hidden_layer_nodes, output_layer_nodes)   # there are 20 newsgroups
											)

	def forward(self, _input):
		'''1. Extract features by convolving raw tensor into features.
			2. Flatten the tensor; then feed into fully connected layers.
		'''
		_output = (self.conv_layers(_input)
							.reshape(-1, self.N_EXTRACTED_FEATURES)
					 )
		_output = self.dropout(_output)
		
		return self.fc_layers(_output)