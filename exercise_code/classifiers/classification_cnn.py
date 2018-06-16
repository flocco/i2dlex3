"""ClassificationCNN"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class ClassificationCNN(nn.Module):
    """
    A PyTorch implementation of a three-layer convolutional network
    with the following architecture:

    conv - relu - 2x2 max pool - fc - dropout - relu - fc

    The network operates on mini-batches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, kernel_size=7,
                 stride_conv=1, weight_scale=0.001, pool=2, stride_pool=2, hidden_dim=100,
                 num_classes=10, dropout=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data.
        - num_filters: Number of filters to use in the convolutional layer.
        - kernel_size: Size of filters to use in the convolutional layer.
        - hidden_dim: Number of units to use in the fully-connected hidden layer-
        - num_classes: Number of scores to produce from the final affine layer.
        - stride_conv: Stride for the convolution layer.
        - stride_pool: Stride for the max pooling layer.
        - weight_scale: Scale for the convolution weights initialization
        - pool: The size of the max pooling window.
        - dropout: Probability of an element to be zeroed.
        """
        super(ClassificationCNN, self).__init__()
        channels, height, width = input_dim
        self.num_filters = num_filters
        self.WidthAfterPool = int((width-pool)/stride_pool + 1)
        self.flatLayers = int(num_filters*self.WidthAfterPool*self.WidthAfterPool)
        self.flatfeatures = self.WidthAfterPool*self.WidthAfterPool*channels
        padding = int((kernel_size-1)/2)
        print(weight_scale)
        
        

        ########################################################################
        # TODO: Initialize the necessary trainable layers to resemble the      #
        # ClassificationCNN architecture  from the class docstring.            #
        #                                                                      #
        # In- and output features should not be hard coded which demands some  #
        # calculations especially for the input of the first fully             #
        # convolutional layer.                                                 #
        #                                                                      #
        # The convolution should use "same" padding which can be derived from  #
        # the kernel size and its weights should be scaled. Layers should have #
        # a bias if possible.                                                  #
        #                                                                      #
        # Note: Avoid using any of PyTorch's random functions or your output   #
        # will not coincide with the Jupyter notebook cell.                    #
        ########################################################################
        #(C, H, W) = input_dim.shape
      

        self.conv1 = nn.Conv2d(channels, num_filters, kernel_size, stride_conv, padding, 1, 1)
        self.conv1.weight.data = self.conv1.weight.data*weight_scale

        self.relu1 = nn.ReLU()
        self.maxpool = nn.MaxPool2d(pool, stride_pool, 0, 1, False, False)
        newSize = self.WidthAfterPool*self.WidthAfterPool*self.num_filters
        
        self.fc1 = nn.Linear(newSize, hidden_dim)

        self.dropout = nn.Dropout2d(dropout)
        self.relu2 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        #self._initialize_weights()
    
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

    def forward(self, x):
        """
        Forward pass of the neural network. Should not be called manually but by
        calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """

        ########################################################################
        # TODO: Chain our previously initialized fully-connected neural        #
        # network layers to resemble the architecture drafted in the class     #
        # docstring. Have a look at the Variable.view function to make the     #
        # transition from the spatial input image to the flat fully connected  #
        # layers.                                                              #
        ########################################################################
        #print(x.shape)
        x = x.cuda()
        output = self.conv1(x)
        output2 = self.relu1(output)

        output3 = self.maxpool(output2)
        flattenLength = output3.shape[1]*output3.shape[2]*output3.shape[3] #8192
        
        output3 = output3.view(output2.shape[0], flattenLength)
        
        output4 = self.fc1(output3)
        output5 = self.dropout(output4)
        output6 = self.relu2(output5)
        
        output7 = self.fc2(output6)

        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################
        return output7

    def num_flat_features(self, x):
        """
        Computes the number of features if the spatial input x is transformed
        to a 1D flat input.
        """
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)
