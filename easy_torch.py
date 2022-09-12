from collections import OrderedDict
import torch

# local
from nn_builder import EasyNN


class EasyTorchMLP(EasyNN): 
    """MLP (fully-connected feed-forward) neural network, subclass of EasyNN"""
    
    def __init__(self, features, classes, hidden_layer_sizes=(200,100)):
        super().__init__()

        # if just wanted a simple linear model
        if isinstance(hidden_layer_sizes, int) and hidden_layer_sizes == 0:
            raise Exception("Must provide at least one hidden layer in the form hidden_layer_sizes=(n,m,...)")
        elif hidden_layer_sizes == None or hidden_layer_sizes[0] == 0:
            raise Exception("Must provide at least one hidden layer in the form hidden_layer_sizes=(n,m,...)")

        layers = OrderedDict()
        layers['flatten'] = torch.nn.Flatten()
        layers['input_layer'] = torch.nn.Linear(features, hidden_layer_sizes[0])
        layers['activation_input'] = torch.nn.ReLU(inplace=True)
        for i in range(1,len(hidden_layer_sizes)):
            layers[f'hidden_layer_{i}'] = torch.nn.Linear(hidden_layer_sizes[i-1], hidden_layer_sizes[i])
            layers[f'activation_{i}'] = torch.nn.ReLU(inplace=True)

        layers['output'] = torch.nn.Linear(hidden_layer_sizes[len(hidden_layer_sizes)-1], classes)
        self.nn = torch.nn.Sequential(layers)

        print('Network design: ', self.nn)



class EasyTorchConv(EasyNN):
    """Convolutional neural network, subclass of EasyNN"""

    def __init__(self, classes, conv_layer_sizes=(3,64), full_layer_sizes=(256,64), batch_norm=False):
        super().__init__()
        
        # check for correct input
        if isinstance(full_layer_sizes, int): full_layer_sizes = (full_layer_sizes,)
        if isinstance(conv_layer_sizes, int): conv_layer_sizes = (conv_layer_sizes,)
        if conv_layer_sizes == None or conv_layer_sizes[0] == 0:
            raise Exception("Must provide at least one non-zero convolutional layer in the form conv_layer_sizes=(n,m,...)")
        if full_layer_sizes == None or full_layer_sizes[0] == 0:
            raise Exception("Must provide at least one non-zero fully connected layer in the form full_layer_sizes=(n,m,...)")

        layers = OrderedDict()

        for i in range(1,len(conv_layer_sizes)):
            layers[f'conv_layer_{i}'] = torch.nn.Conv2d(conv_layer_sizes[i-1], conv_layer_sizes[i],
                                                        kernel_size=3, stride=1, padding=1)
            if batch_norm: layers[f'batchNorm_{i}'] = torch.nn.BatchNorm2d(conv_layer_sizes[i])
            layers[f'activationC_{i}'] = torch.nn.ELU(inplace=True)
            if i % 2 == 0:
                layers[f'maxPool_{i}'] = torch.nn.MaxPool2d(2,2)
                layers[f'dropout_{i}'] = torch.nn.Dropout(p=0.5, inplace=False)

        layers[f'flatten'] = torch.nn.Flatten()

        for i in range(1,len(full_layer_sizes)):
            layers[f'hidden_layer_{i}'] = torch.nn.Linear(full_layer_sizes[i-1], full_layer_sizes[i])
            layers[f'activation_{i}'] = torch.nn.ELU(inplace=True)

        layers['output'] = torch.nn.Linear(full_layer_sizes[len(full_layer_sizes)-1], classes)
        
        self.nn = torch.nn.Sequential(layers)

        print('Network design: ', self.nn)