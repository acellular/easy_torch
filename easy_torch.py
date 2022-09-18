from collections import OrderedDict
from turtle import forward
import torch

# local
from nn_builder import EasyNN


class EasyTorchMLP(EasyNN): 
    """MLP (fully-connected feed-forward) neural network, subclass of EasyNN"""
    
    def add_layer(self, layers, i, layer0, layer1):
        
        layers[f'layer_{i}'] = torch.nn.Linear(layer0, layer1)
        if self.batch_norm: layers[f'batchNorm_{i}'] = torch.nn.BatchNorm1d(layer1)
        layers[f'activation_{i}'] = self.actv_func(inplace=True)


    def __init__(self, features, classes, hidden_layer_sizes=(200,100),
                residual_inserts=None, batch_norm=False, actv_func=torch.nn.ReLU):
        super().__init__()

        self.batch_norm = batch_norm
        self.actv_func = actv_func

        # proper input?
        if isinstance(hidden_layer_sizes, int): hidden_layer_sizes = (hidden_layer_sizes,)
        if hidden_layer_sizes == None or hidden_layer_sizes[0] == 0:
            raise Exception("Must provide at least one hidden layer in the form hidden_layer_sizes=(n,m,...)")

        self.net = torch.nn.ModuleDict()

        # so can take any dimensional data
        self.net['flatten'] = torch.nn.Flatten()
        self.add_layer(self.net, 0, features, hidden_layer_sizes[0]) # and input layer

        # hidden layers
        for i in range(1,len(hidden_layer_sizes)):
            self.add_layer(self.net, i, hidden_layer_sizes[i-1], hidden_layer_sizes[i])

            if residual_inserts != None and i in residual_inserts:
                self.residuals = OrderedDict()
                for f in range(0,2): #TODO replace with argument?
                    self.add_layer(self.residuals, f, hidden_layer_sizes[i], hidden_layer_sizes[i])
                self.net[f'residual_{i}'] = torch.nn.Sequential(self.residuals)

        self.net['output'] = torch.nn.Linear(hidden_layer_sizes[len(hidden_layer_sizes)-1], classes)

        print("Network design: ", self)


class EasyTorchConv(EasyNN): #now with residual!
    """Convolutional neural network, subclass of EasyNN"""

    def add_layer(self, layers, i, layer0, layer1):
        
        layers[f'layer_{i}'] = torch.nn.Conv2d(layer0, layer1, kernel_size=3, stride=1, padding=1)
        if self.batch_norm: layers[f'batchNorm_{i}'] = torch.nn.BatchNorm2d(layer1)
        layers[f'activation_{i}'] = self.actv_func(inplace=True)


    def __init__(self, classes, conv_layer_sizes=(3,64), full_layer_sizes=(256,64),
                    residual_inserts=None, batch_norm=False, actv_func=torch.nn.ReLU):
        super().__init__()

        self.batch_norm = batch_norm
        self.actv_func = actv_func
        
        # proper input?
        if isinstance(full_layer_sizes, int): full_layer_sizes = (full_layer_sizes,)
        if isinstance(conv_layer_sizes, int): conv_layer_sizes = (conv_layer_sizes,)
        if conv_layer_sizes == None or conv_layer_sizes[0] == 0:
            raise Exception("Must provide at least one non-zero convolutional layer in the form conv_layer_sizes=(n,m,...)")
        if full_layer_sizes == None or full_layer_sizes[0] == 0:
            raise Exception("Must provide at least one non-zero fully connected layer in the form full_layer_sizes=(n,m,...)")

        self.net = torch.nn.ModuleDict()

        for i in range(1,len(conv_layer_sizes)):
            self.add_layer(self.net, i, conv_layer_sizes[i-1], conv_layer_sizes[i])

            if residual_inserts != None and i in residual_inserts:
                self.residuals = OrderedDict()
                for f in range(0,2): #TODO replace with argument?
                    self.add_layer(self.residuals, f, conv_layer_sizes[i], conv_layer_sizes[i])
                self.net[f'residual_{i}'] = torch.nn.Sequential(self.residuals)
            
            if i % 2 == 0 or (residual_inserts != None and i in residual_inserts):
                self.net[f'maxPool_{i}'] = torch.nn.MaxPool2d(2,2)
                self.net[f'dropout_{i}'] = torch.nn.Dropout(p=0.5, inplace=False)
            

        self.net[f'flatten'] = torch.nn.Flatten()

        for i in range(1,len(full_layer_sizes)):
            self.net[f'hidden_layer_{i}'] = torch.nn.Linear(full_layer_sizes[i-1], full_layer_sizes[i])
            self.net[f'activation_{i}'] = torch.nn.ELU(inplace=True)

        self.net['output'] = torch.nn.Linear(full_layer_sizes[len(full_layer_sizes)-1], classes)

        print("Network design: ", self)