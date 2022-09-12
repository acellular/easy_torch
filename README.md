## easy_torch (v0.12)

A set of helper/wrapper classes and functions for easy
one-line creation of cuda-enabled, batch-loaded
MLP and 2d-convolutional neural networks using pytorch


### Usage
#### Data Loading

(See included Jupyter notebook for interactive examples)

To create any supported neural network data must first be in a Dataloader.
These can be loaded in many ways, such as loading from built-in datasets like CIFAR100:
```
transform = transforms.Compose([transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
```
Or by using the added dataloading.from_np() here to create Dataloaders
from numpy arrays (themselves created however you might like):
```
from_np(X_train, X_test, y_train, y_test, batch_size=64)
```

#### Training

To create and train an MLP network (flattens any-dimentional input data):
```
mlp = EasyTorchMLP(features, classes, hidden_layer_sizes=(200,100,...))
mlp.fit(train_dataloader, test_dataloader, epochs=10,
		lr=1e-3, weight_decay=0, opt_func=torch.optim.Adam, verbose=True)
```

To create and train a 2d-convolutional network
(where the first convolutional layer size is 
the number of channels in the original data):
```
covn = EasyTorchConv(classes, conv_layer_sizes=(3,256,...),
					full_layer_sizes=(512,...), batch_norm=True)
covn.fit(train_dataloader, test_dataloader, epochs=10,
		lr=1e-4, weight_decay=1e-4, opt_func=torch.optim.Adam, verbose=True)
```
NOTE: Easiest way to figure out the first full_layer_size: 
enter the convolutional layer sizes you'd like
plus any number for the full layers and wait for the size error


### Dependencies
Tested on Python 3.9.6
Requires torch with cuda (tested on 1.12.1 and CUDA 11.6):
get proper pip download code here: https://pytorch.org/get-started/locally/

sklearn and Jupeter Notebook or Lab
(or relevant plugin for VSCode, Sublime, etc.)
required to run usage-example notebook.


### Development

Possible future additions:  
	- optional residual layers argument for both MLP and convolutional networks  
	- recursive networks (EasyRecurv)  
	- long short-term memory networks (EasyLSTM)  