import torch
from torch.utils.data import TensorDataset, DataLoader


def from_np(X_train, X_test, y_train, y_test, batch_size=1):
    """Create DataLoaders from numpy arrays"""

    # convert to PyTorch tensors
    X_train = torch.from_numpy(X_train).float()
    X_test = torch.from_numpy(X_test).float()
    y_train = torch.from_numpy(y_train).float()
    y_test = torch.from_numpy(y_test).float()
    
    train_dataset = TensorDataset(X_train, y_train) #CAN BE USED TO 2D as well --simply indexed along first dimension
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size) #TODO--add shuffle?
    test_dataset = TensorDataset(X_test, y_test)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    return train_dataloader, test_dataloader


def to_dev(data, device):
    """Send data to device"""

    if isinstance(data,(list,tuple)):
        return [to_dev(x,device) for x in data] #TODO--add non_blocking here??
    return data.to(device, non_blocking=True)


class DLWrapper:
    """Wrap a dataloader to move data to a device"""
    
    def __init__(self, dl, dev):
        self.dl = dl
        self.dev = dev
    
    def __iter__(self):
        """Move next batch to device and yield"""
        for batch in self.dl:
            yield to_dev(batch, self.dev)
            
    def __len__(self):
        return len(self.dl)