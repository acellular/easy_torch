import torch

# local
import dataloading


def sum_mean(outputs):
    """helper function to get mean of training and evaluation outputs"""
    batch_losses = sum(x['loss'] for x in outputs)
    epoch_loss = batch_losses / len(outputs)  # Combine losses
    batch_accs = sum(x['accuracy'] for x in outputs)
    epoch_acc = batch_accs / len(outputs)
    return {'loss': epoch_loss, 'accuracy': epoch_acc}


class EasyNN(torch.nn.Module):
    """Base extension torch.nn.Module containing standard fit() method"""

    def train_test(self, dl, optimizer):
        """Basic training and prediction function
        
        (Remember to put mlp into training or evaluation modes first)"""
        outputs = []
        for batch in dl:
            X, y = batch # Dataset tensors
            loss_func = torch.nn.CrossEntropyLoss() 

            # run model, calculate loss
            # TODO--might need to replace with forward() somewhere
            #       for custom with residual layers and recursive later
            logits = self.nn(X) #creating the predictions
            loss = loss_func(logits, y.long())
            
            # important to be done in this order so gradients aren't carried over
            if self.training:
                optimizer.zero_grad()
                loss.backward() #backpropagation
                optimizer.step() #update model parameters (i.e. weights)
            
            # update loss, accuracy
            loss_score = loss.item()
            accuracy = (y == logits.max(1)[1]).sum().item() / y.size(0) #[1] because output is max values, indices
            outputs.append({'loss': loss_score, 'accuracy': accuracy})
            
        return sum_mean(outputs)
        

    # for more optimized training using smaller samples for each model iteration
    def fit(self, train_dataloader, test_dataloader, epochs=10,
        lr=1e-2, weight_decay=0, opt_func=torch.optim.SGD, verbose=True):
        """Fit an EasyNN network to given training data
        and print loss and accuracy on both training and test data"""

        print(f"#### Start training: ####")

        optimizer = opt_func(self.nn.parameters(), lr=lr, weight_decay=weight_decay)
            
        dev = "cuda" if torch.cuda.is_available() else "cpu" # can use CUDA?
        print(f"Device: {dev}")

        # initialize model and print shape
        self.nn.to(dev)
        #print(torch.cuda.memory_summary(device=None, abbreviated=False))
        #torch.cuda.empty_cache() #TODO--needed?

        train_dataloader = dataloading.DLWrapper(train_dataloader, dev)
        test_dataloader = dataloading.DLWrapper(test_dataloader, dev)

        print("Training...")
        results_train = []
        results_test = []
        # loop epochs
        for epoch in range(0, epochs):
            
            # training mode
            self.train() # put into training mode so model parameters updated during backpropagation.
            results_train.append(self.train_test(train_dataloader, optimizer))
            
            # evaluation mode
            self.eval() # switch in to eval so no gradient change
            with torch.no_grad(): # gradient computation turned off
                results_test.append(self.train_test(test_dataloader, optimizer))
            
            if verbose or epoch == 0 or epoch % 10 == 0:
                print(f"Epoch {epoch}: Train loss: {results_train[epoch]['loss']:.3f},",
                                    f"accuracy: {results_train[epoch]['accuracy']:.3f} ::",
                                    f"Test loss: {results_test[epoch]['loss']:.3f}",
                                    f"accuracy: {results_test[epoch]['accuracy']:.3f}")

        print(f"Training finished.\nFinal train loss: {results_train[epochs-1]['loss']:.3f}", 
                                        f" accuracy: {results_train[epochs-1]['accuracy']:.3f}")
        print(f"Final test loss: {results_test[epochs-1]['loss']:.3f}",
                    f" :: accuracy: {results_test[epochs-1]['accuracy']:.3f}")