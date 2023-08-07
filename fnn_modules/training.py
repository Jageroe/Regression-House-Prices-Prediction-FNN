"""
nn_training.py - methods for the whole training process

"""

import time 
from matplotlib import pyplot as plt
import torch

from fnn_modules.utils import get_linear_layers_size

DEVICE = 'cuda'


def _train_step(model: torch.nn.Module, 
                dataloader: torch.utils.data.DataLoader, 
                loss_fn: torch.nn.Module, 
                optimizer: torch.optim.Optimizer,
                device: torch.device = DEVICE):
    
    """Trains a PyTorch model for a single epoch.

    Args:
        model: The PyTorch model to be trained.
        dataloader: DataLoader instance for training data.
        loss_fn: Loss function to minimize.
        optimizer: Optimizer to update model parameters.
        device: Device to perform computation. Default is "cuda".

    Returns:
        float: Average training loss for the epoch.
    """
    
    model.train() # Put model in train mode
    train_loss = 0.0

    # Loop through data loader data batches
    for batch, (X, y) in enumerate(dataloader):

        # Send data to target device
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X).ravel()

        # 2. Calculate and accumulate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item() 

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

    # Calculate average loss
    train_loss = train_loss / len(dataloader)
    return train_loss


def _test_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module,
               device: torch.device=DEVICE):
    """Tests a PyTorch model for a single epoch.

    Args:
        model: The PyTorch model to be trained.
        dataloader: DataLoader instance for training data.
        loss_fn: Loss function to minimize.
        device: Device to perform computation. Default is "cuda".

    Returns:
        float: Average training loss for the epoch.
    """
    # Put model in eval mode
    model.eval() 

    # Setup test loss and test accuracy values
    test_loss = 0

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            y_pred = model(X).ravel()

            # 2. Calculate and accumulate loss
            loss = loss_fn(y_pred, y)
            test_loss += loss.item()

    # Calculate average loss
    test_loss = test_loss / len(dataloader)
    return test_loss


def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          patience: int=None,
          device: torch.device=DEVICE,
          show_plots: bool=False) -> dict[str, list]:
    """
    Trains and tests a PyTorch model.

    Args:
        model: The PyTorch model to be trained and tested.
        train_dataloader: DataLoader for training data.
        test_dataloader: DataLoader for testing data.
        optimizer: Optimizer for model optimization.
        loss_fn: Loss function for calculating loss.
        epochs: Number of training epochs.
        device: Device for computation. Default is "cuda".
        show_plots: Whether to show loss plots. Default is False.

        
    Returns:
        dict: A dictionary containing model information and training/testing metrics.
    """

    start_time = time.time()

    # Create empty results dictionary to store the results from the epochs
    results = {
        "train_loss": [],
        "test_loss": [],
    }

    model.to(device)

    best_test_loss = float("inf")
    early_stop_counter = 0
    last_epoch = None

    # Loop through training and testing steps for a number of epochs
    for epoch in range(epochs):
        train_loss = _train_step(
                        model=model,
                        dataloader=train_dataloader,
                        loss_fn=loss_fn,
                        optimizer=optimizer,
                        device=device
                    )
        
        test_loss  = _test_step(
            model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            device=device
        )

        # Print out what's happening
        if ((epoch +1) % 10 == 0 or epoch == 0):
            print(
                f"Epoch: {epoch+1} | "
                f"train_loss: {train_loss:.4f} | "
                f"test_loss: {test_loss:.4f} | "
            )

        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["test_loss"].append(test_loss)

        if patience:
            # Check for early stopping
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                early_stop_counter = 0
            else:
                early_stop_counter += 1

            if early_stop_counter >= patience:
                print(f"Early stopping: No improvement in {patience} epochs.")
                last_epoch = epoch
                break


    end_time = time.time()
    run_time = end_time - start_time

    # Plotting the learning curve 
    if show_plots:
        fig, (ax1) = plt.subplots(1)

        ax1.plot(results['train_loss'])
        ax1.plot(results['test_loss'])
        ax1.legend(['train','test'])
        ax1.set_title('Loss')

    result_dict = {
        "model_type": model.__class__.__name__,
        "weight_init":model.weight_init_method,
        "input_size": get_linear_layers_size(model.fnn_block)[0],
        "hidden_sizes": get_linear_layers_size(model.fnn_block)[1:],
        "batch_size": train_dataloader.batch_size,
        "epochs":epoch,
        "learning_rate":optimizer.param_groups[0]['lr'],
        # "weight_decay": optimizer.param_groups[0]['weight_decay'], #In this context, i wont use weight decay
        "train_loss": results['train_loss'][-1],
        "test_loss":results['test_loss'][-1],
        "run_time":run_time

    }

    # Return the filled results at the end of the epochs
    return result_dict