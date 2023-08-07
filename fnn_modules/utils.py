"""
nn_utils.py - Utility functions and custom loss functions

"""

from torch import nn
import torch

class RMSELoss(nn.Module):

    """
    Root Mean Squared Error (RMSE) Loss.
        
    Args:
        eps (float, optional): A small positive constant added to the MSE before taking the square root
                              to avoid numerical instability. Default is 1e-6.
    """

    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
        
    def forward(self,pred,actual):
        """
        Compute the RMSE loss.
        
        Args:
            pred (torch.Tensor): Predicted values.
            actual (torch.Tensor): Actual target values.
            
        Returns:
            torch.Tensor: Calculated RMSE loss.
        """
        loss = torch.sqrt(self.mse(pred,actual) + self.eps)
        return loss
    


class RMSLELoss(nn.Module):
    """
    Root Mean Squared Logarithmic Error (RMSLE) Loss.
       
    Args:
        eps (float, optional): A small positive constant added to the MSE before taking the square root
                              to avoid numerical instability. Default is 1e-6.
    """

    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
        
    def forward(self, pred, actual):
        """
        Compute the RMSLE loss.
        
        Args:
            pred (torch.Tensor): Predicted values.
            actual (torch.Tensor): Actual target values.
            
        Returns:
            torch.Tensor: Calculated RMSLE loss.
        """
        return torch.sqrt(self.mse(torch.log(pred + 1), torch.log(actual + 1)) + self.eps)
    


def get_linear_layers_size(fnn_block:torch.nn.Sequential):
    """
    Retrieve the number of input features according to the input nn.Sequential object

    Args:
        fnn_block (nn.ModuleList or nn.Sequential): A list or sequential container of nn.Linear layers.

    Returns:
        int: Number of input features for the specified linear layer, or 0 if the index is out of range.

    """
    lin_layers = [layer.in_features for layer in fnn_block if isinstance(layer, torch.nn.modules.linear.Linear)]
    
    return lin_layers