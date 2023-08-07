"""
nn_models.py - Contains classes of different FNN model architechtures

"""

from torch import nn
import torch

DEVICE = 'cuda'

def _init_weights(
        fnn_block:torch.nn.Sequential, 
        weight_init_method:str
    ):
    """
    Initialize weights and biases for the linear layers in the given feedforward neural network block.
    
    Args:
        fnn_block (nn.Sequential): A sequential container of nn.Linear layers.
        weight_init_method (str): Weight initialization method. Can be one of ['kaiming_uniform', 'xavier_uniform',
                                 'normal', 'uniform'].
    """
    for module in fnn_block:
        if isinstance(module, nn.Linear):
            if weight_init_method == 'kaiming_uniform':
                # Apply Kaiming (He) initialization to linear layers
                nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='relu')
            elif weight_init_method == 'xavier_uniform':
                # Apply Xavier initialization to linear layers
                nn.init.xavier_uniform_(module.weight)
            elif weight_init_method == 'normal':
                # Apply normal distribution initialization to linear layers
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
            elif weight_init_method == 'uniform':
                # Apply uniform distribution initialization to linear layers
                nn.init.uniform_(module.weight, a=-0.01, b=0.01)
            if module.bias is not None:
                # Set bias to zero for all initialization methods
                module.bias.data.fill_(0.0)


  
class FeedFowardModel1(nn.Module):
    """
    A feedforward neural network model with customizable architecture and weight initialization.
    
    Args:
        input_size: Number of input features.
        hidden_size1: Number of neurons in the first hidden layer.
        hidden_size2: Number of neurons in the second hidden layer.
        hidden_size3: Number of neurons in the third hidden layer.
        output_size: Number of output units. Default is 1.
        weight_init_method: Weight initialization method.
            Can be one of ['kaiming_uniform', 'xavier_uniform', 'normal', 'uniform'].
            Default is 'kaiming_uniform'.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size1: int,
        hidden_size2: int,
        hidden_size3: int,
        output_size: int = 1,
        weight_init_method:str = 'kaiming_uniform'
    ) -> None:
        super().__init__()
        self.weight_init_method = weight_init_method
        self.fnn_block = nn.Sequential(

            nn.Linear(input_size, hidden_size1),
            nn.ReLU(),
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU(),
            nn.Linear(hidden_size2, hidden_size3),
            nn.ReLU(),
            nn.Linear(hidden_size3, output_size),
        )

        # Initialize weights
        _init_weights(self.fnn_block, self.weight_init_method)

    def forward(self, x: torch.Tensor):
        return self.fnn_block(x)

class FeedFowardModel2(nn.Module):
    """
    A feedforward neural network model with customizable architecture and weight initialization.
    
    Args:
        input_size: Number of input features.
        hidden_size1: Number of neurons in the first hidden layer.
        hidden_size2: Number of neurons in the second hidden layer.
        hidden_size3: Number of neurons in the third hidden layer.
        hidden_size4: Number of neurons in the fourth hidden layer.
        output_size: Number of output units. Default is 1.
        weight_init_method: Weight initialization method.
            Can be one of ['kaiming_uniform', 'xavier_uniform', 'normal', 'uniform'].
            Default is 'kaiming_uniform'.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size1: int,
        hidden_size2: int,
        hidden_size3: int,
        hidden_size4: int,
        output_size: int = 1,
        weight_init_method = 'kaiming_uniform'
    ) -> None:
        super().__init__()
        self.weight_init_method = weight_init_method
        self.fnn_block = nn.Sequential(

            nn.Linear(input_size, hidden_size1),
            nn.ReLU(),
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU(),
            nn.Linear(hidden_size2, hidden_size3),
            nn.ReLU(),
            nn.Linear(hidden_size3, hidden_size4),
            nn.ReLU(),
            nn.Linear(hidden_size4, output_size),
        )

        # Initialize weights
        _init_weights(self.fnn_block, self.weight_init_method)

    def forward(self, x: torch.Tensor):
        return self.fnn_block(x)
    

class FeedFowardModel3(nn.Module):
    """
    A feedforward neural network model with customizable architecture and weight initialization.
    
    Args:
        input_size: Number of input features.
        hidden_size1: Number of neurons in the first hidden layer.
        hidden_size2: Number of neurons in the second hidden layer.
        hidden_size3: Number of neurons in the third hidden layer.
        hidden_size4: Number of neurons in the fourth hidden layer.
        output_size: Number of output units. Default is 1.
        weight_init_method: Weight initialization method.
            Can be one of ['kaiming_uniform', 'xavier_uniform', 'normal', 'uniform'].
            Default is 'kaiming_uniform'.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size1: int,
        hidden_size2: int,
        hidden_size3: int,
        hidden_size4: int,
        output_size: int = 1,
        weight_init_method = 'kaiming_uniform'
    ) -> None:
        super().__init__()
        self.weight_init_method = weight_init_method
        self.fnn_block = nn.Sequential(

            nn.Linear(input_size, hidden_size1),
            nn.LeakyReLU(),
            nn.Linear(hidden_size1, hidden_size2),
            nn.LeakyReLU(),
            nn.Linear(hidden_size2, hidden_size3),
            nn.LeakyReLU(),
            nn.Linear(hidden_size3, hidden_size4),
            nn.LeakyReLU(),
            nn.Linear(hidden_size4, output_size),
        )

        # Initialize weights
        _init_weights(self.fnn_block, self.weight_init_method)

    def forward(self, x: torch.Tensor):
        return self.fnn_block(x)
    

class FeedFowardModel4(nn.Module):
    """
    A feedforward neural network model with customizable architecture and weight initialization.
    
    Args:
        input_size: Number of input features.
        hidden_size1: Number of neurons in the first hidden layer.
        hidden_size2: Number of neurons in the second hidden layer.
        hidden_size3: Number of neurons in the third hidden layer.
        hidden_size4: Number of neurons in the fourth hidden layer.
        output_size: Number of output units. Default is 1.
        weight_init_method: Weight initialization method.
            Can be one of ['kaiming_uniform', 'xavier_uniform', 'normal', 'uniform'].
            Default is 'kaiming_uniform'.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size1: int,
        hidden_size2: int,
        hidden_size3: int,
        hidden_size4: int,
        output_size: int = 1,
        weight_init_method = 'kaiming_uniform'
    ) -> None:
        super().__init__()
        self.weight_init_method = weight_init_method
        self.fnn_block = nn.Sequential(

            nn.Linear(input_size, hidden_size1),
            nn.BatchNorm1d(hidden_size1),
            nn.ReLU(),
            nn.Linear(hidden_size1, hidden_size2),
            nn.BatchNorm1d(hidden_size2),
            nn.ReLU(),
            nn.Linear(hidden_size2, hidden_size3),
            nn.BatchNorm1d(hidden_size3),
            nn.ReLU(),
            nn.Linear(hidden_size3, hidden_size4),
            nn.BatchNorm1d(hidden_size4),
            nn.ReLU(),
            nn.Linear(hidden_size4, output_size),
        )

        # Initialize weights
        _init_weights(self.fnn_block, self.weight_init_method)

    def forward(self, x: torch.Tensor):
        return self.fnn_block(x)
    

class FeedFowardModel5(nn.Module):
    """
    A feedforward neural network model with customizable architecture and weight initialization.
    
    Args:
        input_size: Number of input features.
        hidden_size1: Number of neurons in the first hidden layer.
        hidden_size2: Number of neurons in the second hidden layer.
        hidden_size3: Number of neurons in the third hidden layer.
        hidden_size4: Number of neurons in the fourth hidden layer.
        output_size: Number of output units. Default is 1.
        weight_init_method: Weight initialization method.
            Can be one of ['kaiming_uniform', 'xavier_uniform', 'normal', 'uniform'].
            Default is 'kaiming_uniform'.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size1: int,
        hidden_size2: int,
        hidden_size3: int,
        hidden_size4: int,
        output_size: int = 1,
        weight_init_method = 'kaiming_uniform'
    ) -> None:
        super().__init__()
        self.weight_init_method = weight_init_method
        self.fnn_block = nn.Sequential(

            nn.Linear(input_size, hidden_size1),
            nn.BatchNorm1d(hidden_size1),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(hidden_size1, hidden_size2),
            nn.BatchNorm1d(hidden_size2),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(hidden_size2, hidden_size3),
            nn.BatchNorm1d(hidden_size3),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(hidden_size3, hidden_size4),
            nn.BatchNorm1d(hidden_size4),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(hidden_size4, output_size),
        )

        _init_weights(self.fnn_block, self.weight_init_method)

    def forward(self, x: torch.Tensor):
        return self.fnn_block(x)