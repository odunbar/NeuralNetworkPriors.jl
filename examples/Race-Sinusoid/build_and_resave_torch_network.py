import torch
import torch.nn as nn
import numpy as np

#### Neural Network functions and objects 
#Define neural network object
class FeedforwardNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedforwardNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)

def load_weights(model, flattened_params):
    start = 0  # Keep track of the position in the flattened array

    with torch.no_grad():  # Disable gradient tracking during manual weight assignment
        for param in model.parameters():
            param_length = param.numel()  # Number of elements in the parameter tensor
            new_values = flattened_params[start:start + param_length].reshape(param.shape)
            param.copy_(torch.tensor(new_values, dtype=torch.float32))
            start += param_length  # Move the index forward

# Creating network from EKI paper

# Setup for training data
nx = 100
x = np.linspace(-5.0, 5.0, nx).reshape(-1, 1)
y = np.sin((4*np.pi*np.arange(0, nx, 1))/nx)  #forcing 

# Convert to PyTorch tensors (the training data for the network)
x_train = torch.tensor(x, dtype=torch.float32) 
y_train = torch.tensor(y, dtype=torch.float32).view(-1, 1)

# Network parameters (single layer feedforward network with 20 neurons)
input_dim = 1
hidden_dim = 20 
output_dim = 1


model = FeedforwardNN(input_dim, hidden_dim, output_dim) # create network model
weights = np.loadtxt("true_weights.txt", delimiter = ",") # load the weights already calculated
load_weights(model, weights) # load weights into model

# Predict using trained model
y_test_pred = model(x_train).detach().numpy()[:,0]

# save with npz as a dict of numpy arrays - to retain the pytorch shaping
state_dict = model.state_dict()
weights_dict = {k: v.cpu().numpy() for k, v in state_dict.items()}


# Create a flat dictionary for .npz saving (no nested dicts when loading in julia)
npz_dict = {
    "x": x,
    "y": y,
    "input_dim": input_dim,
    "hidden_dim": hidden_dim,
    "output_dim": output_dim,
}

# Add each weight to the npz_dict with its own key
for k, v in weights_dict.items():
    npz_dict[f"weights.{k}"] = v

# Save everything
np.savez("torch_network_and_train_data.npz", **npz_dict)

