import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from PSO import PSO
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.optim as optim

"""
This code uses the PSO class defined in the previous code to optimize the hyperparameters of a neural network for classifying handwritten digits.
The train_function function trains a neural network with hyperparameters specified by the input Z and returns the test loss.
The bounds array specifies the lower and upper bounds for each hyperparameter.
The code then creates an instance of the PSO class, defines the optimization problem, and runs the optimization algorithm to find the best set of hyperparameters.
Finally, the code prints the best set of hyperparameters and their corresponding test loss.


"""


### Load and prepare data ###

data = load_digits() # load the digits dataset
X, y = [], np.array(data.target) # extract the target values

# Reshape the images into 1D arrays
for x in data.images:
    x = x.reshape((-1,))
    X.append(x)
X = np.array(X)

# Scale the data to have zero mean and unit variance
scaler = StandardScaler().fit(X)
X = scaler.transform(X) 

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

# Convert the data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train).to(torch.int64)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test).to(torch.int64)

num_classes = len(np.unique(y_train)) # number of classes in the target variable
num_features = X_train.shape[1] # number of features in the input data

##################################################################
A = {0: nn.ReLU(), 1: nn.Tanh(), 2: nn.Sigmoid(), 3: nn.LeakyReLU()} # dictionary of activation functions

def train_function(Z):
    # Function to train a neural network with hyperparameters specified by Z

    L_test = [] # list to store the test loss for each set of hyperparameters
    
    for z in Z:
        learning_rate = z[0] # learning rate for the optimizer
        batch_size = int(z[1]) # batch size for training
        activation_function = A[int(z[2])] # activation function for the hidden layers
        h = int(z[3]) # number of hidden layers
        m = int(z[4]) # number of neurons in each hidden layer
        p_do = z[5] # dropout rate for the hidden layers
        reg = z[6] # regularization parameter for weight decay

        # Define the architecture of the neural network using a list of layers
        Layers = [nn.Linear(num_features, m), activation_function]
        
        for i in range(h):
            Layers.append(nn.Linear(m, m))
            Layers.append(nn.Dropout(p = p_do))
            Layers.append(activation_function)
            
        Layers.append(nn.Linear(m, num_classes))
        Layers.append(nn.Softmax(dim=1))
        
        model = nn.Sequential(*Layers) # create a PyTorch model from the list of layers

        loss_fn = nn.CrossEntropyLoss() # define the loss function as cross-entropy loss
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay = reg) # define the optimizer as Adam with weight decay

        epochs = 6 # number of epochs to train for

        for epoch in range(epochs):
            for i in range(0, len(X_train), batch_size):
                Xbatch = X_train[i:i+batch_size,:] # extract a batch of training data
                
                y_pred = model(Xbatch) # compute predictions using a forward pass through the model
                
                ybatch = y_train[i:i+batch_size] # extract corresponding target values
                
                loss = loss_fn(y_pred, ybatch) # compute the loss
                
                optimizer.zero_grad() # zero the gradients before running the backward pass
                
                loss.backward() # compute gradients using backpropagation
                
                optimizer.step() # update model parameters using gradient descent

        L_test.append(float(loss_fn(model(X_test), y_test).detach().numpy())) # compute test loss and append to list

    return np.array(L_test)


#### Optimize ###

bounds = np.array([[1e-6, 2, 0, 0, 5, 0, 0], [1e-1, 300, 4, 10, 70, 0.8, 1e-3]]) 

P = PSO(live_plot = True) # create an instance of the PSO class
P.define_problem(func = train_function, n_particles = 10, bounds = bounds) # define the optimization problem
z, f = P.run(max_iter = 10) # run the optimization algorithm

learning_rate = z[0] 
batch_size = int(z[1])
activation_function = A[int(z[2])]
h = int(z[3]) # Number of hidden layers
m = int(z[4]) # Number of neurons in each hidden layer
p_do = z[5] # Dropout value
reg = z[6] # Regularization

print(f'Solution: {z}')
print(f'--- Best fitness {f} with:')
print(f'Learning rate: {learning_rate}')
print(f'Batch size: {batch_size}')
print(f'Activation function: {str(activation_function)}')
print(f'{h} hidden layers with {m} neurons each')
print(f'Dropout: {p_do}, regularization: {reg}')

