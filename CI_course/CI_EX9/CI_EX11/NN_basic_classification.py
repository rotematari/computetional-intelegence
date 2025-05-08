import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.optim as optim

##################################################################
### Load and perpare data ###

data = load_digits()
X, y = [], np.array(data.target)
for x in data.images:
    x = x.reshape((-1,))
    X.append(x)
X = np.array(X)

scaler = StandardScaler().fit(X)
X = scaler.transform(X) 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train).to(torch.int64)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test).to(torch.int64)

num_classes = len(np.unique(y_train))
num_features = X_train.shape[1]

##################################################################
### Build model ###
A = {0: nn.ReLU(), 1: nn.Tanh(), 2: nn.Sigmoid(), 3: nn.LeakyReLU()}

# Random choice
z = [0.1, 20, 0, 0, 50, 0.02, 0.1]

# Optimized solution from the PSO
# z = [9.40030419e-02, 2.45659515e+02, 1.34444780e+00, 1.75410454e+00, 2.92340830e+01, 4.55080588e-03, 2.52426353e-04]

learning_rate = z[0] 
batch_size = int(z[1])
activation_function = A[int(z[2])]
h = int(z[3]) # Number of hidden layers
m = int(z[4]) # Number of neurons in each hidden layer
p_do = z[5] # Dropout value
reg = z[6] # Regularization

Layers = [nn.Linear(num_features, m), activation_function]
for i in range(h):
    Layers.append(nn.Linear(m, m))
    Layers.append(nn.Dropout(p = p_do))
    Layers.append(activation_function)
Layers.append(nn.Linear(m, num_classes))
Layers.append(nn.Softmax(dim=1))

model = nn.Sequential(*Layers)

##################################################################
### Preparare for Training ###

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=reg)

epochs = 20

##################################################################
### Train ###


for epoch in range(epochs):

    for i in range(0, len(X_train), batch_size):
        Xbatch = X_train[i:i+batch_size,:]
        y_pred = model(Xbatch) # Feed-forward
        ybatch = y_train[i:i+batch_size]
        loss = loss_fn(y_pred, ybatch) # Evaluate loss
        optimizer.zero_grad() # Zero the gradients before running the backward pass. This is because by default, gradients are accumulated in buffers ( i.e, not overwritten)
        loss.backward() # Compute gradient of the loss with respect to all the learnable parameters of the model
        optimizer.step() # Update weights

    L_test = loss_fn(model(X_test), y_test).detach().numpy()
    print(f'Epoch {epoch} with test loss: {L_test}')

L_test = loss_fn(model(X_test), y_test).detach().numpy()
print()
print(f'Test loss: {L_test}')

### Evaluate the model ###

correct = 0
for x, y in zip(X_test, y_test):
    output = model(x.reshape(1,-1))
    _, predicted = output.max(1)
    correct += (predicted == y).sum().item()

# Print the accuracy
accuracy = correct / X_test.shape[0] * 100
print('Classification success rate: {}%'.format(np.round(accuracy, 2)))




