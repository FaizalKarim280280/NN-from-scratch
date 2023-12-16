# NN-from-scratch
This is an implementation of Neural Networks using only numpy. Currently, it supports dynamic setting of layers, neurons, activations and 2 loss functios (MSE and Cross Entropy). Available activation functions are ```relu, tanh, sigmoid, softmax, linear, leaky_relu, silu.```

Made as a part of SMAI (Statistical Methods in AI) course assignment. 

# How to use

1. Imports
```python
from nn.net import Model
from nn.losses import CrossEntropyLoss
from nn.layer import Layer
from nn.dataloader import DataLoader
``` 

2. Initialize your model, add layers and loss function.

```python
model = Model()
model.add(Layer(dim_in = 10, dim_out = 16, activation = 'sigmoid'))
model.add(Layer(16, 32, 'sigmoid'))
model.add(Layer(32, 16, 'sigmoid'))
model.add(Layer(16, 3, 'softmax'))

model.loss_fxn = CrossEntropyLoss()
model.lr = 1e-3
```
3. Create train and validation dataloader.

```python
train_loader = DataLoader(X_train, y_train, batch_size=64, drop_last=False)
val_loader = DataLoader(X_val, y_val, batch_size=64, drop_last=False)
```

4. Training Loop.

```python
for epoch in range(epochs):
    loss = 0
    for x, y in train_loader:
        y_pred = model(x)           # forward pass
        loss += model.loss_fxn(y_pred, y)
        model.backward()            # calculate gradients
        model.update_gradients()    # update weights

    loss = loss/len(train_loader)   # take the average loss
```

4. Validation Loop
```py
loss = 0
for x, y in val_loader:
    y_pred = model(x)               
    loss += model.loss_fxn(y_pred, y)
    # only forward pass and no calculating/updating gradients

loss = loss/len(val_loader)
```



