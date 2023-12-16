import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

import sklearn
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from nn.net import Model
from nn.losses import CrossEntropyLoss
from nn.layer import Layer
from nn.dataloader import DataLoader

class Trainer:
    def __init__(self,
                 model):
        self.model = model
        self.accuracy = accuracy_score

    def forward(self, x):
        return self.model(x)

    def training_step(self, x, y):
        y_pred = self.forward(x)
        loss = self.model.loss_fxn(y, y_pred)
        # acc = self.accuracy(np.argmax(y, axis=-1), np.argmax(y_pred, axis=-1))

        self.model.backward()
        self.model.update_gradients()

        return loss, 0

    def validation_step(self, x, y):
        y_pred = self.forward(x)
        loss = self.model.loss_fxn(y, y_pred)
        # acc = self.accuracy(np.argmax(y, axis=-1), np.argmax(y_pred, axis=-1))

        return loss, 0

    @staticmethod
    def go_one_epoch(loader, step_fxn):
        loss, acc = 0, 0
        for x, y in loader:
            loss_batch, acc_batch = step_fxn(x, y)
            loss += loss_batch
            acc += acc_batch
        return loss/len(loader), acc/len(loader)

    def train(self,
              train_loader,
              val_loader,
              epochs):

        for epoch in (range(epochs)):
            train_loss, train_acc = self.go_one_epoch(train_loader, self.training_step)
            val_loss, val_acc = self.go_one_epoch(val_loader, self.validation_step)

            if epoch % 10 == 0:
                print(f"Epoch:[{epoch}]")
                print(f"Train:[loss: {train_loss:.4f} acc:{train_acc:.4f}]")
                print(f"Val: [loss: {val_loss:.4f} acc:{val_acc:.4f}]")
                print()
                
            if (epoch + 1) % 500 == 0:
                self.model.lr = self.model.lr/3


def main():

    X, y = make_multilabel_classification(n_samples=3000, n_features=8, n_classes=5, n_labels=3, allow_unlabeled=False)

    standard_scaler = StandardScaler()
    X = standard_scaler.fit_transform(X)
    print(y)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

    train_loader = DataLoader(X_train, y_train, batch_size=64, drop_last=False)
    val_loader = DataLoader(X_val, y_val, batch_size=64, drop_last=False)

    model = Model()
    model.add(Layer(8, 16, 'sigmoid'))
    model.add(Layer(16, 32, 'sigmoid'))
    model.add(Layer(32, 32, 'sigmoid'))
    model.add(Layer(32, 5, 'sigmoid'))      # apply sigmoid in last layer since more than one neuron can be 1 (multi-label task)

    model.loss_fxn = CrossEntropyLoss()
    model.lr = 1e-2

    print(model)

    trainer = Trainer(model=model)

    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=2000
    )


if __name__ == "__main__":
    main()

