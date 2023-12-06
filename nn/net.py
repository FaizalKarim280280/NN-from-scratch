from nn.losses import *
import numpy as np

class Model:
    def __init__(self,
                 loss_fxn=None,
                 logger=None,
                 lr=1e-3,
                 type='regression',
                 epochs=1000,
                 verbose=False):

        self.loss_fxn = loss_fxn
        self.layers = []
        self.lr = lr
        self.dW, self.dB = [], []
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': []
        }
        self.accuracy = self.classification_accuracy if type == 'classification' else self.regression_accuracy
        self.logger = logger
        self.epochs = epochs
        self.verbose = verbose

    def classification_accuracy(self, y_pred, y):
        return accuracy_score(np.argmax(y, axis=-1), np.argmax(y_pred, axis=-1))

    def regression_accuracy(self, y_pred, y):
        return r2_score(y, y_pred)

    def __str__(self):
        out = ""
        for layer in self.layers:
            out += layer.__str__() + "\n"
        return out

    def add(self, layer):
        self.layers.append(layer)

    def __call__(self, x):
        '''
            x: (bs, dim_in)
        '''
        for layer in self.layers:
            x = layer(x)
        return x

    def backward(self):
        dLy = self.loss_fxn.grad()
        common = dLy

        for i in range(len(self.layers) - 1, -1, -1):
            dzw, dzx, daz = self.layers[i].get_grads()
            if i != len(self.layers) - 1:
                common = common @ self.layers[i + 1].W
            common = common * daz
            dw = common[:, :, None] * dzw
            db = common[:, :] * 1

            self.dW.append(np.mean(dw, axis=0))
            self.dB.append(np.mean(db, axis=0))

    def update_gradients(self):
        for i, (dw, db) in enumerate(zip(reversed(self.dW), reversed(self.dB))):
            self.layers[i].W += - self.lr * dw
            self.layers[i].b += - self.lr * db

        self.dW.clear()
        self.dB.clear()

    def training_step(self, loader):
        loss, acc = 0, 0
        for x, y in loader:
            y_pred = self.__call__(x)
            loss += self.loss_fxn(y, y_pred)
            acc += self.accuracy(y_pred, y)

            self.backward()
            self.update_gradients()

        return loss / len(loader), acc / len(loader)

    def validate_step(self, loader):
        loss, acc = 0, 0
        for x, y in loader:
            y_pred = self.__call__(x)
            loss += self.loss_fxn(y, y_pred)
            acc += self.accuracy(y_pred, y)

        return loss / len(loader), acc / len(loader)

    def train(self, X, y):

        train_loader = DataLoader(X, y, batch_size=len(X))

        for epoch in range(self.epochs):

            train_loss, train_acc = self.training_step(train_loader)
            # val_loss, val_acc = self.validate_step(val_loader)

            self.history['train_loss'].append(train_loss)
            # self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            # self.history['val_acc'].append(val_acc)

            if epoch % 20 == 0 and self.verbose:
                print(f"epoch: {epoch} \tTrain:[loss:{train_loss:.4f} acc:{train_acc:.4f}]]")

            if self.logger is not None and self.verbose:
                self.logger.log(
                    {"train_acc": train_acc, "train_loss": train_loss, 'val_acc': val_acc, 'val_loss': val_loss})
        if not self.verbose:
            print(f"epoch: {epoch} \tTrain:[loss:{train_loss:.4f} acc:{train_acc:.4f}]]")

