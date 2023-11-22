import torch
from torch import nn
from main import *


class Train():

    def __init__(self, model_parameters) -> None:
        self.loss_fn = nn.L1Loss()
        self.optimizer = torch.optim.SGD(params=model_parameters,
                                         lr=0.01)
        self.train_data = None
        self.train_label = None
        self.test_data = None
        self.test_label = None
        self.epoch_count = []
        self.loss_values = []
        self.test_loss_values = []

    def train_model(self, model, epochs: int):

        torch.manual_seed(42)
        self.epochs = epochs

        for epoch in range(self.epochs):

            model.train()
            y_preds = model(self.train_data)
            # print(y_preds)
            print(epoch)

            loss = self.loss_fn(y_preds, self.train_label)
            self.optimizer.zero_grad()

            loss.backward()
            self.optimizer.step()

            model.eval()

            with torch.inference_mode():
                test_pred = model(self.test_data)
                test_loss = self.loss_fn(test_pred, self.test_label)

            if epoch % 10 == 0:
                print(f"epoch : {epoch} | Loss : {loss} | Test Loss : {test_loss}")
                self.epoch_count.append(epoch)
                self.loss_values.append(loss)
                self.test_loss_values.append(test_loss)
                print(model.state_dict())

        return model, self.epoch_count, self.loss_values, self.test_loss_values
