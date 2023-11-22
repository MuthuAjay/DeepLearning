import torch
from torch import nn
from utils import *

class TrainModel():
    def __init__(self, epochs : int= 1000, manual_seed: int|float = 42) -> None:
        self.epochs = epochs
        self.MANUALSEED = manual_seed
        self.binary_loss = nn.BCEWithLogitsLoss()
        self.Multiclass_loss = nn.CrossEntropyLoss()
        self.optimizer = None
    
    def train_binary_class_model(self, model, X_train: torch.Tensor,
              X_test:torch.Tensor,
              y_train: torch.Tensor,
              y_test: torch.Tensor,
              learning_rate : int|float) -> nn.Module :
        torch.manual_seed(self.MANUALSEED)
        torch.cuda.manual_seed(self.MANUALSEED)
        
        self.optimizer = torch.optim.SGD(
            params= model.parameters(),
            lr=learning_rate
        )
        for epoch in range(self.epochs):

            model.train()

            y_logits = model(X_train).squeeze()
            y_preds = torch.round(torch.sigmoid(y_logits))

            loss = self.binary_loss(y_logits, y_train)
            acc = Utils.accuracy_fn(y_train, y_preds)

            self.optimizer.zero_grad()
            loss.backward()

            self.optimizer.step()

            model.eval()

            with torch.inference_mode():

                test_logits = model(X_test).squeeze()
                test_preds = torch.round(torch.sigmoid(test_logits))

                test_loss = self.binary_loss(test_logits, y_test)
                test_acc = Utils.accuracy_fn(y_test, test_preds)

            if epoch % 100 == 0:
                print(f"Epochs : {epoch} | Loss : {loss:.5f} , Acc: {acc:.2f}% | Test Loss : {test_loss:.5f}, Test Acc: {test_acc:.2f}%")

        return model

            
    def train_multiclass_model(self, model, X_train : torch.Tensor,
                               X_test: torch.Tensor,
                               y_train: torch.Tensor,
                               y_test: torch.Tensor,
                               learning_rate: int|float) -> nn.Module:
        torch.manual_seed(self.MANUALSEED)
        torch.manual_seed(self.MANUALSEED)

        self.optimizer = torch.optim.SGD(params= model.parameters(),
                                         lr= learning_rate)
        
        for epoch in range(self.epochs):
            model.train()

            y_logits = model(X_train)
            y_preds = torch.softmax(y_logits, dim = 1).argmax(dim= 1)

            loss = self.Multiclass_loss(y_logits, y_train)
            acc = Utils.accuracy_fn(y_train, y_preds)
            
            self.optimizer.zero_grad()

            loss.backward()

            self.optimizer.step()

            model.eval()

            with torch.inference_mode():

                test_logits = model(X_test)
                test_preds =  torch.softmax(test_logits, dim = 1).argmax(dim = 1)

                test_loss = self.Multiclass_loss(test_logits, y_test)
                test_acc = Utils.accuracy_fn(y_test, test_preds)

            if epoch % 100 == 0:

                print(f"Epochs : {epoch} | Loss : {loss:.5f} , Acc: {acc:.2f}% | Test Loss : {test_loss:.5f}, Test Acc: {test_acc:.2f}%")

        return model

        
