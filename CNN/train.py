import torch
from torch import nn
from torchvision import datasets
from torch.utils.data import DataLoader
from typing import Tuple, Dict
from tqdm.auto import tqdm


class Train:
    def __init__(self):
        self.MANUAL_SEED = 42
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.manual_seed(self.MANUAL_SEED)
        torch.cuda.manual_seed(self.MANUAL_SEED)

    def train_step(self,
                   model: torch.nn.Module,
                   dataloader: torch.utils.data.DataLoader,
                   loss_fn: torch.nn.Module,
                   optimizer: torch.optim.Optimizer) -> Tuple[float, float]:
        """
        Perform a single training step.

        Args:
            model (torch.nn.Module): The PyTorch model.
            dataloader (torch.utils.data.DataLoader): DataLoader for training data.
            loss_fn (torch.nn.Module): The loss function.
            optimizer (torch.optim.Optimizer): The optimizer.

        Returns:
            Tuple[float, float]: Tuple containing training loss and training accuracy.
        """
        model.train()
        train_loss, train_acc = 0.0, 0.0
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(self.device), y.to(self.device)
            optimizer.zero_grad()
            y_pred_logits = model(X)
            loss = loss_fn(y_pred_logits, y)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

            y_preds_class = torch.argmax(torch.softmax(y_pred_logits, dim=1), dim=1)
            train_acc += (y_preds_class == y).sum().item() / len(y_preds_class)

        train_loss /= len(dataloader)
        train_acc /= len(dataloader)

        print(f"Train Loss: {train_loss:.5f} | Train Acc: {train_acc:.5f}")
        return train_loss, train_acc

    def test_step(self,
                  model: torch.nn.Module,
                  dataloader: torch.utils.data.DataLoader,
                  loss_fn: torch.nn.Module,
                  ) -> Tuple[float, float]:
        """
        Perform a single testing step.

        Args:
            model (torch.nn.Module): The PyTorch model.
            dataloader (torch.utils.data.DataLoader): DataLoader for testing data.
            loss_fn (torch.nn.Module): The loss function.

        Returns:
            Tuple[float, float]: Tuple containing testing loss and testing accuracy.
        """
        model.eval()
        test_loss, test_acc = 0.0, 0.0

        with torch.inference_mode():  # No need for inference_mode here
            for batch, (X, y) in enumerate(dataloader):
                X, y = X.to(self.device), y.to(self.device)
                y_preds_logits = model(X)
                loss = loss_fn(y_preds_logits, y)

                test_loss += loss.item()
                y_pred_classes = torch.argmax(torch.softmax(y_preds_logits, dim=1), dim=1)
                test_acc += (y_pred_classes == y).sum().item() / len(y)

            test_loss /= len(dataloader)
            test_acc /= len(dataloader)
            print(f"Test Loss: {test_loss:.5f} | Test Acc: {test_acc:.5f}")

        return test_loss, test_acc

    def train(self, model: torch.nn.Module,
              train_dataloader: torch.utils.data.DataLoader,
              test_dataloader: torch.utils.data.DataLoader,
              optimizer: torch.optim.Optimizer,
              loss_fn: torch.nn.Module = nn.CrossEntropyLoss,
              epochs: int = 5
              ) -> Dict:
        """
        Train the model.

        Args:
            model (torch.nn.Module): The PyTorch model.
            train_dataloader (torch.utils.data.DataLoader): DataLoader for training data.
            test_dataloader (torch.utils.data.DataLoader): DataLoader for testing data.
            optimizer (torch.optim.Optimizer): The optimizer.
            loss_fn (torch.nn.Module): The loss function.
            epochs (int): Number of training epochs.

        Returns:
            Dict: Dictionary containing training and testing results.
        """
        results = {'train_loss': [],
                   'train_acc': [],
                   'test_loss': [],
                   'test_acc': []}

        for epoch in tqdm(range(epochs)):
            train_loss, train_acc = self.train_step(model=model,
                                                    dataloader=train_dataloader,
                                                    loss_fn=loss_fn,
                                                    optimizer=optimizer)

            test_loss, test_acc = self.test_step(model=model,
                                                 dataloader=test_dataloader,
                                                 loss_fn=loss_fn)

            results['train_loss'].append(train_loss)
            results['test_loss'].append(test_loss)
            results['train_acc'].append(train_acc)
            results['test_acc'].append(test_acc)

        return results
