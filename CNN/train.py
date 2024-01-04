import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from typing import Tuple

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
