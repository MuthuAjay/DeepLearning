# %%
import torch
from torch import nn
from create_data import DataSet
from matplotlib import pyplot as plt


class LinearRegressionModel(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.weights = nn.Parameter(torch.randn(1,
                                                dtype=torch.float,
                                                requires_grad=True))
        self.bias = nn.Parameter(torch.randn(1,
                                             dtype=torch.float,
                                             requires_grad=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias


def plot_predictions(train_data,
                     train_label,
                     test_data,
                     test_label,
                     predictions=None):
    plt.figure(figsize=(10, 7))

    plt.scatter(train_data, train_label, c='b', s=4, label="Training Data")

    plt.scatter(test_data, test_label, c='g', s=4, label="Test Data")

    if predictions is not None:
        plt.scatter(test_data, predictions, c='r', s=4, label="Predictions")

    plt.legend(prop={"size": 14})


def plot_metrics(epoch_count, loss_values, test_loss_values):
    plt.plot(epoch_count, torch.tensor(loss_values).numpy(), label="Train Loss")
    plt.plot(epoch_count, torch.tensor(test_loss_values).numpy(), label="Test Loss")
    plt.title("training and test loss curves")
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.legend()


if __name__ == '__main__':
    data = DataSet(weights=2, bias=1)
    x_train, x_test, y_train, y_test = data.create_1D_data_set(1, 20, 0.02)

    torch.manual_seed(42)
    model = LinearRegressionModel()

    from train_models import Train

    train_model = Train(model.parameters())
    train_model.train_data = x_train
    train_model.test_data = x_test
    train_model.train_label = y_train
    train_model.test_label = y_test

    model_up, epochs, loss, test_loss = train_model.train_model(model, epochs=15)

    plot_metrics(epochs, loss, test_loss)

    with torch.inference_mode():
        y_predS_new = model_up(x_test)

    plot_predictions(x_train, y_train, x_test, y_test, predictions=y_predS_new)

# %%
