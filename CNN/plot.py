import matplotlib.pyplot as plt
from typing import Dict, List


def plot_loss_curves(results: Dict[str, List[float]]):
    """ Plot the training curves of the results dictionary"""

    loss = results["train_loss"]
    test_loss = results["test_loss"]

    acc = results["train_acc"]
    test_acc = results["test_acc"]

    # figure ou how many epochs where there

    epochs = range(len(results["train_loss"]))

    # set up the plot
    plt.figure(figsize=(15, 7))

    # plot the loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label="train_loss")
    plt.plot(epochs, test_loss, label="test_loss")
    plt.title("Loss")
    plt.xlabel("epochs")
    plt.legend()

    # plot the accuracy

    plt.subplot(1, 2, 2)
    plt.plot(epochs, acc, label="train_acc")
    plt.plot(epochs, test_acc, label="test_acc")
    plt.title("Accuracy")
    plt.xlabel("epochs")
    plt.legend()
