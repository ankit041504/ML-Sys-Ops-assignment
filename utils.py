import torch


def calculate_accuracy(outputs, labels):
    """
    Calculate classification accuracy
    """
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == labels).sum().item()
    total = labels.size(0)

    return correct / total


def flatten_images(images):
    """
    Flatten MNIST images from 28x28 → 784
    """
    return images.view(images.size(0), -1)
