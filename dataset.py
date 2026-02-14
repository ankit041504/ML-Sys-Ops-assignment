from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_mnist_dataset(batch_size=64):
    """
    Returns MNIST train loader
    """

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )

    return dataset, loader
