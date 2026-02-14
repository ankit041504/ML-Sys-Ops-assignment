import torch
import torch.multiprocessing as mp
import time

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from model import SimpleNN
from utils import flatten_images, calculate_accuracy


def get_dataset():
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )
    return dataset


def average_models(models):
    """
    Average model weights across workers
    """
    avg_model = models[0]

    for key in avg_model.state_dict().keys():
        avg_param = torch.stack(
            [m.state_dict()[key].float() for m in models], dim=0
        ).mean(dim=0)

        avg_model.state_dict()[key].copy_(avg_param)

    return avg_model


def train_worker(rank, world_size, return_dict):

    device = torch.device("cpu")

    dataset = get_dataset()

    # Split dataset manually across workers
    data_per_worker = len(dataset) // world_size
    start = rank * data_per_worker
    end = start + data_per_worker

    subset = Subset(dataset, range(start, end))
    loader = DataLoader(subset, batch_size=64, shuffle=True)

    model = SimpleNN().to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = torch.nn.CrossEntropyLoss()

    epochs = 3

    for epoch in range(epochs):

        total_loss = 0
        total_acc = 0
        batches = 0

        for images, labels in loader:

            images = flatten_images(images).to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = loss_fn(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = calculate_accuracy(outputs, labels)

            total_loss += loss.item()
            total_acc += acc
            batches += 1

        print(
            f"[Worker {rank}] Epoch {epoch+1} | "
            f"Loss {total_loss/batches:.4f} | "
            f"Acc {(total_acc/batches)*100:.2f}%"
        )

    return_dict[rank] = model.state_dict()


def main():

    world_size = 2

    manager = mp.Manager()
    return_dict = manager.dict()

    print("Running Simulated Distributed Training")

    start_time = time.time()

    processes = []

    for rank in range(world_size):
        p = mp.Process(
            target=train_worker,
            args=(rank, world_size, return_dict)
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    total_time = time.time() - start_time

    print(f"\nSimulated Distributed Training Time: {total_time:.2f} sec")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
