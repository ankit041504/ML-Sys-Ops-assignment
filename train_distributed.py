import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms
from model import SimpleNN
import time

def setup(rank, world_size):
    dist.init_process_group(
        backend="gloo",
        init_method="tcp://127.0.0.1:12355",
        rank=rank,
        world_size=world_size
    )

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size):
    setup(rank, world_size)

    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST(".", download=True, transform=transform)

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    loader = DataLoader(dataset, batch_size=64, sampler=sampler)

    model = SimpleNN()
    model = torch.nn.parallel.DistributedDataParallel(model)

    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = torch.nn.CrossEntropyLoss()

    start = time.time()

    for epoch in range(3):
        sampler.set_epoch(epoch)
        for x, y in loader:
            x = x.view(x.size(0), -1)
            pred = model(x)
            loss = loss_fn(pred, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

    end = time.time()

    if rank == 0:
        print("Distributed Training Time:", end - start)

    cleanup()

def main():
    world_size = 2
    mp.spawn(train, args=(world_size,), nprocs=world_size)

if __name__ == "__main__":
    main()
