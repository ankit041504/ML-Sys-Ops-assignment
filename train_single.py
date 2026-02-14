import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import SimpleNN
import time

device = "cpu"

transform = transforms.Compose([transforms.ToTensor()])

dataset = datasets.MNIST(".", download=True, transform=transform)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

model = SimpleNN().to(device)
opt = torch.optim.SGD(model.parameters(), lr=0.01)
loss_fn = torch.nn.CrossEntropyLoss()

start = time.time()

for epoch in range(3):
    for x, y in loader:
        x = x.view(x.size(0), -1).to(device)
        y = y.to(device)

        pred = model(x)
        loss = loss_fn(pred, y)

        opt.zero_grad()
        loss.backward()
        opt.step()

end = time.time()
print("Single Node Time:", end - start)
