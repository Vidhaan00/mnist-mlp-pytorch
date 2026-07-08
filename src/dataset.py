import torch
import torchvision
from torchvision.transforms import v2

transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize((0.5,), (0.5,))
])


def get_dataloaders(batch_size=4):

    trainset = torchvision.datasets.MNIST(
    root="../data",
    train=True,
    download=True,
    transform=transform)

    trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=batch_size,
    shuffle=True)

    testset = torchvision.datasets.MNIST(
    root="../data",
    train=False,
    download=True,
    transform=transform)

    testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=batch_size,
    shuffle=False)

    classes = ('0','1','2','3','4','5','6','7','8','9')


    return trainloader, testloader, classes
