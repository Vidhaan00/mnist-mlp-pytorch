import torch
import torch.nn as nn
import torch.optim as optim

from dataset import get_dataloaders
from model import Net


BATCH_SIZE = 4
LEARNING_RATE = 0.001
EPOCHS = 5

def main():

    trainloader, testloader, classes = get_dataloaders(
    batch_size=BATCH_SIZE)

    model= Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
    model.parameters(),
    lr=LEARNING_RATE,
    momentum=0.9)

    for epoch in range(EPOCHS):

        running_loss = 0.0

        for images, labels in trainloader:

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        average_loss = running_loss / len(trainloader)

        print(f"Epoch {epoch+1}/{EPOCHS} "
              f"Loss: {average_loss:.4f}")
    torch.save(
    model.state_dict(),
    "../saved_models/mnist_cnn.pt")



if __name__ == "__main__":
    main()
