import torch

from dataset import get_dataloaders
from model import Net


BATCH_SIZE = 4


def main():

    _, testloader, classes = get_dataloaders(
    batch_size=BATCH_SIZE)

    model = Net()
    model.load_state_dict(
    torch.load("../saved_models/mnist_cnn.pt"))

    model.eval()
    with torch.no_grad():

        correct = 0
        total = 0

        correct_pred = {c:0 for c in classes}
        total_pred = {c:0 for c in classes}

        for images, labels in testloader:

            outputs = model(images)
            _, predicted = torch.max(outputs,1)
            total += labels.size(0)

            correct += (predicted == labels).sum().item()

            for label, prediction in zip(labels,predicted):

                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1


        accuracy = 100 * correct / total
        print(f"Test Accuracy: {accuracy:.2f}%")

    for classname in classes:
        accuracy = (100* correct_pred[classname]/ total_pred[classname])
        print(f"{classname}: {accuracy:.2f}%")

