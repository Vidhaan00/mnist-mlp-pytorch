import torch
from PIL import Image
from torchvision.transforms import v2

from model import Net

transform = v2.Compose([
    v2.ToImage(),
    v2.Resize((28, 28)),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize((0.5,), (0.5,))
])

classes = tuple(str(i) for i in range(10))

model = Net()

model.load_state_dict(
    torch.load("../saved_models/mnist_cnn.pt")
)

model.eval()

model = Net()

model.load_state_dict(
    torch.load("../saved_models/mnist_cnn.pt")
)

model.eval()

image = Image.open("../samples/five.png").convert("L")


image = transform(image)


image = image.unsqueeze(0)

with torch.no_grad():

    outputs = model(image)

    _, predicted = torch.max(outputs,1)


probabilities = torch.softmax(outputs,dim=1)


confidence = probabilities[0,predicted.item()]


print(f"Prediction : {classes[predicted.item()]}")
print(f"Confidence : {confidence*100:.2f}%")
