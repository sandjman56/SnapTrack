import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn, optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

"""
This script uses 32x32 pixel images from the Alphabets And Number Images data set (https://www.kaggle.com/datasets/kushagra3204/alphabets-and-number-images?resource=download)

I only used the Validation folder from this data set and split it into training and testing data for an initial model as the entire data set is very large containing
around 2 million images total

I then save the trained model parameters uner "model_weights.pth" and save the loss plot of the model under "CNN_loss_curve.png"
"""


class CNN(nn.Module):
    def __init__(self, num_classes=len(train_dataset.classes)):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(64*8*8, 120)
        self.fc2 = nn.Linear(120, 80)
        self.fc3 = nn.Linear(80, num_classes)

        self.act = nn.functional.relu

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.pool(x)

        x = self.act(self.conv2(x))
        x = self.pool(x)

        x = x.view(-1, 64*8*8)
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.fc3(x)

        return x

def main():

    transform = transforms.Compose([
        transforms.Resize((32, 32)), # ensure images are 32x32  
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load training and testing data
    train_dataset = datasets.ImageFolder(root='data/data/Validation', transform=transform)
    test_dataset = datasets.ImageFolder(root='data/data/validation_subset', transform=transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU detected. Using CUDA.")
    else:
        device = torch.device("cpu")
        print("No GPU detected. Using CPU.")

    model = CNN().to(device)

    lr = 0.005
    epochs = 15

    loss_fun = nn.CrossEntropyLoss()
    opt = optim.Adam(params=model.parameters(), lr=lr)

    loss_curve = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            out = model(images)
            loss = loss_fun(out, labels)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()

        loss_curve.append(total_loss / len(train_dataset))

        if epoch % int(epochs / epochs) == 0:
            print(f"Epoch: {epoch+1}/{epochs}    Training Loss: {total_loss / len(train_dataset)}")

    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Test Accuracy: {100 * correct / total:.2f}%")

    torch.save(model.state_dict(), "model_weights.pth")
        
    plt.plot(loss_curve)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig("CNN_loss_curve.png")

if __name__ == "__main__":
    main()
