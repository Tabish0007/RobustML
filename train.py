import torch
import torch.optim as optim
import torch.nn as nn
from datasets.load_cifar10 import load_data
from models.simple_cnn import SimpleCNN

trainloader, testloader = load_data()
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SimpleCNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Quick 1-epoch training to check setup
for epoch in range(1):
    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

print("Training done for 1 epoch")
