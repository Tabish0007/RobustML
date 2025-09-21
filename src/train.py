import torch
import torch.nn as nn
import torch.optim as optim
from src.data.load_data import get_data
from src.models.robust_model import RobustCNN
from src.attacks.fgsm import fgsm_attack

# ---------------------------
# Setup device
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# Load data
# ---------------------------
train_loader, test_loader = get_data(batch_size=64)

# ---------------------------
# Initialize model, loss, optimizer
# ---------------------------
model = RobustCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ---------------------------
# Training loop with adversarial training
# ---------------------------
epochs = 5  # increase for better accuracy
epsilon = 0.1  # strength of FGSM attack

for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Standard training
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Adversarial training
        adv_images = fgsm_attack(model, images, labels, epsilon)
        optimizer.zero_grad()
        adv_outputs = model(adv_images)
        adv_loss = criterion(adv_outputs, labels)
        adv_loss.backward()
        optimizer.step()

        running_loss += (loss.item() + adv_loss.item()) / 2

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")

# ---------------------------
# Save model
# ---------------------------
torch.save(model.state_dict(), "src/models/robust_model.pth")
print("Adversarially trained model saved to src/models/robust_model.pth")
