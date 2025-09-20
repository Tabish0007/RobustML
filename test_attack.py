import torch
from datasets.load_cifar10 import load_data
from models.simple_cnn import SimpleCNN
from attacks.fgsm import fgsm_attack

trainloader, testloader = load_data()
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SimpleCNN().to(device)
model.eval()

images, labels = next(iter(testloader))
images, labels = images.to(device), labels.to(device)

adv_images = fgsm_attack(model, images, labels, eps=0.03)
print("Adversarial images created:", adv_images.shape)
