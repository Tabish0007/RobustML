import torch
import torch.nn as nn

def fgsm_attack(model, images, labels, eps=0.03):
    images.requires_grad = True
    outputs = model(images)
    loss = nn.CrossEntropyLoss()(outputs, labels)
    model.zero_grad()
    loss.backward()
    adv_images = images + eps * images.grad.sign()
    adv_images = torch.clamp(adv_images, 0, 1)
    return adv_images
