import torch

def predict(model, tensor):
    model.eval()
    with torch.no_grad():
        output = model(tensor)
        pred = output.argmax(dim=1).item()
    return pred
