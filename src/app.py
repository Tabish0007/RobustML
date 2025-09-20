import streamlit as st
from PIL import Image, ImageDraw
import torch
import torchvision.transforms as transforms
import sys
sys.path.append(".")  # adds root to path
from models.mnist_model import SimpleCNN

# Load model (untrained for now)
model = SimpleCNN()
model.eval()

st.title("RobustML – MNIST Digit Recognizer")

# Drawing canvas
canvas_size = 280
img = Image.new("L", (canvas_size, canvas_size), color=0)
draw = ImageDraw.Draw(img)
st.write("Draw a digit below:")

# Let user upload an image or draw manually (simplified)
uploaded_file = st.file_uploader("Or upload a digit image", type=["png", "jpg"])
if uploaded_file:
    img = Image.open(uploaded_file).convert("L")

# Preprocess image
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])

tensor = transform(img).unsqueeze(0)  # Add batch dimension

# Predict button
if st.button("Predict"):
    with torch.no_grad():
        output = model(tensor)
        pred = output.argmax(dim=1).item()
        st.success(f"Predicted Digit: {pred}")
