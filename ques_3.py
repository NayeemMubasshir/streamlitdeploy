#setup and dataset

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

transform = transforms.ToTensor()
train_dataset = MNIST(root="./data", train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)


#model architecture

class DigitGeneratorNet(nn.Module):
    def __init__(self, latent_dim=100, num_classes=10):
        super(DigitGeneratorNet, self).__init__()
        self.label_embedding = nn.Embedding(num_classes, num_classes)

        self.model = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 28 * 28),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        c = self.label_embedding(labels)
        x = torch.cat([noise, c], 1)
        img = self.model(x)
        return img.view(img.size(0), 1, 28, 28)


#training script

model = DigitGeneratorNet()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0002)

for epoch in range(10):  # Or more depending on quality
    for i in range(len(train_loader)):
        labels = torch.randint(0, 10, (64,))
        noise = torch.randn(64, 100)
        fake_images = model(noise, labels)
        
        target = torch.ones_like(fake_images)  # Simulate ideal images for now
        loss = criterion(fake_images, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

torch.save(model.state_dict(), "digit_generator.pth")



#app

import streamlit as st
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from model import DigitGeneratorNet  # move model class to model.py

# Load model
model = DigitGeneratorNet()
model.load_state_dict(torch.load("digit_generator.pth", map_location="cpu"))
model.eval()

st.title("Handwritten Digit Generator")
digit = st.number_input("Select a digit (0–9):", min_value=0, max_value=9, step=1)

if st.button("Generate Images"):
    fig, axs = plt.subplots(1, 5, figsize=(15, 3))
    for i in range(5):
        noise = torch.randn(1, 100)
        label = torch.tensor([digit])
        with torch.no_grad():
            img = model(noise, label).squeeze().numpy()
        axs[i].imshow(img, cmap="gray")
        axs[i].axis("off")
    st.pyplot(fig)
