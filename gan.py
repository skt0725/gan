import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "0, 1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('Device:', device)
print('Count of using GPUs:', torch.cuda.device_count())
print('Current cuda device:', torch.cuda.current_device())

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [1])
])
data_root = './data'
train_data = datasets.MNIST(data_root, transform=transform, train=True, download=True)

dataloader = DataLoader(train_data, batch_size = 128, shuffle=True, num_workers = 4)

# example data
train_x, train_y = next(iter(dataloader))
img = train_x[0].squeeze()
label = train_y[0]
plt.imshow(img, cmap="gray")
plt.title = label
plt.show()

# https://machinelearningmastery.com/how-to-train-stable-generative-adversarial-networks/
# Leaky ReLU for discriminator, relu&tanh for generator
class Generator(nn.Module):
    def __init__(self, latent_size, image_size):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(latent_size, 128),
            nn.ReLU(inplace = True),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, image_size),
            nn.Tanh()
        ) 
    def forward(self, z):
        output = self.layer(z)
        return output
class Discriminator(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        output = self.layer(x)
        return output

 
