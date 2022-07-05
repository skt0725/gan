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

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        # self.layer1 = 
    def forward(self, x):
        output = self.
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(

        )
 
