import numpy as np
from PIL import Image
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 16 * 16 * 16)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


num_classes = 10  # CIFAR-10 has 10 classes

model = SimpleCNN(num_classes)
model.load_state_dict(torch.load('cifar10_model.pth'))
model.eval()

# Transformation for the input image
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])
