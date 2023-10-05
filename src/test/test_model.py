"""misc{pascal-voc-2007,
        author = "Everingham, M. and Van~Gool, L. and Williams, C. K. I. and Winn, J. and Zisserman, A.",
        title = "The {PASCAL} {V}isual {O}bject {C}lasses {C}hallenge 2007 {(VOC2007)} {R}esults",
        howpublished = "http://www.pascal-network.org/challenges/VOC/voc2007/workshop/index.html"}"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from rich import print

from torch.utils.data import DataLoader, dataset

transform = transforms.Compose([
    transforms.ToTensor(),  # Convert the image to PyTorch Tensor
])

cifar_train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
cifar_val_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

batch_size = 5
cifar_train_loader = DataLoader(cifar_train_dataset, batch_size=batch_size, shuffle=True)
cifar_val_loader = DataLoader(cifar_val_dataset, batch_size=batch_size, shuffle=False)

# Define a simple CNN model
class ExCNN(nn.Module):
    def __init__(self, num_classes):
        super(ExCNN, self).__init__()
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

# Parameters
num_classes = 10  # CIFAR-10 has 10 classes
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)

# Load CIFAR-10 dataset
transform = transforms.Compose([
    transforms.ToTensor(),
])
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Initialize the model, loss function, and optimizer
model = ExCNN(num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 5

for epoch in range(num_epochs):
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)  # Move data to GPU
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

# Save the trained model
torch.save(model.state_dict(), 'cifar10_model.pth')

# Load the trained model
model = ExCNN(num_classes)
model.load_state_dict(torch.load('cifar10_model.pth'))
model.eval()  # Set the model to evaluation mode

# Load CIFAR-10 dataset
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Use the test set for demonstration
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Function to display a sample of images with predictions
def display_sample_predictions(model, num_samples=5):
    model.eval()
    samples = np.random.randint(0, len(test_dataset), num_samples)
    
    for idx in samples:
        image, label = test_dataset[idx]
        image = image.unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            prediction = model(image)
        
        _, predicted_class = torch.max(prediction, 1)
        predicted_class = predicted_class.item()
        
        plt.imshow(np.transpose(image.squeeze(), (1, 2, 0)))
        plt.title(f'Predicted: {predicted_class}')
        plt.show()

# Display sample predictions
num_samples = 5
display_sample_predictions(model, num_samples)

def main():
    print("Main 0/n")

main()
