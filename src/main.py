from numpy import mod
from track_api import track
import socketserver
import torch
from torchvision import transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F


def server_startup(Handler, PORT):
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print("Serving at port", PORT)
        httpd.serve_forever()

class SimpleCNN(nn.Module):
    def __init__(self):
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

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    ])


num_classes = 10  # CIFAR-10 has 10 classes

model = SimpleCNN()
track.set_model_state_dict(model.state_dict())
#model.load_state_dict(torch.load(r'saved_models/cifar10_model.pth'))
#model.eval()

optim_t = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
track.set_model_optimizer(optim_t.state_dict())

if __name__ == '__main__':
    print()
    #track.parse_model_data("Simple CNN", model, optim_t, "")
