from enum import global_str
import json
from struct import Struct

global_str = "Something went wrong"

class JsonModelData:
    def __init__(self, model_name, model_architecure, 
                 model_optimizer, model_training_data): 
        self.model_name = model_name
        self.model_architecture = model_architecure
        self.model_optimizer = model_optimizer
        self.model_training_data = model_training_data

def set_model_state_dict(model_architecture):
    try:
        model_architecture_json = {
                key: model_architecture[key].size()
                for key in model_architecture
                }
        parse_model_data(model_architecture_json)
    except:
        print(global_str)

def set_model_optimizer(model_optimizer):
    try:
        for var_name in model_optimizer:
            model_optimizer.state_dict()[var_name]
        parse_model_data(*model_optimizer)
    except:
        print(global_str)

def set_model_training_data(epoch, loss, step, *args, **kwargs):
    try:
        training_data = {epoch, loss, step, args}
        return training_data
    except:
        print(global_str)

def save_model_data(model_name: str):
    try:
        if model_name == "":
            raise TypeError("model_name param must be string")
        else:
            parse_model_data(model_name)
    except:
        print(global_str)


"""
Parses model architecture & model optimizer data
from a pytorch model to a json file: "model_data.json"
"""
def parse_model_data(*args, **kwargs):
    """
    Example code ->

    class MyModel(nn.Module):
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

    model = MyModel()
    optim = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    track.parse_model_data("Simple CNN", model, optim)            
    """
    try:
        model_name = kwargs.get("model_name", None)
        model_architecure = kwargs.get("model_architecure", None)
        model_optimizer = kwargs.get("model_optimizer", None)
        model_training_data = kwargs.get("model_training_data", None)

        model_data = {
            model_name: {
                "model_architecure": model_architecure,
                "model_optimizer": model_optimizer,
                "model_training_data": model_training_data,
            }
        }
        model_data.update(model_data)

        with open("model_data.json", "w") as write_file:
            json.dump(model_data, write_file, indent=2)     

    except NameError:
        print("Params not defined")
    except:
        print("Something went wrong?")
