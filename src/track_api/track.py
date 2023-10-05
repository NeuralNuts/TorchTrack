import json

global_error_msg = "Something went wrong"

class JsonModelData:
    def __init__(self, _model_name: str, _model_architecure, 
                 _model_optimizer, _model_training_data): 
        self._model_name = _model_name
        self._model_architecture = _model_architecure
        self._model_optimizer = _model_optimizer
        self._model_training_data = _model_training_data

    def set_model_state_dict(self):
            model_architecture_json = {
                    key: self._model_architecture[key].size()
                        for key in self._model_architecture
                    }
            return model_architecture_json

    def set_model_optimizer(self):
            for var_name in self._model_optimizer:
                model_optimizer = self._model_optimizer[var_name]
                return model_optimizer

    def set_model_training_data(self):
            training_data = [self._model_training_data]
            return training_data

    def save_model_data(self):
            if self._model_name == "":
                raise TypeError("model_name param must be string")
            else:
                json_model_data = JsonModelData(self._model_name, self._model_architecture, 
                                                self._model_optimizer, self._model_training_data)
            
                print(json_model_data._model_architecture)

                parse_model_data(json_model_data)

"""
Parses model architecture & model optimizer data
from a pytorch model to a json file: "model_data.json"
"""
def parse_model_data(json_model_data):
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
    
    model_data = {
            json_model_data._model_name: {
                "model_architecure": json_model_data._model_architecure,
                "model_optimizer": json_model_data._model_optimizer,
                "model_training_data": json_model_data._model_training_data,
            }
        }
        #model_data.update(model_data)

    with open("model_data.json", "w") as write_file:
        json.dump(model_data, write_file, indent=2)     

