import json
from . import torch_track_api
from . import torch_file_functions
import requests

global_error_msg = "Something went wrong!"

"""
Sets the model data state that is parsed in by the user for further functions
"""
class TorchModelData:
    def __init__(self, _model_name: str, _model_architecure, 
                _model_optimizer): 
        self._model_name = _model_name
        self._model_architecture = _model_architecure
        self._model_optimizer = _model_optimizer

    """
    Saves model data to the TorchTrackApp api
    
    options: "api" -> saves to the api, "file" -> saves to a json file
    
    example ->
    var_name.save_training_data("option")
    """
    def save_model_data(self, option: str):
            if self._model_name == "":
                raise ValueError("model_name param and file_path cannot be null")
            else:
                json_model_data = TorchModelData(self._model_name, self._model_architecture, 
                                                self._model_optimizer)
                parse_model_data(json_model_data, option)

    """
    Prints saved model data to the terminal 
    """
    def print_saved_model_data(self) -> bool:
        print(f"---------------------------------")
        print(f"\n--- Model Name: {self._model_name} ---")
        print(f"\n--- Model Architecture: {self._model_architecture} ---")
        print(f"\n--- Model Optimizer: {self._model_optimizer} ---")
        print(f"---------------------------------")

        return True

"""
Sets the training data state that is parsed by the user for further functions
"""
class TorchTrainingData:
    def __init__(self, _model_training_data): 
        self._model_training_data = _model_training_data

    """
    Saves training data to the TorchTrackApp api
    options: "api" -> saves to the api, "file" -> saves to a json file
    example ->
    var_name.save_training_data("option")
    """
    def save_training_data(self, option: str):
                json_model_data = TorchTrainingData(self._model_training_data)

                parse_training_data(json_model_data, option)

    """
    Prints saved training data to the terminal
    """
    def print_saved_training_data(self) -> bool:
        print(f"---------------------------------")
        print(f"\n--- Model Traing Data: {self._model_training_data} ---")
        print(f"---------------------------------")

        return True

    def print_json_training_data(self):
        torch_file_functions.read_json_training_data()

"""
Parses model training data
from a pytorch model to this json file: "model_data.json"
"""
def parse_training_data(json_training_data, option: str):
    json_string_model_train = json.dumps(json_training_data._model_training_data)
    training_data = {
        "model_training_data": str(json_string_model_train),
    }

    if(option == "api"):
        print("Saving training data to api")
        torch_track_api.post_training_data(training_data)
    elif(option == "file"):
        print("Saving training data to file")
        torch_file_functions.write_training_data_to_json(training_data)
    else:
        print("Please pick a valid option: api, file")

"""
Parses model architecture & model op
from a pytorch model to this json file: "model_data.json"
"""
def parse_model_data(json_model_data, option: str):
    json_string_model_opti = json.dumps(json_model_data._model_optimizer)
    model_data = {
                "model_name": json_model_data._model_name,
                "model_architecure": str(json_model_data._model_architecture),
                "model_optimizer": str(json_string_model_opti),
    }

    if(option == "api"):
        print("Saving model data to api")
        torch_track_api.post_model_data(model_data)
    elif(option == "file"):
        print("Saving model data to file")
        torch_file_functions.write_model_data_to_json(model_data)
    else:
        print("Please pick a valid option: api, file")
