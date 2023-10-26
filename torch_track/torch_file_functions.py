import json
from textwrap import indent

def write_model_data_to_json(model_data):
    with open("json_data/model_data.json", "w+") as write_file:
        json.dump(model_data, write_file, indent=2)

def write_training_data_to_json(training_data):
    with open("json_data/training_data.json", "w+") as write_file:
        json.dump(training_data, write_file, indent=2)   

def read_json_training_data():
    with open("json_data/training_data.json", "r+") as read_file:
        json_object = json.load(read_file)
        json_forrmated = json.dumps(json_object, indent=2)
        print(json_forrmated)



