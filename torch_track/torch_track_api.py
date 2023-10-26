from os import error
import requests

def post_model_data(model_data):
    try:
        post_endpoint = "https://torchtrackapp.azurewebsites.net/api/TorchTrack/PostModelData"
        requests.post(post_endpoint, json = model_data)
    except:
        return requests.ConnectionError() 

def post_training_data(training_data):
    try: 
        post_endpoint = "https://torchtrackapp.azurewebsites.net/api/TorchTrack/PostTrainginData"
        requests.post(post_endpoint, json = training_data)
    except:
        return requests.ConnectionError()


