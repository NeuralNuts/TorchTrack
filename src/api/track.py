import http.server
import json
import numpy as np
import torch

# Handler to serve predictions for the /predict endpoint
def set_prediction(model):
    class PredictionHandler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            self.model = model
            super().__init__(*args, **kwargs)

        def GET_preds(self):
            print()

        def GET_model(self):
            print(model)

        def do_GET(self):
            if self.path == '/predict':
                PredictionHandler.GET_preds(self) 
            else:
                super().do_GET()

    return PredictionHandler
