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
            # Simulate predictions (replace this with actual model predictions)
                # For demonstration, we're returning a random prediction
                sample_image = np.random.rand(3, 32, 32).astype(np.float32)
                sample_image_tensor = torch.tensor(sample_image).unsqueeze(0)

                with torch.no_grad():
                    prediction = model(sample_image_tensor)
                _, predicted_class = torch.max(prediction, 1)

                predicted_class = predicted_class.item()

                # Send the prediction as a JSON response
                prediction_response = json.dumps({'prediction': predicted_class})
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(prediction_response.encode())


        def GET_model(self):
            prediction_response = json.dumps({'model': model})
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(prediction_response.encode())

        def do_GET(self):
            if self.path == '/predict':
                PredictionHandler.GET_preds(self) 
            #if self.path == '/model':
                #PredictionHandler.GET_model(self)
            else:
                super().do_GET()

    return PredictionHandler
