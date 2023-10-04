import json

"""
Parses model architecture & model optimizer data
from a pytorch model to a json file: "model_data.json"
"""
def parse_model_data(model_name: str, model_state_dict, model_optim):
    """
    
    """

    try:
        if model_name == "":
            raise TypeError("model_name param must be string")
        else:
            for var_name in model_optim.state_dict():
                model_architecture = model_state_dict.state_dict()

                model_architecture_json = {
                    key: model_architecture[key].size()
                    for key in model_architecture} 

                model_data = {
                    model_name: {
                        "model_architecure": model_architecture_json,
                        "model_optimizer": model_optim.state_dict()[var_name]
                    }
                }
                with open("model_data.json", "w") as write_file:
                    json.dump(model_data, write_file, indent=2)     

    except NameError:
        print("Params not defined")

    except:
        print("Something went wrong?")
