from .simple_cnn import SimpleCNN
from .simple_dnn import SimpleDNN

def get_model(name="SimpleCNN", **kwargs):
    if name.lower() == "simplecnn":
        print(f"Initializing SimpleCNN model.")
        return SimpleCNN(**kwargs)
    elif name.lower() == "simplednn":
        print(f"Initializing SimpleDNN model.")
        return SimpleDNN(**kwargs)
    else:
        raise ValueError(f"Model {name} not recognized. Available: SimpleCNN, SimpleDNN") 