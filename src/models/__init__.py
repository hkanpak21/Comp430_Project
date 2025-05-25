from .simple_dnn import SimpleDNN
from .simple_cnn import SimpleCNN

def get_model(model_name):
    """
    Returns a model instance by name.
    
    Args:
        model_name (str): Name of the model to instantiate.
        
    Returns:
        Model instance.
    """
    if model_name == "SimpleDNN":
        return SimpleDNN()
    elif model_name == "SimpleCNN":
        return SimpleCNN()
    else:
        raise ValueError(f"Unknown model: {model_name}") 