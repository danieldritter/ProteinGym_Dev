
available_models = {}

def register_model(model_name):
    """
    Register a model and add it to the available_models dictionary.

    Args:
        model: The model to be registered.

    Returns:
        The registered model.

    Raises:
        AssertionError: If the model is already registered.
    """
    assert model_name not in available_models, f"{model_name} already registered"
    def decorator(model):
        available_models[model_name] = model
        return model
    return decorator

