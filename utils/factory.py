import os
import yaml
from pathlib import Path
from huggingface_hub import hf_hub_download
from ..decoder.pretrained import WavTokenizer


def load_model_dict_from_yaml(yaml_file: str):
    """Load the model dictionary from a YAML file."""
    with open(yaml_file, 'r') as file:
        model_dict = yaml.safe_load(file)
    return model_dict



def get_pretrained_model(name: str, model_config_path: Path = Path(os.path.dirname(os.path.abspath(__file__)))/'../configs/pretrained.yaml'):
    """
    Load a pretrained model by name.
    
    Args:
        name (str): The name of the pretrained model to load.
        model_config_path (str): Path to the YAML file containing model dictionary.

    Returns:
        model: The loaded WavTokenizer model.
    """
    # Load model dictionary from YAML file
    model_dict = load_model_dict_from_yaml(model_config_path)

    # Verify if the model name exists in the loaded dictionary
    if name not in model_dict['MODELS']:
        raise ValueError(f"Model name '{name}' not found in the model dictionary.")

    # Get config and checkpoint paths from YAML
    config_file = model_dict['MODELS'][name]["config"]
    checkpoint_url = model_dict['MODELS'][name]["checkpoint"]

    # Load the YAML configuration file
    config_path = os.path.join(os.path.abspath(os.path.dirname(model_config_path)), config_file)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file '{config_path}' not found.")

    # Download the checkpoint from Hugging Face hub
    model_path = hf_hub_download(os.path.dirname(checkpoint_url), filename=os.path.basename(checkpoint_url))
    print('Pretrained model will be loaded from:', model_path)
    # Load the model using WavTokenizer
    model = WavTokenizer.from_pretrained0802(config_path, model_path)

    return model
