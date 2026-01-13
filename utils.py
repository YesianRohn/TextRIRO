import os
import torch
import importlib
from omegaconf import OmegaConf
from src.utils import instantiate_from_config


def load_state_dict(path, location='cpu'):
    _, ext = os.path.splitext(path)
    if ext.lower() == ".safetensors":
        import safetensors.torch
        state_dict = safetensors.torch.load_file(path, device=location)
    else:
        state_dict = torch.load(path, map_location=location)
        if isinstance(state_dict, dict) and 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
    print(f'Loaded state_dict from [{path}]')
    return state_dict


def create_model(config_path):
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model).cpu()
    print(f'Loaded model config from [{config_path}]')
    return model

def get_obj_from_str(string, reload=False, invalidate_cache=True):
    module, cls = string.rsplit(".", 1)
    if invalidate_cache:
        importlib.invalidate_caches()
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)
def create_data(config):
    data_cls = get_obj_from_str(config.target)
    data = data_cls(data_config=config)
    return data
