import json
import os
import torch

def load_config(path: str) -> dict:
    """Load a mass-spring model config from a JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def save_config(config: dict, path: str):
    """Save a mass-spring model config to a JSON file."""
    with open(path, "w") as f:
        json.dump(config, f, indent=4)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def model_to_config(model) -> dict:
    """Convert a MassSpringModel instance to a config dict."""
    def tensor_to_list(tensor):
        return tensor.detach().cpu().tolist() if tensor is not None else []
    def float_or_list(value):
        if isinstance(value, (list, tuple)):
            return [float(v) for v in value]
        return float(value)
    def int_or_list(value):
        if isinstance(value, (list, tuple)):
            return [int(v) for v in value]
        return int(value)
    def mass_name(i, j, k):
        return f"m_{i}_{j}_{k}"
    geom = {
        "dx": int_or_list(model.dimX),
        "dy": int_or_list(model.dimY),
        "dz": int_or_list(model.dimZ),
        "distance": float_or_list(model.dist),
        "massesRadius": float_or_list(model.radius[0].item()),
        "interactionType": model.interactionType,
    }
    params = {
        "M": float_or_list(torch.mean(1/model.inv_mass).item()),
        "K": float_or_list(torch.mean(model.k).item()),
        "C": float_or_list(torch.mean(model.z).item()),
        "K1": float_or_list(torch.mean(model.k1).item()) if hasattr(model, 'k1') else None,
        "K2": float_or_list(torch.mean(model.k2).item()) if hasattr(model, 'k2') else None
    }
    sonification_set_up = {
        "drivers": [mass_name(*idx) for idx in tensor_to_list(model.drivers)],
        "listeners": [mass_name(*idx) for idx in tensor_to_list(model.listeners)],
    }
    config = {
        "geometry": geom,
        "parameters": params,
        "sonification_set_up": sonification_set_up,
        "model": "3D",
        "global_friction": float(model.fric.item()),
        "bounds": model.bounds,
    }
    return config
