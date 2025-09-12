import numpy as np
import torch

def event_dict_to_tensor(event_dict, device=None, sample_rate=16000):
    # Conver event_dict to tensor of rows of forces N,3
    # N = event_dict.keys().max() * sample_rate
    if len(event_dict) == 0:
        return None
    max_step = max(event_dict.keys())
    total_steps = int(max_step * sample_rate)
    forces = torch.zeros((total_steps, 3), device=device)
    for step, force in event_dict.items():
        step_idx = max(0,int(step * sample_rate) - 1)
        forces[step_idx] += torch.tensor(force, device=device)
    return forces

def event_dict_seconds_to_samples(event_dict, sample_rate=16000):
    # Convert event_dict keys from seconds to samples (int)
    new_dict = {}
    for k, v in event_dict.items():
        k_samples = max(0, int(k * sample_rate) - 1)
        new_dict[k_samples] = v
    return new_dict

if __name__ == "__main__":
    # Example usage
    raw_events = {0.0: (1, 0, 0), 0.5: (0, 1, 0), 1.0: (0, 0, 1)}
    event_tensor = event_dict_to_tensor(raw_events, device='cpu', sample_rate=16000)
    print(event_tensor)
    sample_events = event_dict_seconds_to_samples(raw_events, sample_rate=16000)
    print(sample_events)