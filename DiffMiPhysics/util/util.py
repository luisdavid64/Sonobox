import numpy as np
import torch
import json
import os

def event_dict_to_tensor(event_dict, device=None, sample_rate=16000):
    # Conver event_dict to tensor of rows of forces N,3
    # N = event_dict.keys().max() * sample_rate
    if len(event_dict) == 0:
        return None
    max_step = max(event_dict.keys())
    total_steps = int(max_step * sample_rate)
    forces = torch.zeros((total_steps, 3), device=device)
    for step, force in event_dict.items():
        step_idx = max(0,int(step * sample_rate))
        if step_idx == total_steps:
            step_idx = total_steps - 1
        forces[step_idx] += torch.tensor(force, device=device)
    return forces

def event_dict_seconds_to_samples(event_dict, sample_rate=16000):
    # Convert event_dict keys from seconds to samples (int)
    new_dict = {}
    for k, v in event_dict.items():
        k_samples = max(0, int(k * sample_rate))
        new_dict[k_samples] = v
    return new_dict

def event_dict_samples_to_seconds(event_dict, sample_rate=16000):
    # Convert event_dict keys from samples (int) to seconds (float)
    new_dict = {}
    for k, v in event_dict.items():
        k_seconds = k / sample_rate
        new_dict[k_seconds] = v
    return new_dict

def save_event_to_json(event_dict, path):
    with open(path, "w") as f:
        json.dump(event_dict, f, indent=4)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    
def load_event_from_json(path):
    with open(path, "r") as f:
        return json.load(f)

if __name__ == "__main__":
    # Example usage
    raw_events = {0.0: (1, 0, 0), 0.5: (0, 1, 0), 1.0: (0, 0, 1)}
    event_tensor = event_dict_to_tensor(raw_events, device='cpu', sample_rate=16000)
    print(event_tensor)
    sample_events = event_dict_seconds_to_samples(raw_events, sample_rate=16000)
    print(sample_events)
    seconds_events = event_dict_samples_to_seconds(sample_events, sample_rate=16000)
    print(seconds_events)