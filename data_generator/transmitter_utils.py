from typing import Callable, Tuple
import json
import os
import pandas as pd
import numpy as np

class Parameters:
    def __init__(self, filename):
        with open(filename, 'r') as f:
            entries = json.load(f)
        self.__dict__.update(entries)

def load_data(contributions, inDataDir, prefix="",suffix=""):
    """
    Contributions: points within the spatial domain from which we extract the temporal data (=displacements)
    Load displacement data for the specified contributions
    """
    data = []
    for roi in contributions:
        filename = f"{prefix}{roi}{suffix}.json"
        filepath = os.path.join(inDataDir, filename)
        data.append(pd.read_json(filepath))
    return data

def load_norm_data(contributions, inDataDir):
    return load_data(contributions, inDataDir, suffix="_norm")

def load_data_fem(contributions, inDataDir):
    return load_data(contributions, inDataDir, prefix="U_var_")

def load_norm_data_fem(contributions, inDataDir):
    return load_data(contributions, inDataDir, prefix="U_var_", suffix="_normalized")

def merge_data(dfs: list[pd.DataFrame]):
    return pd.concat([df.set_index("time") for df in dfs], axis=1).reset_index()

def compute_forces(data_list, stiffness):
    """
    Compute forces the excitation forces: F = U x K
    :param data_list: list with all the extracted data from the contributing nodes / pixels (spatial key points)
    :param stiffness: list with the stiffness values corresponding to those nodes / pixels
    """
    forces = []
    for data, k in zip(data_list, stiffness):
        displacement = data.iloc[:, 1].to_list()
        forces.append([u * k for u in displacement])
    return forces

def compute_forces_vec(data, stiffness) -> np.ndarray:
    """
    Compute forces the excitation forces: F = U x K
    :param data_list: list with all the extracted data from the contributing nodes / pixels (spatial key points)
    :param stiffness: list with the stiffness values corresponding to those nodes / pixels
    """
    if isinstance(data, list):
        data = np.asarray(data)
    if isinstance(data,pd.DataFrame):
        data = data.to_numpy()
    if isinstance(stiffness, list):
        stiffness = np.asarray(stiffness)
    data = data[:,1:]
    forces = data * stiffness
    return forces

def send_message(client, address, message):
    """
    Send a message using the UDP client.
    """
    print(f"Sending {message} to {address}")
    client.send_message(address, message)

def get_loaders(mode: str) -> Tuple[Callable, Callable]:
    if mode == "us":
        return load_data, load_norm_data
    elif mode == "fem":
        return load_data_fem, load_norm_data_fem
    else:
        raise ValueError("Invalid mode. Choose 'us' or 'fem'.")