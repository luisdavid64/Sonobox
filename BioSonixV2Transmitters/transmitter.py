## TRANSMITTER ADAPTED TO THE NEW IMPLEMENTATION OF BIOSONIX-V2

import argparse
from pythonosc.udp_client import SimpleUDPClient
import transmitter_utils
from transmitter_utils import Parameters, get_loaders, compute_forces_vec, merge_data
import time
import numpy as np


def transmitter(config_path, input_path=None, mode="us"):
    params = Parameters(config_path)
    inDataDir = input_path
    load_data, load_data_norm = get_loaders(mode)

    print("################################################################")
    print(f"PLAYING DATA FROM: {inDataDir}")
    print("################################################################")

    # Load the Data
    contributing_pixels = params.sonification_set_up["contributing_pixels"]
    stiffness_keys = params.sonification_set_up["stiffnessArray"]
    trial_params = params.parameters[params.surgery]
    stiffness_values = []
    for key in stiffness_keys:
        k_value = trial_params["K"][key]  # Use .get() to avoid crash on bad key
        if k_value is not None:
            stiffness_values.append(k_value)
    stiffness_values = np.asarray(stiffness_values)

    if params.use_norm_data:
        print("Loading normalized data ...")
        u_data = load_data_norm(contributing_pixels, inDataDir)
    else:
        print("Loading NOT normalized data ...")
        u_data = load_data(contributing_pixels, inDataDir)

    acoustic_scaling_factor = params.sonification_set_up["acoustic_scaling_factor"]
    u_data = merge_data(u_data)

    # INSERTION TIMING
    timestamps = u_data["time"]
    target_total_duration = timestamps.iloc[-1]
    num_iterations = len(timestamps)
    time_per_iteration = target_total_duration / num_iterations

    # COMPUTE EXCITATION FORCES
    forces = compute_forces_vec(u_data, stiffness_values)

    # UDP client setup
    localhost_ip = "127.0.0.1"
    processing_port_1d_model = 12001
    client = SimpleUDPClient(localhost_ip, processing_port_1d_model)

    # EXCITATION
    start_time = time.time()
    for i in range(num_iterations):

        # Apply scaling to forces
        scaled_forces = forces[i] * acoustic_scaling_factor
        transmitter_utils.send_message(client, '/multipleToggle', scaled_forces)

        # State transition messages
        if i == 0:
            transmitter_utils.send_message(client, '/msgTrigger', ["START"])
        if i == num_iterations - 1:
            transmitter_utils.send_message(client, '/msgTrigger', ["STOP"])
        # Time synchronization
        elapsed_time = time.time() - start_time
        expected_time = (i + 1) * time_per_iteration
        time.sleep(max(0, expected_time - elapsed_time))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default="/Users/luisreyes/Sonify/BioSonix-V2/SoundModels/PhysModel/config/config.json")           # positional argument
    parser.add_argument('--input_path', default="/Users/luisreyes/Sonify/BioSonix-V2/data/BioSonix/abstract-model/target-and-sensitive/2/to-sensitive/data")           # positional argument
    parser.add_argument('--mode', choices=["fem", "us"], default="fem")           # positional argument
    args = parser.parse_args()
    transmitter(args.config_path, args.input_path, args.mode)