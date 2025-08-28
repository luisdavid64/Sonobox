
"""
    Plan for systematic analysis of models
    We can generate variations of our data with our api from an initial configuration.
"""

import argparse
from http import client
import os
import time
from pythonosc.udp_client import SimpleUDPClient
from transmitter import transmitter

def send_and_wait(client : SimpleUDPClient, address,value, wait_time=0.5):
    """Send a message and wait for a specified time."""
    client.send_message(address, value)
    time.sleep(wait_time)

def transmit_and_record(client: SimpleUDPClient, config_path, input_path, mode):
    send_and_wait(client, "/record/start", "/Users/luisreyes/Sonify/SonoBox/ex/record1.wav")
    transmitter(config_path, input_path, mode, client=client)
    send_and_wait(client, "/record/end", "")
    pass


def folder_generator(base, phase):
    folder_name = f"{base}/{phase}"
    os.makedirs(folder_name, exist_ok=True)
    return folder_name

PHASES_DYNAMICS = [
    "CHANGE_DAMPING",
    "CHANGE_STIFFNESS",
    "CHANGE MASSES"
    "CHANGE_FRICTION",
    "CHANGE_INTERACTIONS"
    
]

PHASES_GEOMETRY = [
    "CHANGE_RESOLUTION",
    "ADD_X",
    "ADD_Z",
    "MOVE_DRIVERS",
    "CHANGE_MASSES",
    "CHANGE_STIFFNESSES",
    "CHANGE_INTERACTIONS",
    "FRICTION"
]

def data_generator(config_path, input_path, mode):
    # Step one
    # Transmit with each modification

    # Set up the OCS client
    localhost_ip = "127.0.0.1"
    processing_port_1d_model = 12001
    client = SimpleUDPClient(localhost_ip, processing_port_1d_model)
    
    # Send transmission and record it
    transmit_and_record(client, config_path, input_path, mode)
    
    # Step one
    
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default="/Users/luisreyes/Sonify/SonoBox/config/config.json")           # positional argument
    parser.add_argument('--input_path', default="/Users/luisreyes/Sonify/BioSonix-V2/data/BioSonix/abstract-model/ordering/S-T-Br/1/data")           # positional argument
    parser.add_argument('--mode', choices=["fem", "us"], default="fem")           # positional argument
    args = parser.parse_args()
    data_generator(args.config_path, args.input_path, args.mode)