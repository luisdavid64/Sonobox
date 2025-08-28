
"""
    Plan for systematic analysis of models
    We can generate variations of our data with our api from an initial configuration.
"""

import argparse
import time
from pythonosc.udp_client import SimpleUDPClient
from transmitter import transmitter

def send_and_wait(client : SimpleUDPClient, address,value, wait_time=0.5):
    """Send a message and wait for a specified time."""
    client.send_message(address, value)
    time.sleep(wait_time)


def data_generator(config_path, input_path, mode):
    # Step one
    # Transmit with each modification

    localhost_ip = "127.0.0.1"
    processing_port_1d_model = 12001
    client = SimpleUDPClient(localhost_ip, processing_port_1d_model)
    send_and_wait(client, "/record/start", "/Users/luisreyes/Sonify/SonoBox/ex/record1.wav")
    transmitter(config_path, input_path, mode, client=client)
    send_and_wait(client, "/record/end", "")
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default="/Users/luisreyes/Sonify/SonoBox/config/config.json")           # positional argument
    parser.add_argument('--input_path', default="/Users/luisreyes/Sonify/BioSonix-V2/data/BioSonix/abstract-model/ordering/S-T-Br/1/data")           # positional argument
    parser.add_argument('--mode', choices=["fem", "us"], default="fem")           # positional argument
    args = parser.parse_args()
    data_generator(args.config_path, args.input_path, args.mode)