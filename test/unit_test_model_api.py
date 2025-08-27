"""Unit test for the model controller API framework."""

import argparse
from pythonosc.udp_client import SimpleUDPClient
import time
import numpy as np

def send_and_wait(client : SimpleUDPClient, address,value, wait_time=0.5):
    """Send a message and wait for a specified time."""
    client.send_message(address, value)
    time.sleep(wait_time)

def test(config_path, input_path=None, mode="us"):

    # UDP client setup
    localhost_ip = "127.0.0.1"
    processing_port_1d_model = 12001
    client = SimpleUDPClient(localhost_ip, processing_port_1d_model)

    # Test sending messages to the model controller API
    print("Testing: Increase dimX")
    send_and_wait(client, '/dim/x', +1)
    print("Testing: Decrease dimX")
    send_and_wait(client, '/dim/x', -1)
    print("Testing: Increase dimZ")
    send_and_wait(client, '/dim/z', +1)
    print("Testing: Decrease dimZ")
    send_and_wait(client, '/dim/z', -1)
    print("Testing: Cycle Interactions")
    send_and_wait(client, '/dim/x', +1, 0.1)
    send_and_wait(client, '/dim/x', +1, 0.1)
    send_and_wait(client, '/dim/z', +1, 0.1)
    send_and_wait(client, '/dim/z', +1, 0.1)
    send_and_wait(client, '/interaction/next', "", 0.1)
    send_and_wait(client, '/interaction/next', "", 0.1)
    send_and_wait(client, '/interaction/next', "", 0.1)
    print("Testing: Increase Mass Radius")
    send_and_wait(client, '/mass/radius', +1)
    print("Testing: Decrease Mass Radius")
    send_and_wait(client, '/mass/radius', -1)
    print("Testing: Shift Audio Input/Output")
    send_and_wait(client, '/shiftInOut', "x")
    send_and_wait(client, '/shiftInOut', "z")
    print("Testing: Update masses")
    send_and_wait(client, '/mass/m', [5,5,5])
    print("Testing: Update Spring Stiffness")
    send_and_wait(client, '/spring/k', [0.15, 3, 7.0])
    print("Testing: Update Spring Damping")
    send_and_wait(client, '/spring/c', [0.5, 0.5, 0.5])
    print("Testing: Set Interaction Type")
    send_and_wait(client, '/interaction/set', "DILATED2")
    print("Testing: Save mode to path")
    send_and_wait(client, '/config/save', "/Users/luisreyes/Sonify/SonoBox/ex/2.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default="/Users/luisreyes/Sonify/SonoBox/config/config.json")           # positional argument
    parser.add_argument('--input_path', default="/Users/luisreyes/Sonify/BioSonix-V2/data/BioSonix/abstract-model/ordering/S-T-Br/1/data")           # positional argument
    parser.add_argument('--mode', choices=["fem", "us"], default="fem")           # positional argument
    args = parser.parse_args()
    test(args.config_path, args.input_path, args.mode)