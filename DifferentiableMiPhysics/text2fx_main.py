from pathlib import Path
from tqdm import tqdm

from diff_mass_spring_model_tied_exp import MassSpringModel
import torch
import numpy as np
from audiotools import AudioSignal
import soundfile as sf, sounddevice as sd
from typing import Union, List

from torch.utils.tensorboard import SummaryWriter
import json
# from msclap import CLAP

from core import Channel, create_save_dir, preprocess_audio, detensor_dict, slugify
from constants import RUNS_DIR, SAMPLE_RATE, DEVICE
torch.autograd.set_detect_anomaly(True)

alpha = 0.5               # stability margin
dt = 1/16000               # your physics step
omega_max = alpha / dt
eps = 1e-6


def stability_penalty(k, m, edge_index, ground_mask, omega_max):
    # We have masses, we want masses per edge
    edges_i, edges_j = edge_index
    mi = m[edges_i]
    mj = m[edges_j]                 # (E,)
    m_eff = (mi * mj) / (mi + mj + eps)

    # Local edge frequency
    omega = torch.sqrt(k / (m_eff + eps))

    # Timestep-dependent cap
    omega_max = (alpha / dt)

    # Hinge-squared penalty (smooth, zero when within cap)
    L_omega = torch.clamp(omega - omega_max, min=0).pow(2).mean()

    return L_omega 



"""
EX CLI USAGE
python -m text2fx --input_audio "assets/speech_examples/VCTK_p225_001_mic1.flac"\
                 --text "this sound is happy" \
                 --criterion "cosine-sim" \
                 --n_iters 600 \
                 --lr 0.01 
                 --params_init_type "zeros"
                 --
"""
device = DEVICE #torch.device("cuda:0") if torch.cuda.is_available() else "cpu"


def get_model(model_choice: str):
    if model_choice == "ms_clap":
        from msclap_wrapped import MSCLAPWrapper
        model = MSCLAPWrapper()
    else:
        raise ValueError('choose a model!!!!!!')
    return model


def clip_directional_loss(
        a1: torch.Tensor, 
        a2: torch.Tensor, 
        b1: torch.Tensor, 
        b2: torch.Tensor
    ):
        a_dir = a1 - a2
        a_dir /= a_dir.clone().norm(dim=-1, keepdim=True)

        b_dir = b1 - b2
        b_dir /= b_dir.clone().norm(dim=-1, keepdim=True)

        loss = 1 - torch.cosine_similarity(a_dir, b_dir, dim=-1)
        return loss

def text2fx(
    model_name: str,
    mass_spring_model: MassSpringModel,
    text: Union[str, List[str]],   
    device: str = "cuda" if torch.cuda.is_available() else "cpu", 
    log_audio_every_n: int = 1, 
    lr: float = 1e-2, 
    n_iters: int = 600,
    criterion: str = "standard", 
    save_dir: str = None, # figure out a save path automatically,
    params_init_type: str = "random",
    # seed_i: int = 0,
    roll_amt: int = None,
    detailed_log: bool = False,
    export_audio: bool = False,
    log_tensorboard: bool = False,
):

    clap = get_model(model_name)

    if log_tensorboard or export_audio or detailed_log:
        if not save_dir:
            save_dir = create_save_dir(f'{text}_{lr}_{criterion}', RUNS_DIR)
        else:
            save_dir = Path(save_dir)
            # save_dir = create_save_dir(f'{text}', Path(save_dir))
            save_dir.mkdir(exist_ok=True, parents=True)

    # create a writer for saving stuff to tensorboard
    if log_tensorboard:
        writer_dir = save_dir / "logs"
        writer_dir.mkdir(exist_ok=True)
        writer = SummaryWriter(writer_dir) #SummaryWriter is tensorboard writer
    else:
        writer = False

    # Print number of tunable parameters:
    print("No tunable params:", sum(p.numel() for p in mass_spring_model.parameters() if p.requires_grad))
    
    # Log the model, torch amount, starting parameters, and their values
    if log_tensorboard or export_audio or detailed_log:
        log_file = save_dir / f"experiment_log.txt"
        with open(log_file, "w") as log:
            log.write(f"Model: {model_name}\n")
            log.write(f"MassSpringModel #params: {sum(p.numel() for p in mass_spring_model.parameters())}\n")
            log.write(f"Learning Rate: {lr}\n")
            log.write(f"Number of Iterations: {n_iters}\n")
            log.write(f"Criterion: {criterion}\n")
            log.write(f"Params Initialization Type: {params_init_type}\n")
            log.write(f"Custom roll?: {roll_amt}\n")
            log.write("="*40 + "\n")

    optimizer = torch.optim.Adam(mass_spring_model.parameters(), lr=lr)     # the optimizer!

    events = {
        8000: (3, 3, 3),
        12000: (3, 3, 5),
        # more events...
    }
    fs = 16000
    seconds = 1 
    init_sig = mass_spring_model.render_audio(
        seconds=seconds,
        fs=fs,
        observable='pos',
        axis='all',
        listener_ids=mass_spring_model.get_listener_ids(),
        layout='mono',
        pan_method='by_position',
        hp=True,
        events=events
    )  # [T_audio, 1]    
    
    # Logging
    if writer:
        writer.add_audio("effected", init_sig, 0, sample_rate=fs)
    # sig_in.clone().cpu().write(save_dir / 'input.wav')
    if export_audio: #starting audio
        sf.write(save_dir / f'starting.wav', init_sig.detach().cpu().numpy(), 16000)

    # Preparing our text target
    sig = AudioSignal(init_sig, sample_rate=fs)

    if isinstance(text, str):
        text = [text]
    assert len(text) == sig.batch_size or len(text) == 1

    if len(text) < sig.batch_size:
        text = text * sig.batch_size

    # Preprocess text
    text_processed = [
        f"this sound is {t}" for t in text
    ]
    # breakpoint()
    embedding_target = clap.get_text_embeddings(text_processed).detach()
    print(f"Text processed: {text_processed}")
    print(f"Text embedding: {embedding_target.shape}")

    if criterion == "directional_loss":
        audio_in_emb = clap.get_audio_embeddings(sig.to(device)).detach()

        text_neg_processed = [
            f"this sound is not {t}" for t in text
        ]
        text_anchor_emb = clap.get_text_embeddings(text_neg_processed).detach()

    final_losses = []

    # Single-Instance Optimization: Optimize our parameters by matching effected audio against the target text embedding
    pbar = tqdm(range(n_iters), total=n_iters)
    for n in pbar:
        # Apply effect with out estimated parameters
        # Code for signal rolling
        sig_roll = sig.clone()
        if roll_amt or roll_amt == 0:
            roll_amount = torch.randint(-roll_amt, roll_amt + 1, (sig_roll.batch_size,))
        else:
            roll_amount = torch.randint(0, sig_roll.signal_length, (sig_roll.batch_size,))

        if log_tensorboard or export_audio or detailed_log:
            with open(log_file, "a") as log:
                params = torch.cat([p.view(-1) for p in mass_spring_model.parameters() if p.requires_grad])
                log.write(f"Iteration {n} Params Values: {params.data.cpu().numpy()}\n")
                log.write(f"Iteration {n} Loss: {loss.item()}\n")

        for i in range(sig_roll.batch_size):
            rolled = torch.roll(sig_roll.samples[i], shifts=roll_amount[i].item(), dims=-1)
            # print(rolled)
            sig_roll.samples[i:i+1] = rolled

        mass_spring_model.detach_state(reset_to_rest=True)
        # print(f"Param values iter {n}:")
        # print("K:",    (mass_spring_model.k * torch.exp(mass_spring_model.theta_k)).detach().item())
        # print("Z:",    (mass_spring_model.z * torch.exp(mass_spring_model.theta_z)).detach().item())
        # print("fric:", (mass_spring_model.fric * torch.exp(mass_spring_model.theta_fric)).detach().item())
        print(f"Param values iter {n}:")
        print("K:",    (mass_spring_model.k).detach().item())
        print("Z:",    (mass_spring_model.z).detach().item())
        print("fric:", (mass_spring_model.fric).detach().item())
        signal_mi = mass_spring_model.render_audio(
            seconds=seconds,
            fs=fs,
            observable='pos',
            axis='all',
            listener_ids=mass_spring_model.get_listener_ids(),
            layout='mono',
            pan_method='by_position',
            hp=True,
            events=events
        )  # [T_audio, 1]
        signal_sim = AudioSignal(signal_mi, sample_rate=fs)

        # Get CLAP embedding for effected audio
        embedding_sim = clap.get_audio_embeddings(signal_sim) #.get_audio_embeddings takes in preprocessed audio

        # Calculating Loss
        if criterion == "directional_loss":
            batch_loss = clip_directional_loss(embedding_sim, audio_in_emb, embedding_target, text_anchor_emb)
        elif criterion == "standard": #is neg dot product loss aims to minimize the dot prod b/w dissimilar items, no direction intake
            batch_loss = -(embedding_sim @ embedding_target.T)
        elif criterion == "cosine-sim": # cosine_sim loss aims to maximize the cosine similarity between similar items, normalized
            batch_loss = 1 - torch.cosine_similarity(embedding_sim, embedding_target, dim=-1)
        else:
            raise ValueError(f"Criterion {criterion} not recognized")
        
        loss = batch_loss.mean()
        # loss += 0.1* stability_penalty(mass_spring_model.k, 1/mass_spring_model.inv_mass, mass_spring_model.edge_index, mass_spring_model.fixed_mask, omega_max)
        if writer:
            writer.add_scalar("loss", loss.item(), n)

        # Optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_description(f"step: {n+1}/{n_iters}, loss: {loss.item():.3f}")

        #saving last batch_loss
        if n == n_iters - 1:
            final_losses = batch_loss.detach().cpu().numpy()

        if n % log_audio_every_n == 0:
            # Save audio
            sf.write(save_dir / f'optim_{n}.wav', signal_mi.detach().cpu().numpy(), 16000)

        # detailed logging, log params + signal every 100 iters
        if detailed_log:
            init_sig_path = Path(init_sig.path_to_file)
            detailed_dir = Path(save_dir) / 'detailed_logs'
            detailed_dir.mkdir(parents=True, exist_ok=True)
            json_log_path = detailed_dir / "params_log.json"  # Path to save the JSON file
            sig.clone().detach().cpu().write(detailed_dir / f'{init_sig_path.stem}__ref.wav')
            if n % 100 == 0 or n==n_iters-1:
                params_i = params.detach().cpu()
                out_params_dict = channel.save_params_to_dict(params.detach().cpu())
                print(out_params_dict)
                with open(json_log_path, "a") as json_log_file:
                    json.dump({"iteration": n, "params": detensor_dict(out_params_dict)}, json_log_file)
                    json_log_file.write("\n")  # For better readability in the file
                    json.dump({"iteration": n, "raw_params": params_i.tolist()}, json_log_file)
                    json_log_file.write("\n")  # For better readability in the file

                signal_sim.detach().cpu().ensure_max_of_audio().write(detailed_dir / f'{init_sig_path.stem}_{n}.wav')

    if log_tensorboard or export_audio or detailed_log:
        with open(log_file, "a") as log:
            log.write(f"ENDING Params Values: {params.data.cpu().numpy()}\n")
    
    # min_loss_index = int(np.argmin(final_losses)) # used for comparing across multiple runs

    # Play final signal with optimized effects parameters
    # out_sig = channel(clean_sig.clone().to(device), torch.sigmoid(params)).clone().detach().cpu()
    # out_sig = mass_spring_model.render_audio(
    #     seconds=seconds,
    #     fs=fs,
    #     observable='pos',
    #     axis='all',
    #     listener_ids=mass_spring_model.get_listener_ids(),
    #     layout='mono',
    #     pan_method='by_position',
    #     hp=True,
    #     events=events
    # )  # [T_audio, 1]
    # out_sig = AudioSignal(out_sig, sample_rate=fs)
    # # out_sig = channel(sig.clone().to(device), torch.sigmoid(params)).clone().detach().cpu()
    # out_sig = preprocess_audio(out_sig) 
    # out_params = params.detach().cpu() #optimized output FXparams
    # out_params_dict = channel.save_params_to_dict(out_params) #mapping back to FX ranges

    # if export_audio:
    #     if sig.batch_size == 1:
    #         out_sig.detach().cpu().write(save_dir / f'{init_sig_path.stem}_final.wav')
    #         # out_sig.clone().detach().cpu().write(save_dir / f'{init_sig_path.stem}_final.wav')
    #     else:
    #         for i, s in enumerate(out_sig):
    #             i_init_sig_path = Path(init_sig.path_to_file[i])
    #             out_sig[i].detach().cpu().write(save_dir / f'{i_init_sig_path.stem}_final.wav')

    # # out_sig.write(save_dir / "final.wav")

    # if writer:
    #     writer.add_audio("final", out_sig.samples[0][0], n_iters, sample_rate=out_sig.sample_rate)
    #     writer.close()

    # return out_sig, out_params, out_params_dict