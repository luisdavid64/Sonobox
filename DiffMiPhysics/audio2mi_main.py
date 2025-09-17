from pathlib import Path
from tqdm import tqdm

from util.audio_helpers import clap_preprocess
from util.util import event_dict_seconds_to_samples, load_event_from_json, scale_dict_samples
from diff_mass_spring_model import MassSpringModel
import torch
import numpy as np
from audiotools import AudioSignal
import soundfile as sf, sounddevice as sd
from typing import Union, List

from torch.utils.tensorboard import SummaryWriter
import json
# from msclap import CLAP

from core import create_save_dir, detensor_dict
from constants import RUNS_DIR, SAMPLE_RATE, DEVICE

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
python -m text2mi --input_audio "assets/speech_examples/VCTK_p225_001_mic1.flac"\
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

def audio2mi(
    model_name: str,
    mass_spring_model: MassSpringModel,
    reference_audio_path: str,  # <-- now expects a path to reference audio
    device: str = "cuda" if torch.cuda.is_available() else "cpu", 
    log_audio_every_n: int = 1, 
    lr: float = 1e-2, 
    n_iters: int = 600,
    criterion: str = "standard", 
    save_dir: str = None, # figure out a save path automatically,
    params_init_type: str = "random",
    # seed_i: int = 0,
    detailed_log: bool = False,
    export_audio: bool = False,
    log_tensorboard: bool = False,
):
    clap = get_model(model_name)

    if log_tensorboard or export_audio or detailed_log:
        if not save_dir:
            save_dir = create_save_dir(f'{reference_audio_path}_{lr}_{criterion}', RUNS_DIR)
        else:
            save_dir = Path(save_dir)
            save_dir.mkdir(exist_ok=True, parents=True)

    # create a writer for saving stuff to tensorboard
    if log_tensorboard:
        writer_dir = save_dir / "logs"
        writer_dir.mkdir(exist_ok=True)
        writer = SummaryWriter(writer_dir)
    else:
        writer = False

    print("No tunable params:", sum(p.numel() for p in mass_spring_model.parameters() if p.requires_grad))
    if log_tensorboard or export_audio or detailed_log:
        log_file = save_dir / f"experiment_log.txt"
        with open(log_file, "w") as log:
            log.write(f"Model: {model_name}\n")
            log.write(f"MassSpringModel #params: {sum(p.numel() for p in mass_spring_model.parameters())}\n")
            log.write(f"Learning Rate: {lr}\n")
            log.write(f"Number of Iterations: {n_iters}\n")
            log.write(f"Criterion: {criterion}\n")
            log.write(f"Params Initialization Type: {params_init_type}\n")
            log.write("="*40 + "\n")

    optimizer = torch.optim.Adam(mass_spring_model.parameters(), lr=lr)

    fs = 16000
    seconds = 1 
    events = load_event_from_json("events/two_hits.json")
    events = event_dict_seconds_to_samples(events, fs)
    init_sig = mass_spring_model.render_audio(
        seconds=seconds,
        fs=fs,
        observable='pos',
        axis='all',
        listener_ids=mass_spring_model.get_listener_ids(),
        layout='mono',
        pan_method='by_position',
        hp=True,
        mix_audio=True,
        events=events
    )  # [T_audio, 1]    
    init_sig = init_sig.squeeze()
    if writer:
        writer.add_audio("effected", init_sig, 0, sample_rate=fs)
    if export_audio:
        sf.write(save_dir / f'starting.wav', init_sig.squeeze().detach().cpu().numpy(), 16000)

    # Load and preprocess reference audio
    ref_audio, ref_sr = sf.read(reference_audio_path)
    if ref_audio.ndim > 1:
        ref_audio = ref_audio.mean(axis=1)  # convert to mono if needed
    ref_audio = torch.tensor(ref_audio, dtype=torch.float32)
    if ref_sr != 44100:
        import torchaudio
        ref_audio = torchaudio.functional.resample(ref_audio, orig_freq=ref_sr, new_freq=44100)
    ref_audio = ref_audio.unsqueeze(0)  # [1, T]
    ref_signal = AudioSignal(ref_audio, sample_rate=44100)
    embedding_target = clap.get_audio_embeddings(ref_signal).detach()
    print(f"Reference audio embedding: {embedding_target.shape}")

    final_losses = []
    if log_tensorboard or export_audio or detailed_log:
        with open(log_file, "a") as log:
            log.write(f"Beginning Parameters:\n") 
            if mass_spring_model.k.dim() > 0 and mass_spring_model.k.size(0) != 1:
                log.write(f"K: {mass_spring_model.k.detach().cpu().numpy()}\n")
            else:
                log.write(f"K1: {mass_spring_model.k_1.detach().item()}\n")
                log.write(f"K2: {mass_spring_model.k_2.detach().item()}\n")
            log.write(f"Z: {mass_spring_model.z.detach().item()}\n")
            log.write(f"Friction: {mass_spring_model.fric.detach().item()}\n")

    pbar = tqdm(range(n_iters), total=n_iters)
    for n in pbar:
        mass_spring_model.detach_state(reset_to_rest=True)
        print(f"Param values iter {n}:")
        if mass_spring_model.k.shape[0] != mass_spring_model.springs.shape[0]:
            print("K:",    (mass_spring_model.k).detach().item())
        else:
            print("K1:",   (mass_spring_model.k_1).detach().item())
            print("K2:",   (mass_spring_model.k_2).detach().item())
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
            mix_audio=True,
            hp=True,
            events=events
        )  # [T_audio, 1]
        signal_mi = signal_mi.squeeze()
        signal_sim = clap_preprocess(signal_mi, fs_in=fs)
        signal_sim = AudioSignal(signal_sim, sample_rate=44100)
        embedding_sim = clap.get_audio_embeddings(signal_sim)

        if criterion == "directional_loss":
            raise NotImplementedError("Directional loss not supported for audio-to-audio mode.")
        elif criterion == "standard":
            batch_loss = -(embedding_sim @ embedding_target.T)
        elif criterion == "cosine-sim":
            batch_loss = 1 - torch.cosine_similarity(embedding_sim, embedding_target, dim=-1)
        else:
            raise ValueError(f"Criterion {criterion} not recognized")
        loss = batch_loss.mean()
        if writer:
            writer.add_scalar("loss", loss.item(), n)
        if log_tensorboard or export_audio or detailed_log:
            with open(log_file, "a") as log:
                params = torch.cat([p.view(-1) for p in mass_spring_model.parameters() if p.requires_grad])
                log.write(f"Iteration {n} Parameters:\n") 
                if mass_spring_model.k.shape[0] != mass_spring_model.springs.shape[0]:
                    log.write(f"K: {mass_spring_model.k.detach().cpu().numpy()}\n")
                else:
                    log.write(f"K1: {mass_spring_model.k_1.detach().item()}\n")
                    log.write(f"K2: {mass_spring_model.k_2.detach().item()}\n")
                log.write(f"Z: {mass_spring_model.z.detach().item()}\n")
                log.write(f"Friction: {mass_spring_model.fric.detach().item()}\n")
                log.write(f"Loss: {loss.item()}\n")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pbar.set_description(f"step: {n+1}/{n_iters}, loss: {loss.item():.3f}")
        if n == n_iters - 1:
            final_losses = batch_loss.detach().cpu().numpy()
        if n % log_audio_every_n == 0:
            sf.write(save_dir / f'optim_{n}.wav', signal_mi.squeeze().detach().cpu().numpy(), 16000)
    if log_tensorboard or export_audio or detailed_log:
        with open(log_file, "a") as log:
            log.write(f"ENDING Params Values: {params.data.cpu().numpy()}\n")
    min_loss_index = int(np.argmin(final_losses)) # used for comparing across multiple runs

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