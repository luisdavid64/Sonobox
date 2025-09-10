from diff_mass_spring_model import MassSpringModel
import torch
torch.autograd.set_detect_anomaly(True)

if __name__ == "__main__":
    import torch
    import sounddevice as sd

    # Simple test of the mass-spring model and audio output
    fs = 16000
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MassSpringModel.from_json("/Users/luisreyes/Sonify/SonoBox/model_configs/sonobox_data/baselines/biosonix_3D.json", device=device, dt=1/fs).to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    # Simulate and get audio
    seconds = 1.0  # seconds
    # Let's optimize for 100 iters to make the audio 0
    iters = 100
    for i in range(iters):
        model.detach_state()
        audio = model.render_audio(seconds=seconds, fs=fs, axis='all', listener_ids=model.get_listener_ids(), layout='mono')

        loss = (audio**2).mean()
        print(f"Iter {i+1}/{iters}, audio power: {loss.item():.6f}")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()