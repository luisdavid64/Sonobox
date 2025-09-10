from diff_mass_spring_model import MassSpringModel
import torch
torch.autograd.set_detect_anomaly(True)

if __name__ == "__main__":
    import torch
    import sounddevice as sd

    # Simple test of the mass-spring model and audio output
    fs = 16000
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ## Add mps if on available
    # device = 'mps' if torch.backends.mps.is_available() else device
    model = MassSpringModel.from_json("/Users/luisreyes/Sonify/SonoBox/model_configs/sonobox_data/baselines/biosonix_3D.json", device=device, dt=1/fs).to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    events = {
        3000: (3, 3, 3),
        6000: (3, 0, 5),
        # more events...
    }
    init_k = model.k.clone()
    init_edgeZ = model.z.clone()
    init_fric = model.fric.clone()
    init_invM = model.inv_mass.clone()

    # Simulate and get audio
    seconds = 0.5  # seconds
    # Let's optimize for 100 iters to make the audio 0
    iters = 100
    for i in range(iters):
        model.detach_state(reset_to_rest=True)
        audio = model.render_audio(seconds=seconds, fs=fs, axis='all', listener_ids=model.get_listener_ids(), layout='mono', events=events)  # [T_audio, 1]
        # Contains nan, optimized values become unstable (i guess invM)
        # print(audio.isnan().any())
        loss = (audio**2).mean()
        print(f"Iter {i+1}/{iters}, audio power: {loss.item():.6f}")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        import soundfile as sf, sounddevice as sd
        sf.write(f"mass_spring_{i}.wav", audio.detach().cpu().numpy(), 16000)
        # Print current params vs original at init
        print("  |K|   :", torch.mean(torch.abs(model.k)).item(), " (init ", torch.mean(torch.abs(init_k)).item(), ")")
        print("  |edgeZ|:", torch.mean(torch.abs(model.z)).item(), " (init ", torch.mean(torch.abs(init_edgeZ)).item(), ")")
        # Print inv mass and friction
        print("  |invM| :", torch.mean(torch.abs(model.inv_mass)).item(), " (init ", torch.mean(torch.abs(init_invM)).item(), ")")
        print(" |fric| :", torch.mean(torch.abs(model.fric)).item(), " (init ", torch.mean(torch.abs(init_fric)).item(), ")")