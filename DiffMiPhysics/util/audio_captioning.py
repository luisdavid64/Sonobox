"""
This is an example using CLAPCAP for audio captioning.
"""
from msclap import CLAP

# Load and initialize CLAP
clap_model = CLAP(version = 'clapcap', use_cuda=False)

#Load audio files
audio_files = [
    'exp/exp_with_exponential_params/church-bell-long-ring-002/optim_68_quite_nice.wav',
    'exp/exp_with_exponential_params/kick-drum-003/optim_35_decent.wav',
    'exp/exp_with_exponential_params/glassy/optim_59_very_nice_too.wav',
    'exp/2025-09-14/emotional-piano-playing-001/optim_22_yes.wav',
    '/Users/luisreyes/Sonify/SonoBox/DiffMiPhysics/samples/piano_note.wav'
]

# Generate captions for the recording
captions = clap_model.generate_caption(audio_files, resample=True, beam_size=5, entry_length=67, temperature=0.01)

# Print the result
for i in range(len(audio_files)):
    print(f"Audio file: {audio_files[i]} \n")
    print(f"Generated caption: {captions[i]} \n")
