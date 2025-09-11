import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Paths to your WAV files
file1 = 'exp/2025-09-11/wooden-mallet-marimba-like-003/starting.wav'
file2 = 'exp/2025-09-11/wooden-mallet-marimba-like-003/optim_40_nice_wooden.wav'
file1 = '/Users/luisreyes/Sonify/SonoBox/DifferentiableMiPhysics/exp/2025-09-11/church-bell-long-ring-002/starting.wav'
file2 = '/Users/luisreyes/Sonify/SonoBox/DifferentiableMiPhysics/exp/2025-09-11/church-bell-long-ring-002/optim_40.wav'

# Load audio
y1, sr1 = librosa.load(file1, sr=None)
y2, sr2 = librosa.load(file2, sr=None)

# Compute spectrograms
S1 = librosa.amplitude_to_db(librosa.stft(y1), ref=np.max)
S2 = librosa.amplitude_to_db(librosa.stft(y2), ref=np.max)

# Plot side by side
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
librosa.display.specshow(S1, sr=sr1, x_axis='time', y_axis='hz')
plt.title('Spectrogram: File 1')
plt.colorbar(format='%+2.0f dB')

plt.subplot(1, 2, 2)
librosa.display.specshow(S2, sr=sr2, x_axis='time', y_axis='hz')
plt.title('Spectrogram: File 2')
plt.colorbar(format='%+2.0f dB')

plt.tight_layout()
plt.show()