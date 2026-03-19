import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# 1. Load the Audio (The Nyquist-Shannon Step)
# librosa.load mathematically resamples the audio to 22,050 Hz by default
# 'sr=None' forces it to keep your original sample rate (e.g., 48000 Hz)
file_path = 'your_audio_file.wav'
y, sr = librosa.load(file_path, sr=48000)

# 2. Define the DSP Parameters (The Math Variables)
# n_fft is the window size for the STFT (How many samples we look at once)
# hop_length is how far we slide the window forward in time
n_fft = 2048
hop_length = 512

# 3. Calculate the Mel-Spectrogram (The AI's "Ears")
# This single function runs the STFT and applies the Mel-scale conversion formula
mel_spectrogram = librosa.feature.melspectrogram(
    y=y,
    sr=sr,
    n_fft=n_fft,
    hop_length=hop_length,
    n_mels=128 # The number of Mel bands (Resolution of the Y-axis)
)

# 4. Convert Power to Decibels (The Logarithmic Volume Step)
# AI models and human ears process volume logarithmically.
# We apply the dB equation: dB = 10 * log10(amplitude / ref)
mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

# --- At this exact point, 'mel_spectrogram_db' is the 2D Tensor ---
# --- You would feed this tensor into PyTorch, OpenAI, or Anthropic APIs ---

print(f"The AI sees a matrix with the shape: {mel_spectrogram_db.shape}")
# Example output: (128, 431) -> 128 frequency bands mapped over 431 time steps.

# 5. Visualize the Data (For the Human Engineer)
plt.figure(figsize=(10, 4))
librosa.display.specshow(
    mel_spectrogram_db,
    sr=sr,
    hop_length=hop_length,
    x_axis='time',
    y_axis='mel'
)
plt.colorbar(format='%+2.0f dB')
plt.title('Mel-Spectrogram (Logarithmic Frequency & Volume)')
plt.tight_layout()

# Save the plot so you can see what the AI is "hearing"
plt.savefig('ai_hearing_output.png')
plt.show()
