import matplotlib.pyplot as plt
import librosa

class visualize():
    def __init__(self, audio_file):
        # Load the audio file using librosa
        numerical_encoding, self.sampling_rate = librosa.load(audio_file)
        
        # Define the maximum duration for zero-padding
        max_duration = 116247
        
        # Zero-pad the audio to the maximum duration
        self.zero_padded_data = librosa.util.fix_length(numerical_encoding, size=max_duration)
        
        # Compute the Mel spectrogram of the zero-padded audio
        self.mel_spectrogram = librosa.feature.melspectrogram(y=self.zero_padded_data, sr=self.sampling_rate)

    def print_info(self):
        # Print information about the audio data
        print(f"Size of data: {len(self.zero_padded_data)}\nSampling rate: {self.sampling_rate}")
        print(f"Shape of data: {self.zero_padded_data.shape}")
        print(f"Shape of Mel spectrogram: {self.mel_spectrogram.shape}\n")

    def plot_audio_wave(self):
        # Plot the audio waveform
        plt.figure(figsize=(15, 5))
        librosa.display.waveshow(self.zero_padded_data, sr=self.sampling_rate)
        plt.show()

    def plot_mel_spectrogram(self):
        # Compute the Mel spectrogram in dB scale
        mel_spectrogram_db = librosa.power_to_db(self.mel_spectrogram)
        
        # Plot the Mel spectrogram
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(mel_spectrogram_db, sr=self.sampling_rate, x_axis='time', y_axis='mel')
        plt.colorbar(format="%+2.0f dB")
        plt.title("Mel Spectrogram")
        plt.show()

# Create instances of the 'visualize' class for different audio files
crema_data_1 = visualize("CREMA\\1001_DFA_ANG_XX.wav")
ravdess_data_1 = visualize("RAVDESS\\03-01-07-02-01-02-19.wav")
tess_data_1 = visualize("TESS\OAF_back_sad.wav")

# Plot the audio waveform and display information for the 'crema_data_1' instance
crema_data_1.plot_audio_wave()
crema_data_1.print_info()
crema_data_1.plot_mel_spectrogram()
