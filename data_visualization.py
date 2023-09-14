import matplotlib.pyplot as plt
import librosa

class visualize():
    def __init__(self, audio_file):
        numerical_encoding, self.sampling_rate = librosa.load(audio_file)
        max_duration = 116247
        self.zero_padded_data = librosa.util.fix_length(numerical_encoding, size=max_duration)
        self.mel_spectrogram = librosa.feature.melspectrogram(y=self.zero_padded_data, sr=self.sampling_rate)

    def print_info(self):
        print(f"size of data: {len(self.zero_padded_data)}\nsampling_rate: {self.sampling_rate}")
        print(f"shape of data: {self.zero_padded_data.shape}")
        print(f"shape of mel spectrogram: {self.mel_spectrogram.shape}\n")

    def plot_audio_wave(self):
        plt.figure(figsize=(15, 5))
        librosa.display.waveshow(self.zero_padded_data, sr=self.sampling_rate)
        plt.show()

    def plot_mel_spectrogram(self):
        mel_spectrogram_db = librosa.power_to_db(self.mel_spectrogram)
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(mel_spectrogram_db, sr=self.sampling_rate, x_axis='time', y_axis='mel')
        plt.colorbar(format="%+2.0f dB")
        plt.title("Mel Spectrogram")
        plt.show()

crema_data_1 = visualize("CREMA\\1001_DFA_ANG_XX.wav")
ravdess_data_1 = visualize("RAVDESS\\03-01-07-02-01-02-19.wav")
tess_data_1 = visualize("TESS\OAF_back_sad.wav")
crema_data_1.plot_audio_wave()
crema_data_1.print_info()
crema_data_1.plot_mel_spectrogram()