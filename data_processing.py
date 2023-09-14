import numpy as np
import librosa
import os
from tqdm import tqdm
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from joblib import Parallel, delayed

def extract_features(audio_file):
    numerical_encoding, sampling_rate = librosa.load(audio_file)
    max_duration = 116247
    zero_padded_data = librosa.util.fix_length(numerical_encoding, size=max_duration)
    mel_spectrogram = librosa.feature.melspectrogram(y=zero_padded_data, sr=sampling_rate)
    return mel_spectrogram

crema_emotions = {"NEU" : 0, "HAP" : 1, "SAD" : 2, "ANG" : 3, "FEA" : 4, "DIS" : 5}
ravdess_emotions = {"01" : 0, "03" : 1, "04" : 2, "05" : 3, "06" : 4, "07" : 5}
tess_emotions = {"neutral" : 0, "happy" : 1, "sad" : 2, "angry" : 3, "fear" : 4, "disgust" : 5}
x_data = np.empty(shape=(11145, 128, 228))
y_data = np.empty(shape=(11145))

for root, dirs, files in os.walk("CREMA"):
        x_data = Parallel(n_jobs=-1, prefer="processes")(delayed(extract_features)(os.path.join(root, file)) for file in tqdm(files, desc=f"CREMA-X"))
        for i, file in enumerate(tqdm(files, desc=f"CREMA-Y")):
            y_data[i] = crema_emotions[file.split('_')[2]]

for root, dirs, files in os.walk("RAVDESS"):
    x_data[7442:8744] = Parallel(n_jobs=-1, prefer="processes")(delayed(extract_features)(os.path.join(root, file)) for file in tqdm(files, desc="RAVDESS-X"))
    for i, file in enumerate(tqdm(files, desc="RAVDESS-Y")):
        if file.split('-')[2] != "02" and file.split('-')[2] != "08":
            y_data[i] = ravdess_emotions[file.split('-')[2]]

for root, dirs, files in os.walk("TESS"):
    x_data[8745:11144] = Parallel(n_jobs=-1, prefer="processes")(delayed(extract_features)(os.path.join(root, file)) for file in tqdm(files, desc="TESS-X"))
    for i, file in enumerate(tqdm(files, desc="TESS-Y")):
        base_name = file.split(".")[0]
        keyword = base_name.split("_")[-1]
        y_data[i] = tess_emotions[keyword]

x_data, y_data = shuffle(x_data, y_data, random_state=42)
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

scalar = MinMaxScaler()
x_train = np.reshape(x_train, (5953, 29184))
x_test = np.reshape(x_test, (1489, 29184))
scalar.fit(x_train)
x_train = np.reshape(scalar.transform(x_train), (5953, 128, 228))
x_test = np.reshape(scalar.transform(x_test), (1489, 128, 228))

encoder = OneHotEncoder()
y_train = np.reshape(y_train, (-1, 1))
y_test = np.reshape(y_test, (-1, 1))
encoder.fit(y_train)
y_train = np.squeeze(encoder.transform(y_train).toarray())
y_test = np.squeeze(encoder.transform(y_test).toarray())

np.save("x_train_c.npy", x_train)
np.save("x_test_c.npy", x_test)
np.save("y_train_c.npy", y_train)
np.save("y_test_c.npy", y_test)