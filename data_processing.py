import numpy as np
import librosa
import os
from tqdm import tqdm
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from joblib import Parallel, delayed

# Define a function to extract audio features from an audio file
def extract_features(audio_file):
    # Load the audio file using librosa
    numerical_encoding, sampling_rate = librosa.load(audio_file)
    
    # Define the maximum duration for zero-padding
    max_duration = 116247
    
    # Zero-pad the audio to the maximum duration
    zero_padded_data = librosa.util.fix_length(numerical_encoding, size=max_duration)
    
    # Compute the Mel spectrogram as audio features
    mel_spectrogram = librosa.feature.melspectrogram(y=zero_padded_data, sr=sampling_rate)
    return mel_spectrogram

# Define dictionaries to map emotions to numerical labels for different datasets
crema_emotions = {"NEU" : 0, "HAP" : 1, "SAD" : 2, "ANG" : 3, "FEA" : 4, "DIS" : 5}
ravdess_emotions = {"01" : 0, "03" : 1, "04" : 2, "05" : 3, "06" : 4, "07" : 5}
tess_emotions = {"neutral" : 0, "happy" : 1, "sad" : 2, "angry" : 3, "fear" : 4, "disgust" : 5}

# Initialize arrays to store audio features and corresponding labels
x_data = np.empty(shape=(11145, 128, 228))
y_data = np.empty(shape=(11145))

# Loop through the "CREMA" dataset directory to extract features and labels
for root, dirs, files in os.walk("CREMA"):
    # Use parallel processing to extract audio features from files
    x_data = Parallel(n_jobs=-1, prefer="processes")(delayed(extract_features)(os.path.join(root, file)) for file in tqdm(files, desc=f"CREMA-X"))
    
    # Extract emotion labels from file names and assign numerical labels
    for i, file in enumerate(tqdm(files, desc=f"CREMA-Y")):
        y_data[i] = crema_emotions[file.split('_')[2]]

# Loop through the "RAVDESS" dataset directory to extract features and labels
for root, dirs, files in os.walk("RAVDESS"):
    # Use parallel processing to extract audio features from files and assign them to the appropriate portion of the data array
    x_data[7442:8744] = Parallel(n_jobs=-1, prefer="processes")(delayed(extract_features)(os.path.join(root, file)) for file in tqdm(files, desc="RAVDESS-X"))
    
    # Extract emotion labels from file names and assign numerical labels, skipping certain emotions
    for i, file in enumerate(tqdm(files, desc="RAVDESS-Y")):
        if file.split('-')[2] != "02" and file.split('-')[2] != "08":
            y_data[i] = ravdess_emotions[file.split('-')[2]]

# Loop through the "TESS" dataset directory to extract features and labels
for root, dirs, files in os.walk("TESS"):
    # Use parallel processing to extract audio features from files and assign them to the appropriate portion of the data array
    x_data[8745:11144] = Parallel(n_jobs=-1, prefer="processes")(delayed(extract_features)(os.path.join(root, file)) for file in tqdm(files, desc="TESS-X"))
    
    # Extract emotion labels from file names and assign numerical labels based on the keyword
    for i, file in enumerate(tqdm(files, desc="TESS-Y")):
        base_name = file.split(".")[0]
        keyword = base_name.split("_")[-1]
        y_data[i] = tess_emotions[keyword]

# Shuffle the data
x_data, y_data = shuffle(x_data, y_data, random_state=42)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

# Initialize Min-Max Scaler to scale the data
scalar = MinMaxScaler()

# Reshape the data for scaling
x_train = np.reshape(x_train, (5953, 29184))
x_test = np.reshape(x_test, (1489, 29184))

# Fit the scaler on the training data and scale both training and testing data
scalar.fit(x_train)
x_train = np.reshape(scalar.transform(x_train), (5953, 128, 228))
x_test = np.reshape(scalar.transform(x_test), (1489, 128, 228))

# Initialize One-Hot Encoder to encode categorical labels
encoder = OneHotEncoder()

# Reshape the labels for encoding
y_train = np.reshape(y_train, (-1, 1))
y_test = np.reshape(y_test, (-1, 1))

# Fit the encoder on the training labels and encode both training and testing labels
encoder.fit(y_train)
y_train = np.squeeze(encoder.transform(y_train).toarray())
y_test = np.squeeze(encoder.transform(y_test).toarray())

# Save the preprocessed data to numpy files
np.save("x_train_c.npy", x_train)
np.save("x_test_c.npy", x_test)
np.save("y_train_c.npy", y_train)
np.save("y_test_c.npy", y_test)
