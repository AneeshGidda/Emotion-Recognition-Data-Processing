# Emotion-Recognition-Data-Processing

The "Emotion Recognition Data Processing" project focuses on the initial stages of preparing and processing audio data for emotion recognition tasks. This project specifically deals with data collection, feature extraction, and preprocessing, setting the foundation for subsequent machine learning tasks. It aims to create a clean and organized dataset ready for model development. The key data processing steps are as follows:

## 1. Feature Extraction
**Mel Spectrogram Extraction:** The heart of this project's data processing lies in the extraction of Mel spectrogram features from the raw audio recordings. Mel spectrograms are a time-frequency representation of audio signals. They capture essential information about the frequency components present in the audio over time

**Why Mel Spectrograms?:** Mel spectrograms are particularly suited for audio analysis tasks because they mimic how the human auditory system perceives sound. They provide a robust and informative representation for training machine learning models to classify emotions

## 2. Data Combination
**Multiple Datasets:** Audio data for this project has been collected from three distinct sources: CREMA, RAVDESS, and TESS. Each dataset provides a unique perspective on emotional expressions. To enhance the diversity and robustness of the model, data from all three sources are combined into a single dataset for training and testing

**Why Combine Datasets?:** Combining datasets helps prevent model overfitting by exposing it to a broader range of emotions and voice characteristics. This makes the model more generalizable and capable of recognizing emotions in a variety of contexts

## 3. Data Preprocessing
**Shuffling:** The combined dataset is shuffled to randomize the order of data samples. This step ensures that the model does not learn any spurious patterns based on the sequence of data

**Splitting:** The shuffled data is then split into two subsets: a training set and a testing set. Typically, a portion of the data (e.g., 80%) is used for training, and the remainder (e.g., 20%) is reserved for testing the model's performance

**Feature Scaling:** The Mel spectrogram features are scaled using Min-Max scaling. This scaling technique transforms the features to a common range (usually between 0 and 1) to ensure that all features contribute equally to the model's learning process

**Label Encoding:** Emotion labels in the original datasets are categorical. To make them compatible with machine learning algorithms, they are one-hot encoded. This means each emotion label is transformed into a binary vector where only one element is '1', representing the class of the emotion

## 4. Data Visualization
**Audio Waveform**__
Purpose: Visualize the raw audio waveform.
Benefit: Gain insights into audio duration, amplitude patterns, and variations

**Mel Spectrogram**__
Purpose: Visualize Mel spectrograms.
Benefit: Explore frequency content, intensity variations, and emotional characteristics
