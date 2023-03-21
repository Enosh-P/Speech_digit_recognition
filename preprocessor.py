import librosa
import librosa.display
import numpy as np
from sklearn import preprocessing
from torch.utils.data import Dataset

SAMPLING_RATE = 8000  # This value is determined by the wav file, DO NOT CHANGE


def extract_melspectrogram(signal, sr, num_mels):
    """
    Given a time series speech signal (.wav), sampling rate (sr),
    and the number of mel coefficients, return a mel-scaled
    representation of the signal as numpy array.
    """

    mel_features = librosa.feature.melspectrogram(
        y=signal,
        sr=sr,
        n_fft=200,  # with sampling rate = 8000, this corresponds to 25 ms
        hop_length=80,  # with sampling rate = 8000, this corresponds to 10 ms
        n_mels=num_mels,  # number of frequency bins, use either 13 or 39
        fmin=50,  # min frequency threshold
        fmax=4000,  # max frequency threshold, set to SAMPLING_RATE/2
    )

    # for numerical stability added this line
    mel_features = np.where(mel_features == 0, np.finfo(float).eps, mel_features)

    # 20 * log10 to convert to log scale
    log_mel_features = 20 * np.log10(mel_features)

    # feature scaling
    scaled_log_mel_features = preprocessing.scale(log_mel_features, axis=1)

    return scaled_log_mel_features


def get_audio_path(row):
    """Get the audio path of a file"""
    return f"{row.file}"


def get_mel_spectrogram(file_path, sr=SAMPLING_RATE, num_mels=13):
    audio_data, sr = librosa.load(file_path, sr=sr)
    return extract_melspectrogram(audio_data, sr, num_mels)


def downsample_spectrogram(spectrogram, n=15, flattened=True):
    """
    Given a spectrogram of an arbitrary length/duration (X ∈ K x T),
    return a downsampled version of the spectrogram v ∈ K * N
    """
    k, t = spectrogram.shape
    split_size = t // n
    split_indices = [i * split_size for i in range(n)] + [t]

    X_splits = [
        spectrogram[:, split_indices[i] : split_indices[i + 1]] for i in range(n)
    ]
    X_downsampled = np.array([np.mean(split, axis=1) for split in X_splits])

    if flattened:
        return X_downsampled.reshape((n * k,))
    return X_downsampled


class SpectrogramDataset(Dataset):
    """Building spectrogram and add its label into an array"""

    def __init__(self, df, n=0, flattened=True):
        """
        :param df: dataframe for the dataset
        :param n: number of length in a split
        """
        self.df = df
        self.data = []
        self.labels = []
        # n cannot be greater than 25 or less than 1. if n = 0, then pooling is not used
        self.n = min(25, n, 0)

        for index in df.index:
            row = df.loc[index]
            file_path = get_audio_path(row)
            spectrogram = get_mel_spectrogram(file_path)
            if n:
                spectrogram = downsample_spectrogram(
                    spectrogram, n, flattened=flattened
                )
            self.data.append(spectrogram)
            self.labels.append(row.label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
