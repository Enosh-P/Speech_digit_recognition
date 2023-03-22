import librosa
import librosa.display
import warnings
import numpy as np
import torch
from sklearn import preprocessing
import torchaudio
import random
from torch.utils.data import Dataset

SAMPLING_RATE = 8000  # This value is determined by the wav file, DO NOT CHANGE
warnings.filterwarnings("ignore")
random_step = random.randint(-4, 4)
pitch_transform = torchaudio.transforms.PitchShift(SAMPLING_RATE, random_step)


def frequency_masking(melspec, freq_mask_param=10, num_masks=1):
    """
    Apply frequency masking to a given mel-spectrogram in PyTorch.

    Args:
    - melspec (Tensor): Input mel-spectrogram (shape: [num_mels, time])
    - freq_mask_param (int): Maximum width of the frequency mask (in frequency bins)
    - num_masks (int): Number of frequency masks to apply

    Returns:
    - masked_melspec (Tensor): Mel-spectrogram with frequency masks applied
    """
    masked_melspec = melspec.clone()

    for i in range(num_masks):
        f = np.random.randint(0, freq_mask_param)
        f0 = np.random.randint(0, melspec.shape[0] - f)

        # Apply mask
        masked_melspec[f0 : f0 + f, :] = 0

    return masked_melspec


def apply_frequency_transforms(mel_spectrogram):
    mel_spectrogram_tensor = torch.Tensor(mel_spectrogram)
    freq_masked_mel_spec = frequency_masking(mel_spectrogram_tensor)
    freq_masked_mel_spec = torchaudio.transforms.FrequencyMasking(freq_mask_param=6)(
        freq_masked_mel_spec
    )
    return freq_masked_mel_spec.numpy()


def pitch_shifting(wave_file: str):
    """
    Applies pitch shifting to the input audio sample

    Args:
    - sample wav file name: The input audio waveform file name
    - n : no of augmented data

    Returns:
    - shifted_samples (List): The pitch-shifted audio waveform (shape: [1, num_samples])
    """

    waveform, sample_rate = torchaudio.load(wave_file, normalize=True)
    waveform_shift = pitch_transform(waveform)

    return waveform_shift.numpy()


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

    # flatten to 2D
    flattened_log_mel_features = log_mel_features.reshape(log_mel_features.shape[0], -1)

    # feature scaling
    scaled_log_mel_features = preprocessing.scale(flattened_log_mel_features, axis=1)

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

    def __init__(self, df, n=0, flattened=True, data_augmentation=False):
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
            if data_augmentation:
                changed_pitch_data = pitch_shifting(file_path)
                augmented_pitch_spectrogram = extract_melspectrogram(
                    changed_pitch_data, SAMPLING_RATE, 13
                )
                if n:
                    augmented_pitch_spectrogram = downsample_spectrogram(
                        augmented_pitch_spectrogram, n, flattened
                    )
                self.data.append(augmented_pitch_spectrogram)
                self.labels.append(row.label)

            if data_augmentation and spectrogram.ndim == 2:
                augmented_spectrogram = apply_frequency_transforms(spectrogram)
                if n:
                    augmented_spectrogram = downsample_spectrogram(
                        augmented_spectrogram, n, flattened=flattened
                    )
                self.data.append(augmented_spectrogram)
                self.labels.append(row.label)
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
