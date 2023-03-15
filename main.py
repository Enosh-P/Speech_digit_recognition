import pandas as pd
from torch.utils.data import DataLoader
from preprocessor import SpectrogramDataset


def split_data(audio_df):
    """ Split the data into a train, valid and test set"""
    train_df = audio_df.loc[sdr_df['split'] == 'TRAIN']
    test_df = audio_df.loc[sdr_df['split'] == 'TEST']
    val_df = audio_df.loc[sdr_df['split'] == 'DEV']

    print(f"# Train Size: {len(train_df)}")
    print(f"# Valid Size: {len(val_df)}")
    print(f"# Test Size: {len(test_df)}")

    return train_df, val_df, test_df


def build_training_data(train_df, valid_df, test_df, train_batch_size, val_batch_size, test_batch_size):
    """ Covert the audio samples into training data"""

    train_data = SpectrogramDataset(train_df)
    train_pr = DataLoader(train_data, batch_size=train_batch_size, shuffle=True, num_workers=2)

    valid_data = SpectrogramDataset(valid_df)
    valid_pr = DataLoader(valid_data, batch_size=val_batch_size, shuffle=True, num_workers=2)

    test_data = SpectrogramDataset(test_df)
    test_pr = DataLoader(test_data, batch_size=test_batch_size, shuffle=True, num_workers=2)

    return train_pr, valid_pr, test_pr


if __name__ == '__main__':

    sdr_df = pd.read_csv('SDR_metadata.tsv', sep='\t', header=0, index_col='Unnamed: 0')

    print('Train Test Split!')
    train, valid, test = split_data(sdr_df)

    print("Preparing Data!")
    train_loader, valid_loader, test_loader = build_training_data(train, valid, test, 32, 32, 32)


