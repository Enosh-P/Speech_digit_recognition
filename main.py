import os
import time

import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from models import CNN
from preprocessor import SpectrogramDataset


def test(model, data_loader, verbose=False):
    """Measures the accuracy of a model on a data set."""
    # Make sure the model is in evaluation mode.
    model.eval()
    correct = 0
    # We do not need to maintain intermediate activations while testing.
    with torch.no_grad():
        # Loop over test data.
        for features, target in tqdm(
            data_loader, total=len(data_loader.batch_sampler), desc="Testing"
        ):
            # Reshape the tensor to have shape [batch_size, input_channels, signal_length]
            features = features.view(features.shape[0], 1, features.shape[1])
            # Forward pass.
            output = model(features.to(device))
            # Get the label corresponding to the highest predicted probability.
            pred = output.argmax(dim=1, keepdim=True)
            # Count number of correct predictions.
            correct += pred.cpu().eq(target.view_as(pred)).sum().item()
    # Print test accuracy.
    percent = 100.0 * correct / len(data_loader.sampler)
    if verbose:
        print("----- Model Evaluation -----")
        print(f"Test accuracy: {correct} / {len(data_loader.sampler)} ({percent:.0f}%)")
    return percent


# using glorot initialization
def init_weights(m):
    if isinstance(m, torch.nn.Conv1d):
        torch.nn.init.xavier_uniform_(m.weight.data)


def train(model, criterion, train_loader, validation_loader, optimizer, num_epochs):
    """Simple training loop for a PyTorch model."""

    # Move model to the device (CPU or GPU).
    model.to(device)

    accs = []
    # Exponential moving average of the loss.
    ema_loss = None

    #     print('----- Training Loop -----')
    # Loop over epochs.
    for epoch in range(num_epochs):
        tick = time.time()
        model.train()
        # Loop over data.
        for batch_idx, (features, target) in tqdm(
            enumerate(train_loader),
            total=len(train_loader.batch_sampler),
            desc="Training",
        ):
            # Reshape the tensor to have shape [batch_size, input_channels, signal_length]
            features = features.view(features.shape[0], 1, features.shape[1])

            # Forward pass.
            output = model(features.to(device))
            loss = criterion(output.to(device), target.to(device))

            # Backward pass.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if ema_loss is None:
                ema_loss = loss.item()
            else:
                ema_loss += (loss.item() - ema_loss) * 0.01

        tock = time.time()
        acc = test(model, validation_loader, verbose=True)
        accs.append(acc)
        # Print out progress the end of epoch.
        print(
            "Epoch: {} \tLoss: {:.6f} \t Time taken: {:.6f} seconds".format(
                epoch + 1, ema_loss, tock - tick
            ),
        )
        torch.save(model.state_dict(), f"saved_models/model_{epoch}.ckpt")
        print("Model Saved!")
        if os.path.isfile(f"saved_models/model_{epoch - 1}.ckpt"):
            os.remove(f"saved_models/model_{epoch - 1}.ckpt")
    return accs


def split_data(audio_df):
    """Split the data into a train, valid and test set"""
    train_df = audio_df.loc[sdr_df["split"] == "TRAIN"]
    test_df = audio_df.loc[sdr_df["split"] == "TEST"]
    val_df = audio_df.loc[sdr_df["split"] == "DEV"]

    print(f"# Train Size: {len(train_df)}")
    print(f"# Valid Size: {len(val_df)}")
    print(f"# Test Size: {len(test_df)}")

    return train_df, val_df, test_df


def build_training_data(
    train_df, valid_df, test_df, train_batch_size, val_batch_size, test_batch_size
):
    """Covert the audio samples into training data"""

    train_data = SpectrogramDataset(train_df, n=15)
    train_pr = DataLoader(
        train_data, batch_size=train_batch_size, shuffle=True, num_workers=2
    )

    valid_data = SpectrogramDataset(valid_df, n=15)
    valid_pr = DataLoader(
        valid_data, batch_size=val_batch_size, shuffle=True, num_workers=2
    )

    test_data = SpectrogramDataset(test_df, n=15)
    test_pr = DataLoader(
        test_data, batch_size=test_batch_size, shuffle=True, num_workers=2
    )

    return train_pr, valid_pr, test_pr


if __name__ == "__main__":

    sdr_df = pd.read_csv("SDR_metadata.tsv", sep="\t", header=0, index_col="Unnamed: 0")

    print("Train Test Split!")
    train_df, valid_df, test_df = split_data(sdr_df)

    print("Preparing Data!")
    train_loader, valid_loader, test_loader = build_training_data(
        train_df, valid_df, test_df, 32, 32, 32
    )

    CnnModel = CNN()

    random_seed = 42
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device used: {device}")

    print("Num Parameters:", sum([p.numel() for p in CnnModel.parameters()]))
    CnnModel.apply(init_weights)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(CnnModel.parameters(), weight_decay=1e-4)

    num_epochs = 100
    accs = train(
        CnnModel,
        criterion,
        train_loader,
        valid_loader,
        optimizer,
        num_epochs=num_epochs,
    )
    acc = test(CnnModel, test_loader, verbose=True)

    plt.plot(accs)
    plt.title("Validation Accuracy")
    plt.xlabel("# Epochs")
    plt.ylabel("Accuracy (%)")
    plt.show()
