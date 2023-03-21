import os
import time

import pandas as pd
import torch

from torch.utils.data import DataLoader
from tqdm import tqdm
from models import CNN, RNNModel
from preprocessor import SpectrogramDataset
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


def test(model, data_loader, verbose=False, verbose_report=False):
    """Measures the accuracy of a model on a data set."""
    # Make sure the model is in evaluation mode.
    model.eval()
    correct = 0
    y_prediction = None
    # We do not need to maintain intermediate activations while testing.
    with torch.no_grad():
        # Loop over test data.
        for features, target in tqdm(
            data_loader, total=len(data_loader.batch_sampler), desc="Testing"
        ):
            # Forward pass.
            output = model(features.to(device))
            # Get the label corresponding to the highest predicted probability.
            pred = output.argmax(dim=1, keepdim=True)
            if y_prediction is None:
                y_prediction = pred.squeeze()
                y_target = target
            else:
                y_prediction = torch.cat((y_prediction, pred.squeeze()), dim=0)
                y_target = torch.cat((y_target, target), dim=0)

            # Count number of correct predictions.
            correct += pred.cpu().eq(target.view_as(pred)).sum().item()
    # Print test accuracy.
    percent = 100.0 * correct / len(data_loader.sampler)
    if verbose:
        print("----- Model Evaluation -----")
        print(f"Test accuracy: {correct} / {len(data_loader.sampler)} ({percent:.0f}%)")
    if verbose_report:
        y_target = y_target.cpu()
        y_prediction = y_prediction.cpu()
        cm_vr = confusion_matrix(y_target, y_prediction)
        accuracy_vr = accuracy_score(y_target, y_prediction)
        report_vr = classification_report(y_target, y_prediction)
        print(f"The Confusion matrix Test set:\n{cm_vr}")
        print(f"The Accuracy for Test set:\n{accuracy_vr}")
        print(f"The Report for Test set:\n{report_vr}")

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
    train_df,
    valid_df,
    test_df,
    train_batch_size,
    val_batch_size,
    test_batch_size,
    n=0,
    flattened=True,
):
    """Covert the audio samples into training data"""

    train_data = SpectrogramDataset(train_df, n=n, flattened=flattened)
    train_pr = DataLoader(
        train_data, batch_size=train_batch_size, shuffle=True, num_workers=2
    )

    valid_data = SpectrogramDataset(valid_df, n=n, flattened=flattened)
    valid_pr = DataLoader(
        valid_data, batch_size=val_batch_size, shuffle=True, num_workers=2
    )

    test_data = SpectrogramDataset(test_df, n=n, flattened=flattened)
    test_pr = DataLoader(
        test_data, batch_size=test_batch_size, shuffle=True, num_workers=2
    )

    return train_pr, valid_pr, test_pr


def start(cnn=False):
    if cnn:
        # normalize data with n=15 for Deep CNN model
        print("Preparing Data for Deep CNN!")
        train_loader, valid_loader, test_loader = build_training_data(
            train_df, valid_df, test_df, 32, 32, 32, n=15
        )

        CnnModel = CNN()

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
        test(CnnModel, test_loader, verbose=True, verbose_report=True)

        plt.plot(accs)
        plt.title("Validation Accuracy CNN")
        plt.xlabel("# Epochs")
        plt.ylabel("Accuracy (%)")
        plt.show()
    else:

        print("Preparing Data for Audio Transformer!")
        at_train_loader, at_valid_loader, at_test_loader = build_training_data(
            train_df, valid_df, test_df, 32, 32, 32, 15, flattened=False
        )

        AudioRNNModel = RNNModel()
        ARCriterion = torch.nn.CrossEntropyLoss()
        AROptimizer = torch.optim.Adam(AudioRNNModel.parameters(), weight_decay=1e-4)

        ARaccs = train(
            AudioRNNModel,
            ARCriterion,
            at_train_loader,
            at_valid_loader,
            AROptimizer,
            num_epochs=100,
        )

        test(AudioRNNModel, at_test_loader, verbose=True, verbose_report=True)

        plt.plot(ARaccs)
        plt.title("Validation Accuracy RNN")
        plt.xlabel("# Epochs")
        plt.ylabel("Accuracy (%)")
        plt.show()


if __name__ == "__main__":

    random_seed = 42
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device used: {device}")

    sdr_df = pd.read_csv("SDR_metadata.tsv", sep="\t", header=0, index_col="Unnamed: 0")

    print("Train Test Split!")
    train_df, valid_df, test_df = split_data(sdr_df)

    start()
