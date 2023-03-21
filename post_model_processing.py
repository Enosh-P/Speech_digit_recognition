import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE


def t_sne_evaluation(model, dataloader, device):
    model.eval()

    features = []
    labels = []

    for spectrogram, target in dataloader:
        with torch.no_grad():
            output = model(spectrogram.to(device), use_last_layer=False)
        features.append(output.cpu().numpy())
        labels.append(target.numpy())

    features = np.concatenate(features)
    labels = np.concatenate(labels)

    tsne = TSNE(n_components=2, random_state=0)
    features_tsne = tsne.fit_transform(features)
    plt.figure(figsize=(10, 10))
    for i in range(10):
        plt.scatter(
            features_tsne[labels == i, 0], features_tsne[labels == i, 1], label=str(i)
        )
    plt.legend()
    plt.show()
