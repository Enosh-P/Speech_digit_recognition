import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(
        self,
        num_classes=10,
    ):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, 5)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 64, 5)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 64, 5)
        self.bn3 = nn.BatchNorm1d(64)
        self.conv4 = nn.Conv1d(64, 64, 5)
        self.bn4 = nn.BatchNorm1d(64)
        self.conv5 = nn.Conv1d(64, 128, 3)
        self.bn5 = nn.BatchNorm1d(128)
        self.conv6 = nn.Conv1d(128, 128, 3)
        self.bn6 = nn.BatchNorm1d(128)
        self.conv7 = nn.Conv1d(128, 128, 3)
        self.bn7 = nn.BatchNorm1d(128)
        self.conv8 = nn.Conv1d(128, 128, 3)
        self.bn8 = nn.BatchNorm1d(128)
        self.conv9 = nn.Conv1d(128, 256, 3)
        self.bn9 = nn.BatchNorm1d(256)
        self.conv10 = nn.Conv1d(256, 256, 3)
        self.bn10 = nn.BatchNorm1d(256)
        self.conv11 = nn.Conv1d(256, 256, 2)
        self.bn11 = nn.BatchNorm1d(256)
        self.conv12 = nn.Conv1d(256, 256, 2)
        self.fc1 = nn.Linear(8704, 512)  # flatten
        self.linear = nn.Linear(512, num_classes)

    # all conv blocks
    def convs(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.max_pool1d(x, kernel_size=2)
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = F.relu(self.bn7(self.conv7(x)))
        x = F.relu(self.bn8(self.conv8(x)))
        x = F.max_pool1d(x, kernel_size=2)
        x = F.relu(self.bn9(self.conv9(x)))
        x = F.relu(self.bn10(self.conv10(x)))
        x = F.relu(self.bn11(self.conv11(x)))
        x = self.conv12(x)
        x = x.view(x.size(0), -1)

        return x

    def forward(self, x):
        x = self.convs(x)
        x = F.relu(self.fc1(x))
        x = self.linear(x)
        return x


class TransformerModel(nn.Module):
    def __init__(
        self,
        n_classes,
        input_shape,
        d_model=256,
        n_heads=4,
        ff_dim=1024,
        dropout_rate=0.2,
    ):
        super(TransformerModel, self).__init__()

        # Resize the input to a fixed shape
        self.resize_layer = nn.Sequential(nn.Linear(input_shape[1], d_model))

        # Normalize the input
        self.normalize_layer = nn.LayerNorm(d_model)

        # Add positional encoding
        self.pos_encoding_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads
        )

        # Add dropout
        self.dropout_layer = nn.Dropout(dropout_rate)

        # Transformer layers
        self.transformer_layer1 = nn.TransformerEncoder(
            self.pos_encoding_layer, num_layers=4
        )

        # Classification layer
        self.classification_layer = nn.Sequential(
            nn.Linear(d_model, ff_dim), nn.ReLU(), nn.Linear(ff_dim, n_classes)
        )

    def forward(self, x):
        # Resize the input to a fixed shape
        x = self.resize_layer(x)

        # Normalize the input
        x = self.normalize_layer(x)

        # Add positional encoding
        x = x.transpose(0, 1)
        x = self.pos_encoding_layer(x)
        x = x.transpose(0, 1)

        # Add dropout
        x = self.dropout_layer(x)

        # Transformer layers
        x = self.transformer_layer1(x)

        # Classification layer
        x = x.mean(dim=0)
        x = self.classification_layer(x)

        return x
