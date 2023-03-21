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
        self.norm = nn.Softmax(dim=1)

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

    def forward(self, x, use_last_layer=False):
        # Reshape the tensor to have shape [batch_size, input_channels, signal_length]
        x = x.view(x.shape[0], 1, x.shape[1])
        x = self.convs(x)
        x = F.relu(self.fc1(x))
        x = self.linear(x)
        if use_last_layer:
            x = self.norm(x)
        return x


class RNNModel(nn.Module):
    def __init__(self):
        super(RNNModel, self).__init__()
        self.rnn = nn.LSTM(input_size=13, hidden_size=256, batch_first=True)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 10)
        self.norm = nn.Softmax(dim=1)

    def forward(self, input, use_last_layer=True):
        output, _ = self.rnn(input)
        output = self.fc1(output[:, -1])
        output = self.fc2(output)
        if use_last_layer:
            output = self.norm(output)
        return output
