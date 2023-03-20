import torch
import torch.nn.functional as F


class CNN(torch.nn.Module):
    def __init__(
        self,
        num_classes=10,
    ):
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Conv1d(1, 64, 5)
        self.bn1 = torch.nn.BatchNorm1d(64)
        self.conv2 = torch.nn.Conv1d(64, 64, 5)
        self.bn2 = torch.nn.BatchNorm1d(64)
        self.conv3 = torch.nn.Conv1d(64, 64, 5)
        self.bn3 = torch.nn.BatchNorm1d(64)
        self.conv4 = torch.nn.Conv1d(64, 64, 5)
        self.bn4 = torch.nn.BatchNorm1d(64)
        self.conv5 = torch.nn.Conv1d(64, 128, 3)
        self.bn5 = torch.nn.BatchNorm1d(128)
        self.conv6 = torch.nn.Conv1d(128, 128, 3)
        self.bn6 = torch.nn.BatchNorm1d(128)
        self.conv7 = torch.nn.Conv1d(128, 128, 3)
        self.bn7 = torch.nn.BatchNorm1d(128)
        self.conv8 = torch.nn.Conv1d(128, 128, 3)
        self.bn8 = torch.nn.BatchNorm1d(128)
        self.conv9 = torch.nn.Conv1d(128, 256, 3)
        self.bn9 = torch.nn.BatchNorm1d(256)
        self.conv10 = torch.nn.Conv1d(256, 256, 3)
        self.bn10 = torch.nn.BatchNorm1d(256)
        self.conv11 = torch.nn.Conv1d(256, 256, 2)
        self.bn11 = torch.nn.BatchNorm1d(256)
        self.conv12 = torch.nn.Conv1d(256, 256, 2)
        self.fc1 = torch.nn.Linear(8704, 512)  # flatten
        self.linear = torch.nn.Linear(512, num_classes)

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
