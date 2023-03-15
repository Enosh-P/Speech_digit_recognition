import torch


class CNN(torch.nn.Module):
    def __init__(
        self,
        channels,
        conv_kernels,
        conv_strides,
        conv_padding,
        pool_padding,
        num_classes=10,
    ):
        assert (
            len(conv_kernels) == len(channels) == len(conv_strides) == len(conv_padding)
        )
        super(CNN, self).__init__()
        # create conv blocks
        self.conv_blocks = torch.nn.ModuleList()
        prev_channel = 1
        for i in range(len(channels)):
            # add stacked conv layer
            block = []
            for j, conv_channel in enumerate(channels[i]):
                block.append(
                    torch.nn.Conv1d(
                        in_channels=prev_channel,
                        out_channels=conv_channel,
                        kernel_size=conv_kernels[i],
                        stride=conv_strides[i],
                        padding=conv_padding[i],
                    )
                )
                prev_channel = conv_channel
                # add batch norm layer
                block.append(torch.nn.BatchNorm1d(prev_channel))
                # adding ReLU
                block.append(torch.nn.ReLU())
            self.conv_blocks.append(torch.nn.Sequential(*block))

        # create pool blocks
        self.pool_blocks = torch.nn.ModuleList()
        for i in range(len(pool_padding)):
            # adding Max Pool (drops dims by a factor of 4)
            self.pool_blocks.append(
                torch.nn.MaxPool1d(kernel_size=4, stride=4, padding=pool_padding[i])
            )

        # global pooling
        self.global_pool = torch.nn.AdaptiveAvgPool1d(1)
        self.linear = torch.nn.Linear(prev_channel, num_classes)

    def forward(self, inwav):
        for i in range(len(self.conv_blocks)):
            # apply conv layer
            inwav = self.conv_blocks[i](inwav)
            # apply max_pool
            if i < len(self.pool_blocks):
                inwav = self.pool_blocks[i](inwav)
        # apply global pooling
        out = self.global_pool(inwav).squeeze()
        out = self.linear(out)
        return out.squeeze()
