import torch.nn as nn

class ConvBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(2, 3, 3),
                 stride=(2, 2, 2), padding=(0, 1, 1)):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.skip = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride),
            # nn.BatchNorm3d(out_channels)
        )

    def forward(self, x):
        identity = self.skip(x)
        out = self.conv(x)
        out = self.bn(out)

        out += identity
        out = self.relu(out)
        return out


class ConvBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=2, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.skip = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
        )

    def forward(self, x):
        identity = self.skip(x)
        out = self.conv(x)
        out = self.bn(out)

        out += identity
        out = self.relu(out)
        return out


class ResidualConv2d(nn.Module):

    def __init__(self, hidden_in, hidden_out, kernel, padding, dilation):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(hidden_in, hidden_out, kernel, padding=padding, dilation=dilation),
            nn.BatchNorm2d(hidden_out),
            nn.ReLU(),
            nn.Conv2d(hidden_out, hidden_out, kernel, padding=padding, dilation=dilation),
            nn.BatchNorm2d(hidden_out)
        )
        self.relu = nn.ReLU()
        self.downscale = nn.Sequential(
            nn.Conv2d(hidden_in, hidden_out, kernel, padding=padding)
        )
    
    def forward(self, x):
        residual = self.downscale(x)
        output = self.main(x)
        return self.relu(output+residual)    


class HiCnMaskEncoder3D(nn.Module):
    def __init__(self):
        super().__init__()
        self._3d = nn.Sequential(
            ConvBlock3D(1, 16, kernel_size=(2, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
            ConvBlock3D(16, 32, kernel_size=(2, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
            ConvBlock3D(32, 64, kernel_size=(2, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
            ConvBlock3D(64, 128, kernel_size=(2, 3, 3), stride=(2, 2, 2), padding=(0, 1, 1)),
        )
        self._2d = nn.Sequential(
            ConvBlock2D(128, 256, kernel_size=3, stride=2, padding=1),
            ConvBlock2D(256, 512, kernel_size=3, stride=2, padding=1),
            ConvBlock2D(512, 1024, kernel_size=3, stride=2, padding=1),
        )
        self.final = nn.Sequential(
            nn.Conv2d(1024, 2048, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Dropout(0.1),
            nn.Linear(2048, 768),
            nn.ReLU(),
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # (B, 2, H, W) → (B, 1, 2, H, W)
        x = self._3d(x)
        x = x.squeeze(2)  # (B, C, 1, H, W) → (B, C, H, W)
        x = self._2d(x)
        x = self.final(x)
        return x
