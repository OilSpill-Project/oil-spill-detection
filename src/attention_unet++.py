import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------
# Convolution Block
# -----------------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


# -----------------------------------
# Attention Gate
# -----------------------------------
class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


# -----------------------------------
# Attention U-Net++
# -----------------------------------
class AttentionUNet++(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(AttentionUNet++, self).__init__()

        filters = [64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)

        # Encoder
        self.conv0_0 = ConvBlock(in_channels, filters[0])
        self.conv1_0 = ConvBlock(filters[0], filters[1])
        self.conv2_0 = ConvBlock(filters[1], filters[2])
        self.conv3_0 = ConvBlock(filters[2], filters[3])

        # Decoder (Nested)
        self.conv0_1 = ConvBlock(filters[0] + filters[1], filters[0])
        self.conv1_1 = ConvBlock(filters[1] + filters[2], filters[1])
        self.conv2_1 = ConvBlock(filters[2] + filters[3], filters[2])

        self.conv0_2 = ConvBlock(filters[0]*2 + filters[1], filters[0])
        self.conv1_2 = ConvBlock(filters[1]*2 + filters[2], filters[1])

        self.conv0_3 = ConvBlock(filters[0]*3 + filters[1], filters[0])

        # Attention Gates
        self.att1 = AttentionBlock(filters[1], filters[0], filters[0]//2)
        self.att2 = AttentionBlock(filters[2], filters[1], filters[1]//2)
        self.att3 = AttentionBlock(filters[3], filters[2], filters[2]//2)

        # Final output
        self.final = nn.Conv2d(filters[0], out_channels, kernel_size=1)


    def forward(self, x):

        # Encoder
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))

        # Attention + Decoder Level 1
        x2_0_att = self.att3(x3_0, x2_0)
        x2_1 = self.conv2_1(torch.cat([x2_0_att, F.interpolate(x3_0, scale_factor=2)], 1))

        x1_0_att = self.att2(x2_0, x1_0)
        x1_1 = self.conv1_1(torch.cat([x1_0_att, F.interpolate(x2_0, scale_factor=2)], 1))

        x0_0_att = self.att1(x1_0, x0_0)
        x0_1 = self.conv0_1(torch.cat([x0_0_att, F.interpolate(x1_0, scale_factor=2)], 1))

        # Decoder Level 2
        x1_2 = self.conv1_2(torch.cat([
            x1_0, x1_1, F.interpolate(x2_1, scale_factor=2)
        ], 1))

        x0_2 = self.conv0_2(torch.cat([
            x0_0, x0_1, F.interpolate(x1_1, scale_factor=2)
        ], 1))

        # Decoder Level 3
        x0_3 = self.conv0_3(torch.cat([
            x0_0, x0_1, x0_2, F.interpolate(x1_2, scale_factor=2)
        ], 1))

        output = self.final(x0_3)

        return output
