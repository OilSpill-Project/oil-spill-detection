import torch
import torch.nn as nn


# -----------------------------
# Double Convolution Block
# -----------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


# -----------------------------
# Attention Gate
# -----------------------------
class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()

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


# -----------------------------
# Attention U-Net++
# -----------------------------
class AttentionUNetPlusPlus(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.enc1 = DoubleConv(1, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = DoubleConv(128, 256)

        # Decoder (Nested Connections)
        self.up2_1 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.att2_1 = AttentionGate(128, 128, 64)
        self.dec2_1 = DoubleConv(256, 128)

        self.up1_1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.att1_1 = AttentionGate(64, 64, 32)
        self.dec1_1 = DoubleConv(128, 64)

        # Nested layer
        self.up1_2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.att1_2 = AttentionGate(64, 64, 32)
        self.dec1_2 = DoubleConv(192, 64)

        self.out = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):

        # Encoder
        x1_0 = self.enc1(x)
        x2_0 = self.enc2(self.pool1(x1_0))
        x3_0 = self.enc3(self.pool2(x2_0))

        # First decoder stage
        x2_1_up = self.up2_1(x3_0)
        x2_0_att = self.att2_1(x2_1_up, x2_0)
        x2_1 = self.dec2_1(torch.cat([x2_1_up, x2_0_att], dim=1))

        x1_1_up = self.up1_1(x2_1)
        x1_0_att = self.att1_1(x1_1_up, x1_0)
        x1_1 = self.dec1_1(torch.cat([x1_1_up, x1_0_att], dim=1))

        # Nested connection
        x1_2_up = self.up1_2(x2_1)
        x1_0_att2 = self.att1_2(x1_2_up, x1_0)
        x1_2 = self.dec1_2(torch.cat([x1_2_up, x1_1, x1_0_att2], dim=1))

        return self.out(x1_2)


# -----------------------------
# Shape Verification
# -----------------------------
if __name__ == "__main__":
    model = AttentionUNetPlusPlus()
    x = torch.randn(1, 1, 256, 256)
    y = model(x)

    print("Input shape :", x.shape)
    print("Output shape:", y.shape)
