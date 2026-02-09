import warnings
import torch
import torch.nn as nn
from nni.compression.utils.counter import count_flops_params


# -----------------------------------------------#
#             Original Network Architecture - Pre-improvement                  #
#          Based on YOLOv8 backbone and detection head             #
#         Detection head: p, left, right, prl, arv            #
# -----------------------------------------------#

class Conv(nn.Module):  # Convolution module, including Conv1d, BN and activation function
    def __init__(self, c1, c2, k=1, s=1, p=0, bn=True, act='silu'):
        """
        Initialize convolution module
        Parameters:
            c1: Number of input channels
            c2: Number of output channels
            k: Convolution kernel size
            s: Stride
            p: Padding
            bn: Whether to use BatchNorm
            act: Activation function type ('silu', 'sigmoid', 'softmax')
        """
        super().__init__()
        self.applyBN = bn
        self.cv = nn.Conv1d(c1, c2, k, s, p)  # 1D convolution layer
        self.bn = nn.BatchNorm1d(c2)  # Batch normalization layer
        self.act = self._get_activation_fn(act)  # Activation function

    @staticmethod
    def _get_activation_fn(act):
        """Get activation function based on type"""
        if act == 'softmax':
            return nn.Softmax(dim=-2)
        elif act == 'sigmoid':
            return nn.Sigmoid()
        else:  # Use SiLU activation function by default
            return nn.SiLU()

    def forward(self, x):
        """Forward propagation: Convolution -> Batch Normalization (optional) -> Activation function"""
        return self.act(self.bn(self.cv(x))) if self.applyBN else self.act(self.cv(x))


class Bottleneck(nn.Module):  # Bottleneck module for building deep networks
    def __init__(self, c1, c2, shortcut=False, k=3, s=1, p=1, e=0.5):
        """
        Initialize bottleneck module
        Parameters:
            c1: Number of input channels
            c2: Number of output channels
            shortcut: Whether to use residual connection
            k: Convolution kernel size
            s: Stride
            p: Padding
            e: Hidden layer channel scaling factor
        """
        super().__init__()
        c_ = int(c2 * e)  # Hidden layer channels (calculated by scaling factor)
        self.cv1 = Conv(c1, c_, k, s, p)  # First convolution layer (dimension reduction)
        self.cv2 = Conv(c_, c2, k, s, p)  # Second convolution layer (dimension recovery)
        self.add = shortcut and c1 == c2  # Residual connection condition: enabled and input/output channels are the same

    def forward(self, x):
        """Forward propagation: return x + convolution result if residual connection is enabled, otherwise return convolution result directly"""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f(nn.Module):  # Improved CSP module for feature extraction
    def __init__(self, c1, c2, n=1, shortcut=False, e=0.5):
        """
        Initialize C2f module
        Parameters:
            c1: Number of input channels
            c2: Number of output channels
            n: Number of Bottleneck modules
            shortcut: Whether to use residual connection
            e: Hidden layer channel scaling factor
        """
        super().__init__()
        self.c = int(c2 * e)  # Hidden layer channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)  # Initial convolution layer (double the number of channels)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # Final convolution layer (feature fusion)
        # Stack n Bottleneck modules
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, k=3, e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))  # Split features into two parts
        y.extend(m(y[-1]) for m in self.m)  # Apply Bottleneck to the second part and concatenate
        return self.cv2(torch.cat(y, 1))  # Fuse all features


class SPPF(nn.Module):  # Spatial Pyramid Pooling module (fast version)
    def __init__(self, c1, c2, k=5):
        """
        Initialize SPPF module
        Parameters:
            c1: Number of input channels
            c2: Number of output channels
            k: Max pooling kernel size
        """
        super().__init__()
        c_ = c1 // 2  # Hidden layer channels
        self.cv1 = Conv(c1, c_, 1, 1)  # Dimension reduction convolution
        self.cv2 = Conv(c_ * 4, c2, 1, 1)  # Fusion convolution (4 feature maps concatenated)
        self.m = nn.MaxPool1d(kernel_size=k, stride=1, padding=k // 2)  # Max pooling layer

    def forward(self, x):
        """Forward propagation: obtain multi-scale features through multiple pooling operations"""
        y = [self.cv1(x)]  # Initial features
        y.extend(self.m(y[-1]) for _ in range(3))  # 3 pooling operations to get multi-scale features
        return self.cv2(torch.cat(y, 1))  # Concatenate and fuse features


class Backbone(nn.Module):  # Backbone network (feature extraction part)
    def __init__(self, three_channel=False):
        """
        Initialize backbone network
        Parameters:
            three_channel: Whether to use 3-channel input (otherwise 1 channel)
        """
        super(Backbone, self).__init__()
        c_in = 3 if three_channel else 1  # Number of input channels
        # Stage sub-modules (downsampling + feature extraction)
        self.sub_module1 = nn.Sequential(
            Conv(c_in, 16, 3, 2, 1)  # Downsample to 1/2 scale
        )
        self.sub_module2 = nn.Sequential(
            Conv(16, 32, 3, 2, 1),  # Downsample to 1/4 scale
            C2f(32, 32, 1)  # C2f module for feature enhancement
        )
        self.sub_module3 = nn.Sequential(
            Conv(32, 64, 3, 2, 1),  # Downsample to 1/8 scale
            C2f(64, 64, 2)  # 2 Bottleneck modules
        )
        self.sub_module4 = nn.Sequential(
            Conv(64, 128, 3, 2, 1),  # Downsample to 1/16 scale
            C2f(128, 128, 2)  # 2 Bottleneck modules
        )
        self.sub_module5 = nn.Sequential(
            Conv(128, 256, 3, 2, 1),  # Downsample to 1/32 scale
            C2f(256, 256, 1)  # 1 Bottleneck module
        )
        self.SPPF = nn.Sequential(
            SPPF(256, 256)  # SPPF module (not used currently)
        )

    def forward(self, x):
        """Forward propagation: output feature maps of different scales"""
        x1 = self.sub_module1(x)  # 1/2 scale features
        x2 = self.sub_module2(x1)  # 1/4 scale features
        x3 = self.sub_module3(x2)  # 1/8 scale features
        x4 = self.sub_module4(x3)  # 1/16 scale features
        x5 = self.sub_module5(x4)  # 1/32 scale features
        # x5 = self.SPPF(x5)  # Optional SPPF processing
        return x1, x2, x3, x4, x5


class Upsample(nn.Module):  # Upsampling module
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x):
        """Upsample using nearest neighbor interpolation (scale factor 2)"""
        return nn.functional.interpolate(x, scale_factor=2, mode='nearest')


class Cat(nn.Module):  # Feature concatenation module
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x1, x2):
        """Concatenate two feature maps along the channel dimension"""
        return torch.cat((x1, x2), dim=1)


class Head_Detect_dfl(nn.Module):  # Detection head with DFL (Distribution Focal Loss)
    def __init__(self, c_in, c_out):
        """
        Initialize detection head (with DFL)
        Parameters:
            c_in: Number of input channels
            c_out: Number of output channels
        """
        super(Head_Detect_dfl, self).__init__()
        self.cv = nn.Sequential(
            Conv(c_in, c_in, 3, 1, 1),  # Convolution for feature enhancement
            nn.Conv1d(c_in, c_out, 1, 1, 0)  # Output layer
        )

    def forward(self, x):
        """Forward propagation: output processed detection results (with softmax)"""
        x = self.cv(x)
        n = (x.size(1) - 1) // 4  # Calculate channels for each branch
        # Split into confidence and 4 coordinate branches
        x_, x1, x2, x3, x4 = torch.split(x, [1, n, n, n, n], dim=1)
        x_ = nn.functional.sigmoid(x_)  # Confidence (sigmoid activation)
        # Coordinate branches use softmax (DFL characteristic)
        x1 = nn.functional.softmax(x1, dim=-2)
        x2 = nn.functional.softmax(x2, dim=-2)
        x3 = nn.functional.softmax(x3, dim=-2)
        x4 = nn.functional.softmax(x4, dim=-2)
        return torch.cat((x_, x1, x2, x3, x4), dim=1)  # Concatenate results


class Head_Detect(nn.Module):  # Basic detection head (without DFL)
    def __init__(self, c_in, c_out):
        """
        Initialize basic detection head
        Parameters:
            c_in: Number of input channels
            c_out: Number of output channels
        """
        super(Head_Detect, self).__init__()
        self.cv = nn.Sequential(
            Conv(c_in, c_in, 3, 1, 1),  # Convolution for feature enhancement
            nn.Conv1d(c_in, c_out, 1, 1, 0)  # Output layer
        )

    def forward(self, x):
        """Forward propagation: output processed detection results (with sigmoid)"""
        x = self.cv(x)
        n = (x.size(1) - 1) // 4  # Calculate channels for each branch
        # Split into confidence and 4 coordinate branches
        x_, x1, x2, x3, x4 = torch.split(x, [1, n, n, n, n], dim=1)
        x_ = nn.functional.sigmoid(x_)  # Confidence (sigmoid activation)
        # Coordinate branches use sigmoid (direct regression)
        x1 = nn.functional.sigmoid(x1)
        x2 = nn.functional.sigmoid(x2)
        x3 = nn.functional.sigmoid(x3)
        x4 = nn.functional.sigmoid(x4)
        return torch.cat((x_, x1, x2, x3, x4), dim=1)  # Concatenate results


class Net(nn.Module):  # Complete network (backbone + neck + detection head)
    def __init__(self, three_channel=False, three_head=True, dfl=True):
        """
        Initialize complete network
        Parameters:
            three_channel: Whether to use 3-channel input
            three_head: Whether to use 3 detection heads (multi-scale detection)
            dfl: Whether to use DFL (Distribution Focal Loss)
        """
        super(Net, self).__init__()
        self.backbone = Backbone(three_channel=three_channel)  # Backbone network
        self.Upsample = Upsample()  # Upsampling module
        self.Concat = Cat()  # Feature concatenation module
        # Neck modules (feature fusion)
        self.C2f1 = C2f(384, 128, 1)
        self.C2f2 = C2f(192, 64, 1)
        self.Conv1 = Conv(64, 64, 3, 2, 1)  # Downsampling convolution
        self.C2f3 = C2f(192, 128, 1)
        self.Conv2 = Conv(128, 128, 3, 2, 1)  # Downsampling convolution
        self.C2f4 = C2f(384, 256, 1)

        self.three_head = three_head  # Whether to use multi-detection heads
        # Initialize detection heads according to configuration
        if three_head:
            if dfl:
                self.Head_Detect1 = Head_Detect_dfl(64, 65)
                self.Head_Detect2 = Head_Detect_dfl(128, 65)
                self.Head_Detect3 = Head_Detect_dfl(256, 65)
            else:
                self.Head_Detect1 = Head_Detect(64, 5)
                self.Head_Detect2 = Head_Detect(128, 5)
                self.Head_Detect3 = Head_Detect(256, 5)
        else:
            if dfl:
                self.Head_Detect = Head_Detect_dfl(64, 65)
            else:
                self.Head_Detect = Head_Detect(64, 5)

    def forward(self, x):
        """Forward propagation: complete network inference"""
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # Ignore warnings
            # Get feature maps output by backbone network (take last three scales)
            _, _, x4, x6, x9 = self.backbone(x)
            # Neck feature fusion (upsampling + concatenation + C2f)
            x12 = self.C2f1(self.Concat(self.Upsample(x9), x6))
            x15 = self.C2f2(self.Concat(self.Upsample(x12), x4))

            if self.three_head:  # Multi-detection head mode
                x18 = self.C2f3(self.Concat(self.Conv1(x15), x12))
                x21 = self.C2f4(self.Concat(self.Conv2(x18), x9))
                # Three detection heads output separately
                y1, y2, y3 = self.Head_Detect1(x15), self.Head_Detect2(x18), self.Head_Detect3(x21)
                # Concatenate and adjust dimension order
                y = torch.cat((y1, y2, y3), dim=-1).permute(0, 2, 1)
            else:  # Single detection head mode
                y = self.Head_Detect(x15).permute(0, 2, 1)
        return y


class CNNNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.cv1 = nn.Sequential(
            Conv(1, 64, 3, 1, 1),
            nn.MaxPool1d(3, 2, 1)
        )
        self.cv2 = nn.Sequential(
            Conv(64, 128, 3, 1, 1),
            nn.MaxPool1d(3, 1, 1)
        )
        self.cv3 = nn.Sequential(
            Conv(128, 128, 3, 1, 1),
            nn.MaxPool1d(3, 1, 1)
        )
        self.cv4 = nn.Sequential(
            Conv(128, 256, 3, 1, 1),
            nn.MaxPool1d(3, 2, 1)
        )
        self.cv5 = nn.Sequential(
            Conv(256, 256, 3, 1, 1),
            nn.MaxPool1d(3, 2, 1)
        )
        self.cv = Conv(256, 256, 3, 1, 1)
        self.head = nn.Conv1d(256, 5, 1, 1, 0)

    def forward(self, x):
        """Forward propagation: output processed detection results (with sigmoid)"""
        x = self.cv1(x)
        x = self.cv2(x)
        x = self.cv3(x)
        x = self.cv4(x)
        x = self.cv5(x)
        x = self.cv(x)
        x = self.head(x)
        x_, x1, x2, x3, x4 = torch.split(x, [1, 1, 1, 1, 1], dim=1)
        x_ = nn.functional.sigmoid(x_)
        x1 = nn.functional.sigmoid(x1)
        x2 = nn.functional.sigmoid(x2)
        x3 = nn.functional.sigmoid(x3)
        x4 = nn.functional.sigmoid(x4)
        y = torch.cat((x_, x1, x2, x3, x4), dim=1).permute(0, 2, 1)
        return y


if __name__ == "__main__":
    # Test output shapes of different network configurations
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 8 network configurations (combinations of channels, number of detection heads, and DFL)
    param = {
        1: {'three_head': False, 'dfl': False, 'three_channel': False},
        2: {'three_head': False, 'dfl': False, 'three_channel': True},
        3: {'three_head': False, 'dfl': True, 'three_channel': False},
        4: {'three_head': False, 'dfl': True, 'three_channel': True},
        5: {'three_head': True, 'dfl': False, 'three_channel': False},
        6: {'three_head': True, 'dfl': False, 'three_channel': True},
        7: {'three_head': True, 'dfl': True, 'three_channel': False},
        8: {'three_head': True, 'dfl': True, 'three_channel': True}
    }

    for i in range(8):
        selected_number = i + 1
        params = param[selected_number]
        net = Net(**params).to(device)
        x = torch.randn(10, 3, 1024).to(device) if i % 2 else torch.randn(10, 1, 1024).to(device)
        _, _, _ = count_flops_params(net, x)
        print(net(x).shape)

    x = torch.randn(10, 1, 1024).to(device)
    net = CNNNet().to(device)
    _, _, _ = count_flops_params(net, x)
    print(net(x).shape)