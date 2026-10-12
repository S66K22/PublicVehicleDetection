# ```python
# import torch
# import torch.nn as nn


# class SeparableConv2d(nn.Module):

#     def __init__(
#         self,
#         in_channels,
#         out_channels,
#         kernel_size,
#         stride=1,
#         padding=0,
#     ):
#         super().__init__()

#         self.depthwise_conv = nn.Conv2d(
#             in_channels,
#             in_channels,
#             kernel_size,
#             stride=stride,
#             padding=padding,
#             groups=in_channels,
#         )

#         self.pointwise_conv = nn.Conv2d(
#             in_channels,
#             out_channels,
#             kernel_size=1,
#             stride=1,
#             padding=0,
#         )

#     def forward(self, inputs):
#         x = self.depthwise_conv(inputs)
#         x = self.pointwise_conv(x)
#         return x


# class SEBlock(nn.Module):

#     def __init__(self, channels, reduction=16):
#         super().__init__()

#         self.squeeze = nn.AdaptiveAvgPool2d(1)

#         self.excitation = nn.Sequential(
#             nn.Linear(
#                 channels,
#                 channels // reduction,
#                 bias=False,
#             ),
#             nn.ReLU(inplace=True),
#             nn.Linear(
#                 channels // reduction,
#                 channels,
#                 bias=False,
#             ),
#             nn.Sigmoid(),
#         )

#     def forward(self, x):

#         b, c, _, _ = x.shape

#         # Squeeze:
#         # [B, C, H, W] -> [B, C]
#         y = self.squeeze(x).view(b, c)

#         # Excitation:
#         # [B, C] -> [B, C]
#         y = self.excitation(y)

#         # Reshape:
#         # [B, C] -> [B, C, 1, 1]
#         y = y.view(b, c, 1, 1)

#         # Channel-wise recalibration
#         return x * y


# class XceptionBlock(nn.Module):

#     def __init__(
#         self,
#         in_channels,
#         out_channels,
#         stride=1,
#         reduction=16,
#     ):
#         super().__init__()

#         self.conv = nn.Sequential(

#             # First separable convolution
#             SeparableConv2d(
#                 in_channels,
#                 out_channels,
#                 kernel_size=3,
#                 stride=stride,
#                 padding=1,
#             ),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),

#             # Second separable convolution
#             SeparableConv2d(
#                 out_channels,
#                 out_channels,
#                 kernel_size=3,
#                 stride=1,
#                 padding=1,
#             ),
#             nn.BatchNorm2d(out_channels),
#         )

#         # Squeeze-and-Excitation
#         self.se = SEBlock(
#             out_channels,
#             reduction=reduction,
#         )

#         # Residual shortcut
#         if in_channels != out_channels or stride != 1:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(
#                     in_channels,
#                     out_channels,
#                     kernel_size=1,
#                     stride=stride,
#                 ),
#                 nn.BatchNorm2d(out_channels),
#             )
#         else:
#             self.shortcut = nn.Identity()

#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, x):

#         identity = self.shortcut(x)

#         x = self.conv(x)

#         # Channel attention
#         x = self.se(x)

#         # Residual connection
#         x = x + identity

#         x = self.relu(x)

#         return x


# class SmallXceptionSE(nn.Module):

#     def __init__(
#         self,
#         num_classes,
#         dropout=0.2,
#         reduction=16,
#     ):
#         super().__init__()

#         self.features = nn.Sequential(

#             # Stem
#             nn.Conv2d(
#                 3,
#                 32,
#                 kernel_size=7,
#                 stride=2,
#                 padding=3,
#             ),
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),

#             nn.MaxPool2d(
#                 kernel_size=3,
#                 stride=2,
#                 padding=1,
#             ),

#             # Block 1
#             XceptionBlock(
#                 32,
#                 64,
#                 stride=1,
#                 reduction=reduction,
#             ),

#             # Block 2
#             XceptionBlock(
#                 64,
#                 128,
#                 stride=2,
#                 reduction=reduction,
#             ),

#             # Block 3
#             XceptionBlock(
#                 128,
#                 256,
#                 stride=2,
#                 reduction=reduction,
#             ),

#             # Block 4
#             XceptionBlock(
#                 256,
#                 512,
#                 stride=2,
#                 reduction=reduction,
#             ),
#         )

#         self.pool = nn.AdaptiveAvgPool2d(1)

#         self.classifier = nn.Sequential(
#             nn.Dropout(dropout),
#             nn.Linear(512, num_classes),
#         )

#     def forward(self, x, return_features=False):

#         x = self.features(x)

#         x = self.pool(x)

#         feat = torch.flatten(x, 1)

#         logits = self.classifier(feat)

#         if return_features:
#             return logits, feat

#         return logits
