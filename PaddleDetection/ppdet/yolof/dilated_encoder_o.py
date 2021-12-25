import paddle
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle.nn.initializer import KaimingUniform, Constant, Normal
from paddle.regularizer import L2Decay
from ppdet.core.workspace import register
from .kaiming_uniform import KaimingUniformImpl
from .utils import bn_attr


'''
This implementation is based on 
https://github.com/chensnathan/YOLOF/blob/master/yolof/modeling/encoder.py
which is the official detectron2 version of YOLOF.
'''

class Bottleneck(paddle.nn.Layer):

    def __init__(self,
                 in_channels,
                 mid_channels,
                 dilation):
        super(Bottleneck, self).__init__()
        self.conv1 = paddle.nn.Sequential(
            paddle.nn.Conv2D(
                in_channels, mid_channels, 1, padding=0,
                weight_attr=self._conv_weight_attr(),
                bias_attr=self._conv_bias_attr()),
            # bn
            paddle.nn.BatchNorm2D(
                mid_channels,
                weight_attr=bn_attr(init=1.0, decay=0.0),
                bias_attr=bn_attr(init=0.0, decay=0.0)),
            paddle.nn.ReLU())
        self.conv2 = paddle.nn.Sequential(
            paddle.nn.Conv2D(
                mid_channels, mid_channels, 3, padding=dilation, dilation=dilation,
                weight_attr=self._conv_weight_attr(),
                bias_attr=self._conv_bias_attr()),
            # bn
            paddle.nn.BatchNorm2D(
                mid_channels,
                weight_attr=bn_attr(init=1.0, decay=0.0),
                bias_attr=bn_attr(init=0.0, decay=0.0)),
            paddle.nn.ReLU())
        self.conv3 = paddle.nn.Sequential(
            paddle.nn.Conv2D(
                mid_channels, in_channels, 1, padding=0,
                weight_attr=self._conv_weight_attr(),
                bias_attr=self._conv_bias_attr()),
            paddle.nn.BatchNorm2D(
                in_channels,
                weight_attr=bn_attr(init=1.0, decay=0.0),
                bias_attr=bn_attr(init=0.0, decay=0.0)),
            paddle.nn.ReLU())

    def _conv_weight_attr(self):
        return ParamAttr(initializer=Normal(mean=0, std=0.01))

    def _conv_bias_attr(self):
        return ParamAttr(initializer=Constant(value=0.0))

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out + identity
        return out


@register
class DilatedEncoder(paddle.nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 block_mid_channels,
                 num_residual_blocks):
        super(DilatedEncoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block_mid_channels = block_mid_channels
        self.num_residual_blocks = num_residual_blocks
        self.block_dilations = [2, 4, 6, 8]

        self.init_layers()

    def init_layers(self):
        self.lateral_conv = paddle.nn.Conv2D(
            self.in_channels, self.out_channels, 1,
            weight_attr=ParamAttr(
                initializer=KaimingUniformImpl(a=1, mode='fan_in', nonlinearity='leaky_relu')),
            bias_attr=ParamAttr(initializer=Constant(value=0.0)))
        # bn
        self.lateral_norm = paddle.nn.BatchNorm2D(
            self.out_channels,
            weight_attr=bn_attr(init=1.0, decay=0.0),
            bias_attr=bn_attr(init=0.0, decay=0.0))
        self.fpn_conv = paddle.nn.Conv2D(
            self.out_channels, self.out_channels, 3, padding=1,
            weight_attr=ParamAttr(
                initializer=KaimingUniformImpl(a=1, mode='fan_in', nonlinearity='leaky_relu')))
        # bn
        self.fpn_norm = paddle.nn.BatchNorm2D(
            self.out_channels,
            weight_attr=bn_attr(init=1.0, decay=0.0),
            bias_attr=bn_attr(init=0.0, decay=0.0))
        # encoder_blocks
        encoder_blocks = []
        for i in range(self.num_residual_blocks):
            dilation = self.block_dilations[i]
            encoder_blocks.append(
                Bottleneck(self.out_channels, self.block_mid_channels, dilation=dilation))
        self.dilated_encoder_blocks = paddle.nn.Sequential(*encoder_blocks)

    def forward(self, feature):
        out = self.lateral_conv(feature[-1])
        out = self.lateral_norm(out)
        out = self.fpn_conv(out)
        out = self.fpn_norm(out)
        return (self.dilated_encoder_blocks(out), )
