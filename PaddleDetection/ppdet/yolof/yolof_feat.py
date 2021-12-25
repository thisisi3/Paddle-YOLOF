import paddle
from paddle.nn.initializer import Normal, Constant
from paddle.regularizer import L2Decay
from ppdet.core.workspace import register
from paddle import ParamAttr

from .utils import bn_attr


@register
class YOLOFFeat(paddle.nn.Layer):
    def __init__(self,
                 feat_in=256,
                 feat_out=256,
                 num_cls_convs=2,
                 num_reg_convs=4,
                 norm_type='bn'):
        super(YOLOFFeat, self).__init__()
        assert norm_type == 'bn', 'only support BN for now'
        self.feat_in = feat_in
        self.feat_out = feat_out
        self.num_cls_convs = num_cls_convs
        self.num_reg_convs = num_reg_convs
        self.norm_type = norm_type

        self.init_layers()

    def init_layers(self):
        cls_subnet = []
        reg_subnet = []
        for i in range(self.num_cls_convs):
            feat_in = self.feat_in if i == 0 else self.feat_out
            cls_subnet.append(paddle.nn.Conv2D(
                feat_in, self.feat_out, 3, stride=1, padding=1,
                weight_attr=ParamAttr(initializer=Normal(mean=0.0, std=0.01)),
                bias_attr=ParamAttr(initializer=Constant(value=0.0))
            ))
            # bn
            cls_subnet.append(paddle.nn.BatchNorm2D(
                self.feat_out,
                weight_attr=bn_attr(init=1.0, decay=0.0),
                bias_attr=bn_attr(init=0.0, decay=0.0)
            ))
            cls_subnet.append(paddle.nn.ReLU())
        for i in range(self.num_reg_convs):
            feat_in = self.feat_in if i == 0 else self.feat_out
            reg_subnet.append(paddle.nn.Conv2D(
                feat_in, self.feat_out, 3, stride=1, padding=1,
                weight_attr=ParamAttr(initializer=Normal(mean=0.0, std=0.01)),
                bias_attr=ParamAttr(initializer=Constant(value=0.0))
            ))
            # bn
            reg_subnet.append(paddle.nn.BatchNorm2D(
                self.feat_out,
                weight_attr=bn_attr(init=1.0, decay=0.0),
                bias_attr=bn_attr(init=0.0, decay=0.0)
            ))
            reg_subnet.append(paddle.nn.ReLU())
        self.cls_subnet = paddle.nn.Sequential(*cls_subnet)
        self.reg_subnet = paddle.nn.Sequential(*reg_subnet)

    def forward(self, feature):
        cls_feat = self.cls_subnet(feature)
        reg_feat = self.reg_subnet(feature)
        return cls_feat, reg_feat
        
        
