from ppdet.core.workspace import register
from .utils import expand_onehot, zero_loss
from paddle.nn import functional as F
import paddle


# FocalLoss is based on
# https://github.com/open-mmlab/mmdetection/blob/v2.16.0/mmdet/models/losses/focal_loss.py


@register
class FocalLoss(paddle.nn.Layer):
    def __init__(self,
                 use_sigmoid=True,
                 gamma=2.0,
                 alpha=0.25,
                 loss_weight=1.0):
        super(FocalLoss, self).__init__()
        self.use_sigmoid = use_sigmoid
        assert use_sigmoid
        self.gamma = gamma
        self.alpha = alpha
        self.loss_weight = loss_weight

    def forward(self, pred, target, avg_factor):
        if pred.size == 0:
            return zero_loss(pred)
        target.stop_gradient = True
        target = expand_onehot(target, pred.shape[-1])
        pred_sigm = F.sigmoid(pred)
        pt = (1 - pred_sigm) * target + pred_sigm * (1 - target)
        focal_weight = (self.alpha * target + (1 - self.alpha) * (1 - target)) * pt.pow(self.gamma)
        loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none') * focal_weight
        loss = loss.sum() / avg_factor
        return loss * self.loss_weight


@register
class L1Loss(paddle.nn.Layer):
    def __init__(self,
                 loss_weight=1.0):
        super(L1Loss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, pred, target, avg_factor):
        if pred.size == 0:
            return zero_loss(pred)
        target.stop_gradient = True
        loss = paddle.abs(pred - target)
        loss = loss.sum() / avg_factor
        return loss * self.loss_weight
