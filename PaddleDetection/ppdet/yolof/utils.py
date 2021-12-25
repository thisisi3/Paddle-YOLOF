import paddle
import numpy as np

from paddle.nn.initializer import Constant
from paddle.regularizer import L2Decay
from paddle import ParamAttr

def batch_transpose(tensors, perm):
    trans = []
    for tensor in tensors:
        trans.append(tensor.transpose(perm))
    return trans

def batch_reshape(tensors, shape):
    reshape = []
    for tensor in tensors:
        reshape.append(tensor.reshape(shape))
    return reshape

# calculate pairwise p_dist
# the first index of x and y are batch
# return [x.shape[0], y.shape[0]]
def batch_p_dist(x, y, p=2):
    x = x.unsqueeze(1)
    diff = x - y
    return paddle.norm(diff, p=p, axis=list(range(2, diff.dim())))

def expand_onehot(labels, label_channels):
    # paddle.full does not support place as argument
    expand = paddle.to_tensor(
        paddle.full((labels.shape[0], label_channels+1), 0, dtype=paddle.float32), place=labels.place)
    expand[paddle.arange(labels.shape[0]), labels] = 1
    return expand[:, :-1]

def zero_loss(like):
    return paddle.to_tensor([0], dtype=like.dtype, place=like.place, stop_gradient=False)

def find_inside_anchor(feat_size, stride, num_anchors, im_shape):
    feat_h, feat_w = feat_size[:2]
    im_h, im_w = im_shape[:2]
    inside_h = min(int(np.ceil(im_h / stride)), feat_h)
    inside_w = min(int(np.ceil(im_w / stride)), feat_w)
    inside_mask = paddle.zeros([feat_h, feat_w], dtype=paddle.bool)
    inside_mask[:inside_h, :inside_w] = True
    inside_mask = inside_mask.unsqueeze(-1).expand([feat_h, feat_w, num_anchors])
    return inside_mask.reshape([-1])

def bn_attr(init, decay=None):
    if decay is None:
        return ParamAttr(initializer=Constant(value=init))
    else:
        return ParamAttr(initializer=Constant(value=init), regularizer=L2Decay(decay))
