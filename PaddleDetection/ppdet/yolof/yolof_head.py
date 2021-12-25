from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle.nn.initializer import Normal, Constant

from ppdet.core.workspace import register
from ppdet.modeling.layers import ConvNormLayer
from .utils import batch_transpose, batch_reshape, find_inside_anchor

def reduce_mean(tensor):
    world_size = paddle.distributed.get_world_size()
    if world_size == 1:
        return tensor
    paddle.distributed.all_reduce(tensor)
    return tensor/world_size


INF = 1e8

@register
class YOLOFHead(paddle.nn.Layer):
    __inject__ = ['conv_feat', 'anchor_generator', 'loss_class', 'loss_bbox',
                  'bbox_assigner', 'bbox_coder', 'nms']

    def __init__(self,
                 num_classes=80,
                 prior_prob=0.01,
                 nms_pre=1000,
                 use_inside_anchor=False,
                 conv_feat=None,
                 anchor_generator=None,
                 bbox_assigner=None,
                 bbox_coder=None,
                 loss_class=None,
                 loss_bbox=None,
                 nms=None):
        super(YOLOFHead, self).__init__()
        self.num_classes = num_classes
        self.prior_prob = prior_prob
        self.nms_pre = nms_pre
        self.use_inside_anchor = use_inside_anchor
        self.conv_feat = conv_feat
        self.anchor_generator = anchor_generator
        self.loss_class = loss_class
        self.loss_bbox  = loss_bbox
        self.bbox_assigner = bbox_assigner
        self.bbox_coder = bbox_coder
        self.nms = nms

        assert loss_class.use_sigmoid, 'only support sigmoid'
        self.cls_out_channels = num_classes

        bias_init_value = - math.log((1 - self.prior_prob) / self.prior_prob)
        self.cls_score = self.add_sublayer(
            'cls_score',
            paddle.nn.Conv2D(
                in_channels=conv_feat.feat_out,
                out_channels=self.cls_out_channels * anchor_generator.num_anchors,
                kernel_size=3,
                stride=1,
                padding=1,
                weight_attr=ParamAttr(initializer=Normal(mean=0.0, std=0.01)),
                bias_attr=ParamAttr(initializer=Constant(value=bias_init_value))))

        self.bbox_pred = self.add_sublayer(
            'bbox_pred',
            paddle.nn.Conv2D(
                in_channels=conv_feat.feat_out,
                out_channels=4 * anchor_generator.num_anchors,
                kernel_size=3,
                stride=1,
                padding=1,
                weight_attr=ParamAttr(initializer=Normal(mean=0.0, std=0.01)),
                bias_attr=ParamAttr(initializer=Constant(value=0))))
        
        self.object_pred = self.add_sublayer(
            'object_pred',
            paddle.nn.Conv2D(
                in_channels=conv_feat.feat_out,
                out_channels=anchor_generator.num_anchors,
                kernel_size=3,
                stride=1,
                padding=1,
                weight_attr=ParamAttr(initializer=Normal(mean=0.0, std=0.01)),
                bias_attr=ParamAttr(initializer=Constant(value=0))))

    def forward(self, neck_feats):
        cls_logits_list = []
        bboxes_reg_list = []
        for feat in neck_feats:
            conv_cls_feat, conv_reg_feat = self.conv_feat(feat)
            cls_logits = self.cls_score(conv_cls_feat)
            N, _, H, W = cls_logits.shape
            cls_logits = cls_logits.reshape([N, -1, self.num_classes, H, W])
            objectness = self.object_pred(conv_reg_feat)
            # implicit objectness
            objectness = objectness.reshape([N, -1, 1, H, W])
            normalized_cls_logits = cls_logits + objectness - paddle.log(
                1.0 + paddle.clip(cls_logits.exp(), max=INF) +
                paddle.clip(objectness.exp(), max=INF))
            normalized_cls_logits = normalized_cls_logits.reshape([N, -1, H, W])
            cls_logits_list.append(normalized_cls_logits)
                
            bboxes_reg = self.bbox_pred(conv_reg_feat)
            bboxes_reg_list.append(bboxes_reg)
        return cls_logits_list, bboxes_reg_list


    def get_loss(self, head_outputs, meta):
        cls_logits, bbox_preds = head_outputs
        anchors = self.anchor_generator(cls_logits)
        assert len(anchors) == len(cls_logits) == len(bbox_preds) == 1, \
            'only support one feature level'

        # cls_logit: [2, 400, 25, 38]
        # bbox_pred: [2, 20,  25, 38]
        anchors, cls_logits, bbox_preds = anchors[0], cls_logits[0], bbox_preds[0]
        feat_size  = cls_logits.shape[-2:]
        cls_logits = cls_logits.transpose([0, 2, 3, 1]) # [2, 25, 38, 400]
        cls_logits = cls_logits.reshape([0, -1, self.cls_out_channels])
        bbox_preds = bbox_preds.transpose([0, 2, 3, 1]) # [2, 25, 38, 20]
        bbox_preds = bbox_preds.reshape([0, -1, 4])

        num_pos_list = []
        cls_pred_list, cls_tar_list, reg_pred_list, reg_tar_list = [], [], [], []
        
        # collect targets in each image and combine them later
        for cls_logit, bbox_pred, gt_bbox, gt_class, im_shape in zip(
                cls_logits, bbox_preds, meta['gt_bbox'], meta['gt_class'], meta['im_shape']):
            if self.use_inside_anchor:
                inside_mask = find_inside_anchor(
                    feat_size,
                    self.anchor_generator.strides[0],
                    self.anchor_generator.num_anchors,
                    im_shape.tolist())
                cls_logit = cls_logit[inside_mask]
                bbox_pred = bbox_pred[inside_mask]
                anchor = anchors[inside_mask]
            else:
                anchor = anchors
            bbox_pred = self.bbox_coder.decode(anchor, bbox_pred)
            # -2:ignore, -1:neg, >=0:pos
            match_labels, pos_bbox_pred, pos_bbox_tar = self.bbox_assigner.assign(bbox_pred, anchor, gt_bbox)
            pos_mask = (match_labels >= 0)
            neg_mask = (match_labels == -1)
            chosen_mask = paddle.logical_or(pos_mask, neg_mask)
            gt_class = gt_class.reshape([-1])
            gt_class = paddle.concat([
                gt_class,
                paddle.to_tensor([self.num_classes], dtype=gt_class.dtype, place=gt_class.place)
            ])
            match_labels = paddle.where(
                neg_mask, paddle.full_like(match_labels, gt_class.size-1), match_labels)
            cls_pred = cls_logit[chosen_mask]
            cls_tar  = gt_class[match_labels[chosen_mask]]
            num_pos_list.append(max(1.0, pos_mask.sum().item()))
            reg_pred = pos_bbox_pred
            reg_tar  = pos_bbox_tar
            cls_pred_list.append(cls_pred)
            cls_tar_list.append(cls_tar)
            reg_pred_list.append(reg_pred)
            reg_tar_list.append(reg_tar)
        num_tot_pos = sum(num_pos_list)
        num_tot_pos = paddle.to_tensor(num_tot_pos)
        num_tot_pos = reduce_mean(num_tot_pos).item()
        num_tot_pos = max(1.0, num_tot_pos)

        cls_pred = paddle.concat(cls_pred_list)
        cls_tar  = paddle.concat(cls_tar_list)
        cls_loss = self.loss_class(cls_pred, cls_tar, avg_factor=num_tot_pos)
        reg_pred_list = [_ for _ in reg_pred_list if _ is not None]
        reg_tar_list  = [_ for _ in reg_tar_list  if _ is not None]
        if len(reg_pred_list) == 0:
            reg_loss = bbox_preds[0, 0].sum() * 0.0 # a fake 0 loss
        else:
            reg_pred = paddle.concat(reg_pred_list)
            reg_tar  = paddle.concat(reg_tar_list)
            reg_loss = self.loss_bbox(reg_pred, reg_tar).sum() / num_tot_pos
        return dict(loss_cls=cls_loss, loss_reg=reg_loss)

    def get_bboxes_single(self,
                          anchors,
                          cls_scores,
                          bbox_preds,
                          im_shape,
                          scale_factor,
                          rescale=True):
        assert len(cls_scores) == len(bbox_preds)
        mlvl_bboxes = []
        mlvl_scores = []
        for anchor, cls_score, bbox_pred in zip(anchors, cls_scores, bbox_preds):
            cls_score = cls_score.reshape([-1, self.cls_out_channels])
            bbox_pred = bbox_pred.reshape([-1, 4])
            if self.nms_pre is not None and cls_score.shape[0] > self.nms_pre:
                max_score = cls_score.max(axis=1)
                _, topk_inds = max_score.topk(self.nms_pre)
                bbox_pred = bbox_pred.gather(topk_inds)
                anchor    = anchor.gather(topk_inds)
                cls_score = cls_score.gather(topk_inds)
            bbox_pred = self.bbox_coder.decode(anchor, bbox_pred, max_shape=im_shape)
            bbox_pred = bbox_pred.squeeze()
            mlvl_bboxes.append(bbox_pred)
            mlvl_scores.append(F.sigmoid(cls_score))
        mlvl_bboxes = paddle.concat(mlvl_bboxes)
        mlvl_bboxes = paddle.squeeze(mlvl_bboxes)
        if rescale:
            mlvl_bboxes = mlvl_bboxes / paddle.concat([scale_factor[::-1], scale_factor[::-1]])
        mlvl_scores = paddle.concat(mlvl_scores)
        mlvl_scores = mlvl_scores.transpose([1, 0])
        return mlvl_bboxes, mlvl_scores

    def decode(self, anchors, cls_scores, bbox_preds, im_shape, scale_factor):
        batch_bboxes = []
        batch_scores = []
        for img_id in range(cls_scores[0].shape[0]):
            num_lvls = len(cls_scores)
            cls_score_list = [cls_scores[i][img_id] for i in range(num_lvls)]
            bbox_pred_list = [bbox_preds[i][img_id] for i in range(num_lvls)]
            bboxes, scores = self.get_bboxes_single(
                anchors,
                cls_score_list,
                bbox_pred_list,
                im_shape[img_id],
                scale_factor[img_id])
            batch_bboxes.append(bboxes)
            batch_scores.append(scores)
        batch_bboxes = paddle.stack(batch_bboxes, axis=0)
        batch_scores = paddle.stack(batch_scores, axis=0)
        return batch_bboxes, batch_scores


    def post_process(self, head_outputs, im_shape, scale_factor):
        cls_scores, bbox_preds = head_outputs
        anchors = self.anchor_generator(cls_scores)
        cls_scores = batch_transpose(cls_scores, [0, 2, 3, 1])
        bbox_preds = batch_transpose(bbox_preds, [0, 2, 3, 1])
        bboxes, scores = self.decode(anchors, cls_scores, bbox_preds, im_shape, scale_factor)

        bbox_pred, bbox_num, _ = self.nms(bboxes, scores)
        return bbox_pred, bbox_num
