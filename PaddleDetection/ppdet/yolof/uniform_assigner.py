import paddle
from ppdet.modeling.bbox_utils import batch_bbox_overlaps
from ppdet.modeling.transformers import bbox_xyxy_to_cxcywh
from ppdet.core.workspace import register
from .utils import batch_p_dist



@register
class UniformAssigner(object):
    def __init__(self,
                 pos_ignore_thr,
                 neg_ignore_thr,
                 match_times=4):
        self.pos_ignore_thr = pos_ignore_thr
        self.neg_ignore_thr = neg_ignore_thr
        self.match_times = match_times

    # return match_labels: -2(ignore), -1(neg) or >=0(pos_inds)
    def assign(self,
               bbox_pred,  # [m, 4]
               anchor,     # [m, 4]
               gt_bboxes,  # [n, 4]
               gt_labels=None):
        num_bboxes = bbox_pred.shape[0]
        num_gts    = gt_bboxes.shape[0]
        # first set everything to negative
        match_labels = paddle.full([num_bboxes], -1, dtype=paddle.int32)
        
        pred_ious   = batch_bbox_overlaps(bbox_pred, gt_bboxes)
        pred_max_iou = pred_ious.max(axis=1)
        neg_ignore = pred_max_iou > self.neg_ignore_thr
        # exclude potential ignored neg samples first, deal with pos samples later
        match_labels = paddle.where(neg_ignore, paddle.full_like(match_labels, -2), match_labels)

        bbox_pred_c = bbox_xyxy_to_cxcywh(bbox_pred) # [m, 4]
        anchor_c = bbox_xyxy_to_cxcywh(anchor) # [m, 4]
        gt_bboxes_c = bbox_xyxy_to_cxcywh(gt_bboxes) # [n, 4]
        bbox_pred_dist = batch_p_dist(bbox_pred_c, gt_bboxes_c, p=1) # [m, n]
        anchor_dist = batch_p_dist(anchor_c, gt_bboxes_c, p=1) # [m, n]

        # TODO: torch seems to have problem with topk on different device, do we care on paddle?
        top_pred = bbox_pred_dist.topk(k=self.match_times, axis=0, largest=False)[1] # [4, n], 4 is match_times
        top_anchor = anchor_dist.topk(k=self.match_times, axis=0, largest=False)[1] # [4, n]

        tar_pred = paddle.arange(num_gts).expand([self.match_times, num_gts]) # [4, n]
        tar_anchor = paddle.arange(num_gts).expand([self.match_times, num_gts]) # [4, n]
        pos_places = paddle.concat([top_pred, top_anchor]).reshape([-1])
        pos_inds = paddle.concat([tar_pred, tar_anchor]).reshape([-1])

        pos_anchor = anchor[pos_places]
        pos_tar_bbox = gt_bboxes[pos_inds]
        pos_ious = batch_bbox_overlaps(pos_anchor, pos_tar_bbox, is_aligned=True)
        pos_ignore = pos_ious < self.pos_ignore_thr
        pos_inds = paddle.where(pos_ignore, paddle.full_like(pos_inds, -2), pos_inds)
        match_labels[pos_places] = pos_inds
        match_labels.stop_gradient = True
        pos_keep = ~pos_ignore
        if pos_keep.sum() > 0:
            pos_places_keep = pos_places[pos_keep]
            pos_bbox_pred = bbox_pred[pos_places_keep].reshape([-1, 4])
            pos_bbox_tar  = pos_tar_bbox[pos_keep].reshape([-1, 4]).detach()
        else:
            pos_bbox_pred = None
            pos_bbox_tar  = None
        return match_labels, pos_bbox_pred, pos_bbox_tar
        
        
        
