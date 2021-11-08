import torch
import torch.nn as nn
import numpy as np
from .base import BaseDetector
from .test_mixins import RPNTestMixin, MaskTestMixin
from .. import builder
from ..registry import DETECTORS
from mmdet.core import bbox2roi, bbox2result, build_assigner, build_sampler
from mmdet.core import bbox_overlaps, bbox2result_with_id


@DETECTORS.register_module
class TwoStageDetector(BaseDetector, RPNTestMixin,
                       MaskTestMixin):

    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 track_head=None,
                 mask_roi_extractor=None,
                 mask_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(TwoStageDetector, self).__init__()
        self.backbone = builder.build_backbone(backbone)

        if neck is not None:
            self.neck = builder.build_neck(neck)
        else:
            raise NotImplementedError

        if rpn_head is not None:
            self.rpn_head = builder.build_head(rpn_head)

        if bbox_head is not None:
            self.bbox_roi_extractor = builder.build_roi_extractor(
                bbox_roi_extractor)
            self.bbox_head = builder.build_head(bbox_head)
        if track_head is not None:
            if 'detach_ref' in track_head:
                self.detach_ref = track_head['detach_ref']
                del track_head['detach_ref']
            else:
                self.detach_ref = False
            self.track_head = builder.build_head(track_head)
        if mask_head is not None:
            self.mask_roi_extractor = builder.build_roi_extractor(
                mask_roi_extractor)
            self.mask_head = builder.build_head(mask_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        # memory queue for testing
        self.prev_bboxes = None
        self.prev_roi_feats = None
        self.init_weights(pretrained=pretrained)

    @property
    def with_rpn(self):
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    def init_weights(self, pretrained=None):
        super(TwoStageDetector, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        if self.with_rpn:
            self.rpn_head.init_weights()
        if self.with_bbox:
            self.bbox_roi_extractor.init_weights()
            self.bbox_head.init_weights()
        if self.with_mask:
            self.mask_roi_extractor.init_weights()
            self.mask_head.init_weights()
        if self.with_track:
            self.track_head.init_weights()

    def extract_feat(self, img):
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_train(self,
                      img,
                      img_meta,
                      gt_bboxes,
                      gt_bboxes_ignore,
                      gt_labels,
                      ref_img, # images of reference frame
                      ref_bboxes, # gt bbox of reference frame
                      gt_pids, # gt ids of current frame bbox mapped to reference frame
                      gt_masks=None,
                      proposals=None):
        x = self.extract_feat(img)
        ref_x = self.extract_feat(ref_img)
        mixed_x = self.track_head.feature_mixup(x, [(a.detach() if self.detach_ref else a) for a in ref_x])
        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            rpn_loss_inputs = rpn_outs + (gt_bboxes, img_meta,
                                          self.train_cfg.rpn)
            rpn_losses = self.rpn_head.loss(*rpn_loss_inputs)
            losses.update(rpn_losses)

            proposal_inputs = rpn_outs + (img_meta, self.test_cfg.rpn)
            proposal_list = self.rpn_head.get_bboxes(*proposal_inputs)
        else:
            proposal_list = proposals

        # assign gts and sample proposals
        if self.with_bbox or self.with_mask:
            bbox_assigner = build_assigner(self.train_cfg.rcnn.assigner)
            bbox_sampler = build_sampler(
                self.train_cfg.rcnn.sampler, context=self)
            num_imgs = img.size(0)
            sampling_results = []
            for i in range(num_imgs):
                assign_result = bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i], gt_pids[i])
                sampling_result = bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    gt_pids[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

        # bbox head forward and loss
        if self.with_bbox:
            rois = bbox2roi([res.bboxes for res in sampling_results])
            bbox_img_n = [res.bboxes.size(0) for res in sampling_results]
            ref_rois = bbox2roi(ref_bboxes)
            ref_bbox_img_n = [x.size(0) for x in ref_bboxes]
            # TODO: a more flexible way to decide which feature maps to use
            bbox_feats = self.bbox_roi_extractor(
                x[:self.bbox_roi_extractor.num_inputs], rois)
            mixed_feats = self.bbox_roi_extractor(
                mixed_x[:self.bbox_roi_extractor.num_inputs], rois)


            ref_bbox_feats = self.bbox_roi_extractor(
                ref_x[:self.bbox_roi_extractor.num_inputs], ref_rois)

            cls_score, bbox_pred = self.bbox_head(mixed_feats)
            # fetch bbox and object_id targets
            bbox_targets, (ids, id_weights) = self.bbox_head.get_target(
                sampling_results, gt_bboxes, gt_labels, self.train_cfg.rcnn)
            loss_bbox = self.bbox_head.loss(cls_score, bbox_pred,
                                            *bbox_targets)
            losses.update(loss_bbox)
            match_score = self.track_head(bbox_feats, ref_bbox_feats, 
                                          bbox_img_n, ref_bbox_img_n)
            loss_match = self.track_head.loss(match_score,
                                              ids, id_weights)
            losses.update(loss_match)
        # mask head forward and loss
        if self.with_mask:
            pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
            mask_feats = self.mask_roi_extractor(
                mixed_x[:self.mask_roi_extractor.num_inputs], pos_rois)
            mask_pred = self.mask_head(mask_feats)

            mask_targets = self.mask_head.get_target(
                sampling_results, gt_masks, self.train_cfg.rcnn)
            pos_labels = torch.cat(
                [res.pos_gt_labels for res in sampling_results])
            loss_mask = self.mask_head.loss(mask_pred, mask_targets,
                                            pos_labels)
            losses.update(loss_mask)

        return losses

    def simple_test_bboxes(self,
                           x,
                           mixed_x_list,
                           img_meta,
                           proposals,
                           rcnn_test_cfg,
                           rescale=False):
        """Test only det bboxes without augmentation."""
        rois = bbox2roi(proposals)

        cls_scores = []
        bbox_preds = []
        for mixed_x in mixed_x_list:
            roi_feats = self.bbox_roi_extractor(
                mixed_x[:len(self.bbox_roi_extractor.featmap_strides)], rois)
            cls_score, bbox_pred = self.bbox_head(roi_feats)
            cls_scores.append(cls_score)
            bbox_preds.append(bbox_pred)
        cls_score = torch.stack(cls_scores,dim=0).mean(dim=0)
        bbox_pred = torch.stack(bbox_preds,dim=0).mean(dim=0)

        img_shape = img_meta[0]['img_shape']
        scale_factor = img_meta[0]['scale_factor']
        is_first = img_meta[0]['is_first']
        det_bboxes, det_labels = self.bbox_head.get_det_bboxes(
            rois,
            cls_score,
            bbox_pred,
            img_shape,
            scale_factor,
            rescale=rescale,
            cfg=rcnn_test_cfg)

        if det_bboxes.nelement()==0:
            det_obj_ids=np.array([], dtype=np.int64)
            if is_first:
                self.prev_bboxes =  None
                self.prev_roi_feats = None
                self.prev_det_labels = None
            return det_bboxes, det_labels, det_obj_ids

        res_det_bboxes = det_bboxes.clone()
        if rescale:
            res_det_bboxes[:, :4] *= scale_factor

        det_rois = bbox2roi([res_det_bboxes])
        det_roi_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], det_rois)

        if is_first or (not is_first and self.prev_bboxes is None):
            det_obj_ids = np.arange(det_bboxes.size(0))
            # save bbox and features for later matching
            self.prev_bboxes = det_bboxes
            self.prev_roi_feats = det_roi_feats
            self.prev_det_labels = det_labels
        else:
            assert self.prev_roi_feats is not None
            # only support one image at a time
            bbox_img_n = [det_bboxes.size(0)]
            prev_bbox_img_n = [self.prev_roi_feats.size(0)]

            match_score = self.track_head(det_roi_feats, self.prev_roi_feats,
                                      bbox_img_n, prev_bbox_img_n)[0]
            match_logprob = torch.nn.functional.log_softmax(match_score, dim=1)

            label_delta = (self.prev_det_labels == det_labels.view(-1,1)).float()
            bbox_ious = bbox_overlaps(det_bboxes[:,:4], self.prev_bboxes[:,:4])
            # compute comprehensive score 
            comp_scores = self.track_head.compute_comp_scores(
                match_logprob, 
                det_bboxes[:,4].view(-1, 1),
                bbox_ious,
                label_delta,
                add_bbox_dummy=True)
            match_likelihood, match_ids = torch.max(comp_scores, dim =1)
            # translate match_ids to det_obj_ids, assign new id to new objects
            # update tracking features/bboxes of exisiting object, 
            # add tracking features/bboxes of new object
            match_ids = match_ids.cpu().numpy().astype(np.int32)
            det_obj_ids = np.ones((match_ids.shape[0]), dtype=np.int32) * (-1)
            best_match_scores = np.ones((self.prev_bboxes.size(0))) * (-100)
            for idx, match_id in enumerate(match_ids):
                if match_id == 0:
                    # add new object
                    det_obj_ids[idx] = self.prev_roi_feats.size(0)
                    self.prev_roi_feats = torch.cat((self.prev_roi_feats, det_roi_feats[idx][None]), dim=0)
                    self.prev_bboxes = torch.cat((self.prev_bboxes, det_bboxes[idx][None]), dim=0)
                    self.prev_det_labels = torch.cat((self.prev_det_labels, det_labels[idx][None]), dim=0)
                else:
                    # multiple candidate might match with previous object, here we choose the one with
                    # largest comprehensive score 
                    obj_id = match_id - 1
                    match_score = comp_scores[idx, match_id]
                    if match_score > best_match_scores[obj_id]:
                        det_obj_ids[idx] = obj_id
                        best_match_scores[obj_id] = match_score
                        # udpate feature
                        self.prev_roi_feats[obj_id] = det_roi_feats[idx]
                        self.prev_bboxes[obj_id] = det_bboxes[idx]

        return det_bboxes, det_labels, det_obj_ids

    def simple_test(self, img, img_meta,ref_imgs,ref_img_metas, proposals=None, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, "Bbox head must be implemented."
        assert self.with_track, "Track head must be implemented"
        is_first = img_meta[0]['is_first']
        if is_first:
            self.cur_idx = 0
            self.len_ref = len(ref_imgs[0])
            self.extracted_feats = [self.extract_feat(img)] + [self.extract_feat(ref_img) for ref_img in ref_imgs[0]]
        else:
            len_ref = len(ref_imgs[0])
            if len_ref > self.len_ref:
                self.cur_idx += 1
                self.extracted_feats.append(self.extract_feat(ref_imgs[0][-1]))
            elif len_ref == self.len_ref:
                self.extracted_feats.append(self.extract_feat(ref_imgs[0][-1]))
                self.extracted_feats.pop(0)
            else:
                self.extracted_feats.pop(0)
            self.len_ref = len_ref

        x = self.extracted_feats[self.cur_idx]
        mixed_feat_list = []
        for i in range(len(self.extracted_feats)):
            if i == self.cur_idx:
                continue
            ref_feat = self.extracted_feats[i]
            mixed_feat = self.track_head.feature_mixup(x,ref_feat)
            mixed_feat_list.append(mixed_feat)

        proposal_list = self.simple_test_rpn(
            x, img_meta, self.test_cfg.rpn) if proposals is None else proposals

        det_bboxes, det_labels, det_obj_ids = self.simple_test_bboxes(
            x,mixed_feat_list, img_meta,proposal_list, self.test_cfg.rcnn, rescale=rescale)
        bbox_results = bbox2result_with_id(det_bboxes, det_labels, det_obj_ids,
                                   self.bbox_head.num_classes)

        if not self.with_mask:
            return bbox_results
        else:
            segm_results = self.simple_test_mask(
                x,mixed_feat_list, img_meta, det_bboxes, det_labels, 
                rescale=rescale, det_obj_ids=det_obj_ids)

            return bbox_results, segm_results

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        # recompute feats to save memory
        proposal_list = self.aug_test_rpn(
            self.extract_feats(imgs), img_metas, self.test_cfg.rpn)
        det_bboxes, det_labels = self.aug_test_bboxes(
            self.extract_feats(imgs), img_metas, proposal_list,
            self.test_cfg.rcnn)

        if rescale:
            _det_bboxes = det_bboxes
        else:
            _det_bboxes = det_bboxes.clone()
            _det_bboxes[:, :4] *= img_metas[0][0]['scale_factor']
        bbox_results = bbox2result(_det_bboxes, det_labels,
                                   self.bbox_head.num_classes)

        # det_bboxes always keep the original scale
        if self.with_mask:
            segm_results = self.aug_test_mask(
                self.extract_feats(imgs), img_metas, det_bboxes, det_labels)
            return bbox_results, segm_results
        else:
            return bbox_results
