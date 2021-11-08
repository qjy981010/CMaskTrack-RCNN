import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mmdet.core import (delta2bbox, multiclass_nms, bbox_target,
                        weighted_cross_entropy, weighted_smoothl1, accuracy)
from ..registry import HEADS
from mmdet.ops import DeformConv
from ..utils import ConvModule
from mmcv.cnn import normal_init
from mmdet.ops import spatial_correlation_sample,SpatialCorrelationSampler


@HEADS.register_module
class TrackHead(nn.Module):
    """Tracking head, predict tracking features and match with reference objects
    """

    def __init__(self,
                 with_avg_pool=False,
                 num_fcs = 2,
                 in_channels=256,
                 roi_feat_size=7,
                 fc_out_channels=1024,
                 match_coeff=None,
                 bbox_dummy_iou=0,
                 stack_convs = 3,
                 ):
        super(TrackHead, self).__init__()
        self.stack_covs = stack_convs,
        self.in_channels = in_channels
        self.with_avg_pool = with_avg_pool
        self.roi_feat_size = roi_feat_size
        self.match_coeff = match_coeff
        self.bbox_dummy_iou = bbox_dummy_iou
        self.num_fcs = num_fcs
        if self.with_avg_pool:
            self.avg_pool = nn.AvgPool2d(roi_feat_size)
        else:
            in_channels *= (self.roi_feat_size * self.roi_feat_size) 
        self.fcs = nn.ModuleList()
        for i in range(num_fcs):

            in_channels = (in_channels
                          if i == 0 else fc_out_channels)
            fc = nn.Linear(in_channels, fc_out_channels)
            self.fcs.append(fc)
        self.relu = nn.ReLU(inplace=True)
        self.debug_imgs = None

        self.feat_channels = 256
        self.max_displacement = 16
        self.cor_channels =  self.max_displacement*self.max_displacement
        

        self.stacked_convs = 3

        self.offset_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            if(i==0):
                input_channels = self.cor_channels
            else:
                input_channels = self.feat_channels
            self.offset_convs.append(
                ConvModule(
                    input_channels,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    normalize=None,
                    )
                )
        self.conv_offset = nn.Conv2d(
            self.feat_channels,
            18,
            3,
            padding=1)
        self.dconv_featureflow = DeformConv(
            self.feat_channels,
            self.feat_channels,
            3,
            padding=1
        )

        self.correlation_sampler = SpatialCorrelationSampler(kernel_size=1,patch_size=self.max_displacement,stride=1,padding=0,dilation=1,dilation_patch=2)

    def correlate(self,input1, input2):
        out_corr = self.correlation_sampler(input1,input2)
        # collate dimensions 1 and 2 in order to be treated as a
        # regular 4D tensor
        b, ph, pw, h, w = out_corr.size()
        out_corr = out_corr.view(b, ph * pw, h, w)/input1.size(1)
        return F.leaky_relu_(out_corr, 0.1)

    def feature_mixup(self,cur_x_list,ref_x_list):
        fixed_features = []

        for cur_x, ref_x in zip(cur_x_list,ref_x_list):
            offset_x = self.correlate(cur_x, ref_x)
            for offset_conv in self.offset_convs:
                offset_x = offset_conv(offset_x)
            offsets = self.conv_offset(offset_x)   

            flowed_feature = self.dconv_featureflow(ref_x,offsets)
            fixed_feature = flowed_feature + cur_x
            fixed_features.append(fixed_feature)

        return fixed_features

    def init_weights(self):
        for fc in self.fcs:
            nn.init.normal_(fc.weight, 0, 0.01)
            nn.init.constant_(fc.bias, 0)

    def compute_comp_scores(self, match_ll, bbox_scores, bbox_ious, label_delta, add_bbox_dummy=False):
        # compute comprehensive matching score based on matchig likelihood,
        # bbox confidence, and ious
        if add_bbox_dummy:
            bbox_iou_dummy =  torch.ones(bbox_ious.size(0), 1, 
                device=torch.cuda.current_device()) * self.bbox_dummy_iou
            bbox_ious = torch.cat((bbox_iou_dummy, bbox_ious), dim=1)
            label_dummy = torch.ones(bbox_ious.size(0), 1,
                device=torch.cuda.current_device())
            label_delta = torch.cat((label_dummy, label_delta),dim=1)
        if self.match_coeff is None:
            return match_ll
        else:
            # match coeff needs to be length of 3
            assert(len(self.match_coeff) == 3)
            return match_ll + self.match_coeff[0] * \
                torch.log(bbox_scores) + self.match_coeff[1] * bbox_ious \
                + self.match_coeff[2] * label_delta
    
    def forward(self, x, ref_x, x_n, ref_x_n):
        # x and ref_x are the grouped bbox features of current and reference frame
        # x_n are the numbers of proposals in the current images in the mini-batch, 
        # ref_x_n are the numbers of ground truth bboxes in the reference images.
        # here we compute a correlation matrix of x and ref_x
        # we also add a all 0 column denote no matching
        assert len(x_n) == len(ref_x_n)
        if self.with_avg_pool:
            x = self.avg_pool(x)
            ref_x = self.avg_pool(ref_x)
        x = x.view(x.size(0), -1)
        ref_x = ref_x.view(ref_x.size(0), -1)
        for idx, fc in enumerate(self.fcs):
            x = fc(x)
            ref_x = fc(ref_x)
            if idx < len(self.fcs) - 1:
                x = self.relu(x)
                ref_x = self.relu(ref_x)
        n = len(x_n)
        x_split = torch.split(x, x_n, dim=0)
        ref_x_split = torch.split(ref_x, ref_x_n, dim=0)
        prods = []
        for i in range(n):
          
            prod = torch.mm(x_split[i], torch.transpose(ref_x_split[i], 0, 1))
            prods.append(prod)
        match_score = []
        for prod in prods:
            m = prod.size(0)
            dummy = torch.zeros( m, 1, device=torch.cuda.current_device())
            
            prod_ext = torch.cat([dummy, prod], dim=1)
            match_score.append(prod_ext)
        return match_score

    def loss(self,
             match_score,
             ids,
             id_weights,
             reduce=True):
        losses = dict()
        n = len(match_score)
        x_n = [s.size(0) for s in match_score]
        ids = torch.split(ids, x_n, dim=0)
        loss_match = 0.
        match_acc = 0.
        n_total = 0
        batch_size = len(ids)
        for score, cur_ids, cur_weights in zip(match_score, ids, id_weights):
            valid_idx = torch.nonzero(cur_weights).squeeze()
            if len(valid_idx.size()) == 0: continue
            n_valid = valid_idx.size(0)
            n_total += n_valid
            loss_match += weighted_cross_entropy(
                score, cur_ids, cur_weights, reduce=reduce)
            match_acc += accuracy(torch.index_select(score, 0, valid_idx), 
                                    torch.index_select(cur_ids,0, valid_idx)) * n_valid
        losses['loss_match'] = loss_match / n
        if n_total > 0:
            losses['match_acc'] = match_acc / n_total
        return losses

