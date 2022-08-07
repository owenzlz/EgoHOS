# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.core import add_prefix
from mmseg.ops import resize
from .. import builder
from ..builder import SEGMENTORS
from .base import BaseSegmentor

import os
import pdb
import random
from PIL import Image, ImageOps
import numpy as np 
from torchvision.transforms.functional import to_tensor
from torchvision.utils import save_image

additional_channel = 'twohands_cb'
use_ccda = True

@SEGMENTORS.register_module()
class EncoderDecoder(BaseSegmentor):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """

    def __init__(self,
                 backbone,
                 decode_head,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(EncoderDecoder, self).__init__(init_cfg)
        if pretrained is not None:
            assert backbone.get('pretrained') is None, \
                'both backbone and segmentor set pretrained weight'
            backbone.pretrained = pretrained
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)
        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # pdb.set_trace()

        assert self.with_decode_head

    def _init_decode_head(self, decode_head):
        """Initialize ``decode_head``"""
        self.decode_head = builder.build_head(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes

    def _init_auxiliary_head(self, auxiliary_head):
        """Initialize ``auxiliary_head``"""
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head.append(builder.build_head(head_cfg))
            else:
                self.auxiliary_head = builder.build_head(auxiliary_head)

    def extract_feat(self, img):
        """Extract features from images."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def encode_decode(self, img, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
                
        if img_metas[0]['additional_channel'] == 'twohands':
            
            img_h, img_w = img.shape[2], img.shape[3]; target_aspect_ratio = img_h / img_w
            aux_list = torch.zeros((img.shape[0], 1, img.shape[2], img.shape[3])).to(img.device)
            for i in range(img.shape[0]):
                img_file = img_metas[i]['filename']
                # path = img_metas[i]['twohands_dir']
                path = os.path.join(os.path.dirname(os.path.dirname(img_file)), 'pred_twohands')
                fname = os.path.basename(img_file).split('.')[0] + '.png'
                aux_file = os.path.join(path, fname)

                aux = Image.open(aux_file); aux_w, aux_h = aux.size[0], aux.size[1]
                if aux_h / aux_w < target_aspect_ratio:
                    new_aux_h = int(target_aspect_ratio * aux_w)
                    aux = ImageOps.pad(aux, (aux_w, new_aux_h), centering=(0,0))
                else:
                    new_aux_w = int(aux_h / target_aspect_ratio)
                    aux = ImageOps.pad(aux, (new_aux_w, aux_h), centering=(0,0))

                aux = torch.from_numpy(np.array(aux.resize((img_w, img_h)))).unsqueeze(0).to(img[i].device).float()
                aux_list[i] = aux

            cat_input = torch.cat([img, aux_list], dim = 1)
            x = self.extract_feat(cat_input)

        elif img_metas[0]['additional_channel'] == 'twohands_cb':
            
            img_h, img_w = img.shape[2], img.shape[3]; target_aspect_ratio = img_h / img_w
            aux_list = torch.zeros((img.shape[0], 1, img.shape[2], img.shape[3])).to(img.device)
            cb_list = torch.zeros((img.shape[0], 1, img.shape[2], img.shape[3])).to(img.device)
            for i in range(img.shape[0]):
                img_file = img_metas[i]['filename']
                # aux_path = img_metas[i]['twohands_dir']
                aux_path = os.path.join(os.path.dirname(os.path.dirname(img_file)), 'pred_twohands')
                # cb_path = img_metas[i]['cb_dir']
                cb_path = os.path.join(os.path.dirname(os.path.dirname(img_file)), 'pred_cb')
                fname = os.path.basename(img_file).split('.')[0] + '.png'
                aux_file = os.path.join(aux_path, fname)
                cb_file = os.path.join(cb_path, fname)

                aux = Image.open(aux_file); aux_w, aux_h = aux.size[0], aux.size[1]
                cb = Image.open(cb_file); cb_w, cb_h = cb.size[0], cb.size[1]
                if aux_h / aux_w < target_aspect_ratio:
                    new_aux_h = int(target_aspect_ratio * aux_w)
                    aux = ImageOps.pad(aux, (aux_w, new_aux_h), centering=(0,0))
                    cb = ImageOps.pad(cb, (aux_w, new_aux_h), centering=(0,0))
                else:
                    new_aux_w = int(aux_h / target_aspect_ratio)
                    aux = ImageOps.pad(aux, (new_aux_w, aux_h), centering=(0,0))
                    cb = ImageOps.pad(cb, (new_aux_w, aux_h), centering=(0,0))

                aux = torch.from_numpy(np.array(aux.resize((img_w, img_h)))).unsqueeze(0).to(img[i].device).float()
                aux_list[i] = aux
                cb = torch.from_numpy(np.array(cb.resize((img_w, img_h)))).unsqueeze(0).to(img[i].device).float()
                cb_list[i] = cb
            
            # print('pdb @ encoder_decoder.py 144'); import pdb; pdb.set_trace()
            
            cat_input = torch.cat([img, aux_list, cb_list], dim = 1)
            x = self.extract_feat(cat_input)

        else:

            x = self.extract_feat(img)



        # if additional_channel == 'twohands':
                    
        #     img_h, img_w = img.shape[2], img.shape[3]; target_aspect_ratio = img_h / img_w
        #     aux_list = torch.zeros((img.shape[0], 1, img.shape[2], img.shape[3])).to(img.device)
        #     for i in range(img.shape[0]):
        #         img_file = img_metas[i]['filename']
        #         if use_ccda:
        #             path = os.path.join(os.path.dirname(os.path.dirname(img_file)), 'pred_twohands_ccda')
        #         else:
        #             path = os.path.join(os.path.dirname(os.path.dirname(img_file)), 'pred_twohands')
        #         fname = os.path.basename(img_file).split('.')[0] + '.png'
        #         aux_file = os.path.join(path, fname)

        #         aux = Image.open(aux_file); aux_w, aux_h = aux.size[0], aux.size[1]
        #         if aux_h / aux_w < target_aspect_ratio:
        #             new_aux_h = int(target_aspect_ratio * aux_w)
        #             aux = ImageOps.pad(aux, (aux_w, new_aux_h), centering=(0,0))
        #         else:
        #             new_aux_w = int(aux_h / target_aspect_ratio)
        #             aux = ImageOps.pad(aux, (new_aux_w, aux_h), centering=(0,0))

        #         aux = torch.from_numpy(np.array(aux.resize((img_w, img_h)))).unsqueeze(0).to(img[i].device).float()
        #         aux_list[i] = aux

        #     cat_input = torch.cat([img, aux_list], dim = 1)
        #     x = self.extract_feat(cat_input)

        # elif additional_channel == 'twohands_cb':
            
        #     img_h, img_w = img.shape[2], img.shape[3]; target_aspect_ratio = img_h / img_w
        #     aux_list = torch.zeros((img.shape[0], 1, img.shape[2], img.shape[3])).to(img.device)
        #     cb_list = torch.zeros((img.shape[0], 1, img.shape[2], img.shape[3])).to(img.device)
        #     for i in range(img.shape[0]):
        #         img_file = img_metas[i]['filename']
        #         if use_ccda:
        #             aux_path = os.path.join(os.path.dirname(os.path.dirname(img_file)), 'pred_twohands_ccda')
        #             cb_path = os.path.join(os.path.dirname(os.path.dirname(img_file)), 'pred_cb_ccda')
        #         else:
        #             aux_path = os.path.join(os.path.dirname(os.path.dirname(img_file)), 'pred_twohands')
        #             cb_path = os.path.join(os.path.dirname(os.path.dirname(img_file)), 'pred_cb')
        #         fname = os.path.basename(img_file).split('.')[0] + '.png'
        #         aux_file = os.path.join(aux_path, fname)
        #         cb_file = os.path.join(cb_path, fname)

        #         aux = Image.open(aux_file); aux_w, aux_h = aux.size[0], aux.size[1]
        #         cb = Image.open(cb_file); cb_w, cb_h = cb.size[0], cb.size[1]
        #         if aux_h / aux_w < target_aspect_ratio:
        #             new_aux_h = int(target_aspect_ratio * aux_w)
        #             aux = ImageOps.pad(aux, (aux_w, new_aux_h), centering=(0,0))
        #             cb = ImageOps.pad(cb, (aux_w, new_aux_h), centering=(0,0))
        #         else:
        #             new_aux_w = int(aux_h / target_aspect_ratio)
        #             aux = ImageOps.pad(aux, (new_aux_w, aux_h), centering=(0,0))
        #             cb = ImageOps.pad(cb, (new_aux_w, aux_h), centering=(0,0))

        #         aux = torch.from_numpy(np.array(aux.resize((img_w, img_h)))).unsqueeze(0).to(img[i].device).float()
        #         aux_list[i] = aux
        #         cb = torch.from_numpy(np.array(cb.resize((img_w, img_h)))).unsqueeze(0).to(img[i].device).float()
        #         cb_list[i] = cb

        #     cat_input = torch.cat([img, aux_list, cb_list], dim = 1)
        #     x = self.extract_feat(cat_input)

        # else:

        #     x = self.extract_feat(img)



        out = self._decode_head_forward_test(x, img_metas)
        out = resize(
            input=out,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        return out

    def _decode_head_forward_train(self, x, img_metas, gt_semantic_seg):
        """Run forward function and calculate loss for decode head in
        training."""

        losses = dict()
        loss_decode = self.decode_head.forward_train(x, img_metas,
                                                     gt_semantic_seg,
                                                     self.train_cfg)

        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

    def _decode_head_forward_test(self, x, img_metas):
        """Run forward function and calculate loss for decode head in
        inference."""
        seg_logits = self.decode_head.forward_test(x, img_metas, self.test_cfg)
        return seg_logits

    def _auxiliary_head_forward_train(self, x, img_metas, gt_semantic_seg):
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                loss_aux = aux_head.forward_train(x, img_metas,
                                                  gt_semantic_seg,
                                                  self.train_cfg)
                losses.update(add_prefix(loss_aux, f'aux_{idx}'))
        else:
            loss_aux = self.auxiliary_head.forward_train(
                x, img_metas, gt_semantic_seg, self.train_cfg)
            losses.update(add_prefix(loss_aux, 'aux'))

        return losses

    def forward_dummy(self, img):
        """Dummy forward function."""
        seg_logit = self.encode_decode(img, None)

        return seg_logit

    def forward_train(self, img, img_metas, gt_semantic_seg):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """


        if additional_channel == 'twohands':
                    
            img_h, img_w = img.shape[2], img.shape[3]; target_aspect_ratio = img_h / img_w
            aux_list = torch.zeros((img.shape[0], 1, img.shape[2], img.shape[3])).to(img.device)
            for i in range(img.shape[0]):
                img_file = img_metas[i]['filename']
                if use_ccda:
                    path = os.path.join(os.path.dirname(os.path.dirname(img_file)), 'pred_twohands_ccda')
                else:
                    path = os.path.join(os.path.dirname(os.path.dirname(img_file)), 'pred_twohands')
                fname = os.path.basename(img_file).split('.')[0] + '.png'
                aux_file = os.path.join(path, fname)

                aux = Image.open(aux_file); aux_w, aux_h = aux.size[0], aux.size[1]
                if aux_h / aux_w < target_aspect_ratio:
                    new_aux_h = int(target_aspect_ratio * aux_w)
                    aux = ImageOps.pad(aux, (aux_w, new_aux_h), centering=(0,0))
                else:
                    new_aux_w = int(aux_h / target_aspect_ratio)
                    aux = ImageOps.pad(aux, (new_aux_w, aux_h), centering=(0,0))

                aux = torch.from_numpy(np.array(aux.resize((img_w, img_h)))).unsqueeze(0).to(img[i].device).float()
                aux_list[i] = aux

            cat_input = torch.cat([img, aux_list], dim = 1)
            x = self.extract_feat(cat_input)

        elif additional_channel == 'twohands_cb':
            
            img_h, img_w = img.shape[2], img.shape[3]; target_aspect_ratio = img_h / img_w
            aux_list = torch.zeros((img.shape[0], 1, img.shape[2], img.shape[3])).to(img.device)
            cb_list = torch.zeros((img.shape[0], 1, img.shape[2], img.shape[3])).to(img.device)
            for i in range(img.shape[0]):
                img_file = img_metas[i]['filename']
                if use_ccda:
                    aux_path = os.path.join(os.path.dirname(os.path.dirname(img_file)), 'pred_twohands_ccda')
                    cb_path = os.path.join(os.path.dirname(os.path.dirname(img_file)), 'pred_cb_ccda')
                else:
                    aux_path = os.path.join(os.path.dirname(os.path.dirname(img_file)), 'pred_twohands')
                    cb_path = os.path.join(os.path.dirname(os.path.dirname(img_file)), 'pred_cb')
                fname = os.path.basename(img_file).split('.')[0] + '.png'
                aux_file = os.path.join(aux_path, fname)
                cb_file = os.path.join(cb_path, fname)

                aux = Image.open(aux_file); aux_w, aux_h = aux.size[0], aux.size[1]
                cb = Image.open(cb_file); cb_w, cb_h = cb.size[0], cb.size[1]
                if aux_h / aux_w < target_aspect_ratio:
                    new_aux_h = int(target_aspect_ratio * aux_w)
                    aux = ImageOps.pad(aux, (aux_w, new_aux_h), centering=(0,0))
                    cb = ImageOps.pad(cb, (aux_w, new_aux_h), centering=(0,0))
                else:
                    new_aux_w = int(aux_h / target_aspect_ratio)
                    aux = ImageOps.pad(aux, (new_aux_w, aux_h), centering=(0,0))
                    cb = ImageOps.pad(cb, (new_aux_w, aux_h), centering=(0,0))

                aux = torch.from_numpy(np.array(aux.resize((img_w, img_h)))).unsqueeze(0).to(img[i].device).float()
                aux_list[i] = aux
                cb = torch.from_numpy(np.array(cb.resize((img_w, img_h)))).unsqueeze(0).to(img[i].device).float()
                cb_list[i] = cb

            # pdb.set_trace()
            cat_input = torch.cat([img, aux_list, cb_list], dim = 1)
            x = self.extract_feat(cat_input)

        else:

            x = self.extract_feat(img)

        losses = dict()
        
        loss_decode = self._decode_head_forward_train(x, img_metas,
                                                      gt_semantic_seg)
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(
                x, img_metas, gt_semantic_seg)
            losses.update(loss_aux)

        return losses

    # TODO refactor
    def slide_inference(self, img, img_meta, rescale):
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """

        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = img.size()
        num_classes = self.num_classes
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, num_classes, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                crop_seg_logit = self.encode_decode(crop_img, img_meta)
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        if torch.onnx.is_in_onnx_export():
            # cast count_mat to constant while exporting to ONNX
            count_mat = torch.from_numpy(
                count_mat.cpu().detach().numpy()).to(device=img.device)
        preds = preds / count_mat
        if rescale:
            preds = resize(
                preds,
                size=img_meta[0]['ori_shape'][:2],
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)
        return preds

    def whole_inference(self, img, img_meta, rescale):
        """Inference with full image."""

        seg_logit = self.encode_decode(img, img_meta)
        if rescale:
            # support dynamic shape for onnx
            if torch.onnx.is_in_onnx_export():
                size = img.shape[2:]
            else:
                size = img_meta[0]['ori_shape'][:2]
            seg_logit = resize(
                seg_logit,
                size=size,
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)

        return seg_logit

    def inference(self, img, img_meta, rescale):
        """Inference with slide/whole style.

        Args:
            img (Tensor): The input image of shape (N, 3, H, W).
            img_meta (dict): Image info dict where each dict has: 'img_shape',
                'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            rescale (bool): Whether rescale back to original shape.

        Returns:
            Tensor: The output segmentation map.
        """

        assert self.test_cfg.mode in ['slide', 'whole']
        ori_shape = img_meta[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in img_meta)
        if self.test_cfg.mode == 'slide':
            seg_logit = self.slide_inference(img, img_meta, rescale)
        else:
            seg_logit = self.whole_inference(img, img_meta, rescale)
        output = F.softmax(seg_logit, dim=1)
        flip = img_meta[0]['flip']
        if flip:
            flip_direction = img_meta[0]['flip_direction']
            assert flip_direction in ['horizontal', 'vertical']
            if flip_direction == 'horizontal':
                output = output.flip(dims=(3, ))
            elif flip_direction == 'vertical':
                output = output.flip(dims=(2, ))

        return output

    def simple_test(self, img, img_meta, rescale=True):
        """Simple test with single image."""
        seg_logit = self.inference(img, img_meta, rescale)
        seg_pred = seg_logit.argmax(dim=1)
        if torch.onnx.is_in_onnx_export():
            # our inference backend only support 4D output
            seg_pred = seg_pred.unsqueeze(0)
            return seg_pred
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred

    def aug_test(self, imgs, img_metas, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        # to save memory, we get augmented seg logit inplace
        seg_logit = self.inference(imgs[0], img_metas[0], rescale)
        for i in range(1, len(imgs)):
            cur_seg_logit = self.inference(imgs[i], img_metas[i], rescale)
            seg_logit += cur_seg_logit
        seg_logit /= len(imgs)
        seg_pred = seg_logit.argmax(dim=1)
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred
