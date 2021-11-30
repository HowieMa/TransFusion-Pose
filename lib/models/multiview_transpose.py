# ------------------------------------------------------------------------------
# Copyright (c) Southeast University. Licensed under the MIT License.
# Written by Sen Yang (yangsenius@seu.edu.cn)
# cat two views together
# Self Attention on (2HW, 2HW)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import math

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from collections import OrderedDict
import copy
from typing import Optional, List


logger = logging.getLogger(__name__)
BN_MOMENTUM = 0.1

# ******************************************************************************************
# ************************************** CNN Backbone **************************************
# ******************************************************************************************


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


# ******************************************************************************************
# ************************************** Transformer ***************************************
# ******************************************************************************************


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers,
                 norm=None, pe_only_at_begin=False, return_atten_map=False):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.pe_only_at_begin = pe_only_at_begin
        self.return_atten_map = return_atten_map
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src
        atten_maps_list = []
        for layer in self.layers:
            if self.return_atten_map:
                output, att_map = layer(output, src_mask=mask, pos=pos,
                                        src_key_padding_mask=src_key_padding_mask)
                atten_maps_list.append(att_map)
            else:
                output = layer(output, src_mask=mask, pos=pos,
                               src_key_padding_mask=src_key_padding_mask)

            # only add position embedding to the first atttention layer
            pos = None if self.pe_only_at_begin else pos

        if self.norm is not None:
            output = self.norm(output)

        if self.return_atten_map:
            return output, torch.stack(atten_maps_list)
        else:
            return output


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class TransformerEncoderLayer(nn.Module):
    """ Modified from https://github.com/facebookresearch/detr/blob/master/models/transformer.py"""

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, return_atten_map=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.return_atten_map = return_atten_map

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        if self.return_atten_map:
            src2, att_map = self.self_attn(q, k, value=src,
                                           attn_mask=src_mask,
                                           key_padding_mask=src_key_padding_mask)
        else:
            src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                                  key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        if self.return_atten_map:
            return src, att_map
        else:
            return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        if self.return_atten_map:
            src2, att_map = self.self_attn(q, k, value=src,
                                           attn_mask=src_mask,
                                           key_padding_mask=src_key_padding_mask)
        else:
            src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                                  key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        if self.return_atten_map:
            return src, att_map
        else:
            return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransPoseR(nn.Module):

    def __init__(self, block, layers, cfg, **kwargs):
        self.inplanes = 64
        extra = cfg.NETWORK.EXTRA
        self.deconv_with_bias = extra.DECONV_WITH_BIAS

        super(TransPoseR, self).__init__()
        # ========================== Backbone ==========================
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)

        # ========================== Transformer ==========================
        w, h = cfg.NETWORK.IMAGE_SIZE  # (256, 256)
        d_model = cfg.NETWORK.DIM_MODEL
        dim_feedforward = cfg.NETWORK.DIM_FEEDFORWARD
        encoder_layers_num = cfg.NETWORK.ENCODER_LAYERS
        n_head = cfg.NETWORK.N_HEAD
        pos_embedding_type = cfg.NETWORK.POS_EMBEDDING

        self.fusion = cfg.NETWORK.FUSION        # whether fuse two views or not
        if self.fusion:
            logger.info('====> Multi-View Fusion Transformer')
        else:
            logger.info('====> Single-View Fusion Transformer')

        self.reduce = nn.Conv2d(self.inplanes, d_model, 1, bias=False)

        # 2D embedding. Default: sine
        self._make_position_embedding(w, h, d_model, pos_embedding_type)

        # 3D position embedding
        self.pos_emb_3d_type = cfg.NETWORK.POS_EMB_3D
        if self.pos_emb_3d_type == 'learnable':
            logger.info('==> 3D Position Encoding: Learnable')
            self.pos_3d = nn.Parameter((torch.randn(2, self.pe_w * self.pe_h, 1, d_model)))  # 2 is the number of views
        elif self.pos_emb_3d_type == 'geometry':
            logger.info('==> 3D Position Encoding: Geometry')
            self.pos_3d_linear = nn.Linear(3, 256)
        else:
            logger.info('==> 3D Position Encoding: None')
            self.pos_3d = None

        # Transformer Layer
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            activation='relu',
            return_atten_map=False
        )
        self.global_encoder = TransformerEncoder(
            encoder_layer,
            encoder_layers_num,
            return_atten_map=False
        )

        # ========================== Output Layer ==========================
        self.inplanes = d_model
        self.deconv_layers = self._make_deconv_layer(1, [256], [4])

        self.final_layer = nn.Conv2d(
            in_channels=d_model,
            out_channels=cfg.NETWORK.NUM_JOINTS,
            kernel_size=extra.FINAL_CONV_KERNEL,
            stride=1,
            padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0
        )

    def _make_position_embedding(self, w, h, d_model, pe_type='sine'):
        assert pe_type in ['none', 'learnable', 'sine']
        if pe_type == 'none':
            self.pos_embedding = None
            logger.info("==> Without any PositionEmbedding~")
        else:
            with torch.no_grad():
                self.pe_h = h // 8
                self.pe_w = w // 8
                length = self.pe_h * self.pe_w
            if pe_type == 'learnable':
                self.pos_embedding = nn.Parameter(
                    torch.randn(length, 1, d_model))
                logger.info("==> Add Learnable PositionEmbedding~")
            else:
                self.pos_embedding = nn.Parameter(
                    self._make_sine_position_embedding(d_model),
                    requires_grad=False)
                logger.info("==> Add Sine PositionEmbedding~")

    def _make_sine_position_embedding(self, d_model, temperature=10000,
                                      scale=2 * math.pi):
        # logger.info(">> NOTE: this is for testing on unseen input resolutions")
        # # NOTE generalization test with interploation
        # self.pe_h, self.pe_w = 256 // 8 , 192 // 8 #self.pe_h, self.pe_w
        h, w = self.pe_h, self.pe_w
        area = torch.ones(1, h, w)  # [b, h, w]
        y_embed = area.cumsum(1, dtype=torch.float32)
        x_embed = area.cumsum(2, dtype=torch.float32)

        one_direction_feats = d_model // 2

        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * scale
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * scale

        dim_t = torch.arange(one_direction_feats, dtype=torch.float32)
        dim_t = temperature ** (2 * (dim_t // 2) / one_direction_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        pos = pos.flatten(2).permute(2, 0, 1)
        return pos  # [h*w, 1, d_model]

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def backbone_forward(self, x):
        # x: (B, 3, 256, 256)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.reduce(x)
        return x

    def single_view_forward(self, x):
        # reduce to TransPose
        x = self.backbone_forward(x)                        # (B, C=256, H=32, W=32)
        bs, c, h, w = x.shape
        x = x.flatten(2).permute(2, 0, 1)                   # (HW, B, C)
        x = self.global_encoder(x, pos=self.pos_embedding)  # (HW, B, C)
        x = x.permute(1, 2, 0).contiguous().reshape(bs, c, h, w)
        x = self.final_layer(self.deconv_layers(x))
        return x

    def forward(self, views, rays=None, centers=None):
        if isinstance(views, list):
            outputs, pos_embs_3d_out = [], []
            # *************************** TransPose on single image ****************************
            if not self.fusion:
                # Transformer on single views
                for i, view in enumerate(views):
                    heatmap = self.single_view_forward(view)     # (bs, num_joints, 64, 64)
                    outputs.append(heatmap)

            # *************************** TransPose Fusion ****************************
            else:
                features, pos_embs = [], []
                num_views = len(views)
                for i, view in enumerate(views):
                    # ======================= visual features =======================
                    x = self.backbone_forward(view)                     # (B, C=256, H=32, W=32)
                    bs, c, h, w = x.shape
                    features.append(x.flatten(2).permute(2, 0, 1))       # (HW, B, C)

                    # ======================= Position embeddings =======================
                    #  2D sine embedding
                    pos_emb_2d = self.pos_embedding.repeat(1, bs, 1)     # (HW, B, C)

                    # 3D position embedding
                    if self.pos_emb_3d_type == 'learnable':
                        pos_emb_3d = self.pos_3d[i]               # (HW, 1, 256)
                        pos_embs.append(pos_emb_2d + pos_emb_3d)  # (HW, B, C)    combine PE

                    elif self.pos_emb_3d_type == 'geometry':        # 3D geometry PE
                        vec_c_p = F.normalize(rays[i] - centers[i], dim=2, p=2)  # (B, HW, 3)
                        pos_emb_3d = self.pos_3d_linear(vec_c_p)             # (B, HW, 3) -> (B, HW, 256)
                        pos_embs_3d_out.append(pos_emb_3d)                  # save for loss calculation

                        pos_emb_3d = pos_emb_3d.permute(1, 0, 2)            # (HW, B, C=256)
                        pos_embs.append(pos_emb_2d + pos_emb_3d)            # (HW, B, C)    combine PE

                    else:
                        pos_embs.append(pos_emb_2d)                          # (HW, B, C)

                feat_embs = torch.cat(features, dim=0)          # (#view * HW, B, C)
                pos_embs = torch.cat(pos_embs, dim=0)           # (#view * HW, B, C)

                # ======================= Transformer Fusion =======================
                feat_embs = self.global_encoder(feat_embs, pos=pos_embs)      # (#view * HW, B, C)
                feat_embs = feat_embs.chunk(num_views, dim=0)     # list, (HW, B, C), length: num views

                # ======================= Deconvolution =======================
                for x in feat_embs:
                    x = x.permute(1, 2, 0).contiguous().reshape(bs, c, h, w)
                    x = self.final_layer(self.deconv_layers(x))
                    outputs.append(x)  # (B, num_joints, 64, 64)
            
            return outputs, pos_embs_3d_out
        else:
            return self.single_view_forward(views)      # (B, num_joints, 64, 64)

    def init_weights(self, pretrained=''):
        if os.path.isfile(pretrained):
            logger.info('=> init final conv weights from normal distribution')
            for name, m in self.final_layer.named_modules():
                if isinstance(m, nn.Conv2d):
                    logger.info(
                        '=> init {}.weight as normal(0, 0.001)'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.normal_(m.weight, std=0.001)
                    nn.init.constant_(m.bias, 0)

            pretrained_state_dict = torch.load(pretrained)
            logger.info('=> loading Pretrained model {}'.format(pretrained))
            existing_state_dict = {}
            for name, m in pretrained_state_dict.items():
                if name in self.state_dict():
                    if 'final_layer' in name or 'pos_embedding' in name:
                        continue
                    existing_state_dict[name] = m
                    logger.info(":: {} is loaded from {}".format(name, pretrained))
            self.load_state_dict(existing_state_dict, strict=False)
        else:
            logger.info(
                '=> NOTE :: Pretrained Weights {} are not loaded ! Please Download it'.format(pretrained))
            logger.info('=> init weights from normal distribution')
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, std=0.001)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.ConvTranspose2d):
                    nn.init.normal_(m.weight, std=0.001)
                    if self.deconv_with_bias:
                        nn.init.constant_(m.bias, 0)


resnet_spec = {18: (BasicBlock, [2, 2, 2, 2]),
               34: (BasicBlock, [3, 4, 6, 3]),
               50: (Bottleneck, [3, 4, 6, 3]),
               101: (Bottleneck, [3, 4, 23, 3]),
               152: (Bottleneck, [3, 8, 36, 3])}


def get_multiview_pose_net(cfg, is_train, **kwargs):
    num_layers = cfg.NETWORK.EXTRA.NUM_LAYERS
    block_class, layers = resnet_spec[num_layers]
    model = TransPoseR(block_class, layers, cfg, **kwargs)

    if is_train and cfg.NETWORK.INIT_WEIGHTS:
        model.init_weights(cfg.NETWORK.PRETRAINED)

    return model

