import copy
import logging
import math
import os
import json

from os.path import join as pjoin

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_fscore_support
from torch.nn.modules.utils import _pair
from scipy import ndimage
import seaborn as sns
import torchvision.transforms as transforms


train_cats = [
    'hair drier', 'clock', 'wine glass', 'book', 'cake', 'tie', 'motorcycle',
    'sheep', 'bottle', 'giraffe', 'cell phone', 'suitcase', 'remote', 'bench',
    'mouse', 'carrot', 'banana', 'train', 'sports ball', 'toothbrush', 'fire hydrant',
    'airplane', 'tv', 'bus', 'refrigerator', 'couch', 'knife', 'toilet', 'elephant',
    'truck', 'parking meter', 'car', 'potted plant', 'kite', 'skateboard', 'orange',
    'horse', 'cat', 'tennis racket', 'bowl', 'scissors', 'baseball glove', 'apple',
    'traffic light', 'handbag', 'donut', 'dog', 'hot dog', 'oven', 'umbrella', 'sink',
    'pizza'
]
val_cats = [
    'cow', 'dining table', 'zebra', 'sandwich', 'bear', 'toaster', 'person',
    'laptop', 'bed', 'teddy bear', 'baseball bat', 'skis'
]
test_cats = [
    'bicycle', 'boat', 'stop sign', 'bird', 'backpack', 'frisbee', 'snowboard',
    'surfboard', 'cup', 'fork', 'spoon', 'broccoli', 'chair', 'keyboard', 'microwave',
    'vase'
]

class WordEmbeddings:
    def __init__(self, phase, text_vector_path, embed_size = 256, add_tokens = True, trainable = True, add_end = False):
        super(WordEmbeddings, self).__init__()

        assert phase in ['train', 'val', 'test']
        cats = {'train': train_cats, 'val': val_cats, 'test': test_cats}[phase]

        self.cats = cats
        self.trainable = trainable
        if add_tokens:
            self.cats.append('<end>')
            self.cats.append('<start>')
            self.cats.append('<pad>')
        if add_end:
            self.cats.append('<end>')

        self.label_to_vector = {}
        self.construct_label()

        self.word_dim = 0
        with open(text_vector_path, 'r') as f:
            for num, line in enumerate(f):
                line = line.rstrip('\r\n')
                data_line = json.loads(line)
                for key, value in data_line.items():
                    if isinstance(value, str):
                        value = eval(value)
                    if key in self.cats:
                        self.label_to_vector[self.cat_to_label[key]] = value
                        if self.word_dim == 0:
                            self.word_dim = len(value)
        self.vocab_size = len(self.cats)

    def construct_label(self):
        self.label_to_cat = dict((i, cat) for i, cat in enumerate(self.cats))
        self.cat_to_label = dict((cat, i) for i, cat in enumerate(self.cats))

    def get_word_embeddings(self, cat):
        label = self.cat_to_label[cat]
        return self.label_to_vector[label]

    def get_word_dim(self):
        return self.word_dim

    def construct_embeddings(self):
        embedding = torch.rand(self.vocab_size, self.word_dim)
        for i in range(len(self.cats)):
            text_vector = torch.Tensor(self.label_to_vector[i])
            embedding[i] = text_vector
        # modified by yk
        if self.trainable:
            embedding = nn.Embedding.from_pretrained(embedding)
            return embedding
        return embedding.cuda()

def swish(x):
    return x * torch.sigmoid(x)

ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}

class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        # self.num_attention_heads = config.transformer["num_heads"]
        self.num_attention_heads = config.num_heads
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        # self.out = Linear(config.hidden_size, config.hidden_size)
        self.out = Linear(self.all_head_size, config.hidden_size)
        self.attn_dropout = Dropout(config.attention_dropout_rate)
        self.proj_dropout = Dropout(config.attention_dropout_rate)

        self.softmax = Softmax(dim=-1)

        self.iter_count = 0

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, vis_local_support_i, tex_fea_i):
        mixed_query_layer = self.query(tex_fea_i)
        mixed_key_layer = self.key(vis_local_support_i)
        mixed_value_layer = self.value(vis_local_support_i)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)


        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)

        return attention_output, weights


class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.mlp_dim)
        self.fc2 = Linear(config.mlp_dim, config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.dropout_rate)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, config):
        super(ChannelAttention, self).__init__()
        inter_dim = int(config.hidden_size / 2)
        self.fc1 = Linear(config.hidden_size, inter_dim)
        self.fc2 =Linear(inter_dim, config.hidden_size)
        self.atten_dropout = Dropout(config.attention_dropout_rate)
        self.act_fn = ACT2FN["gelu"]

    def forward(self, vis_local_support_i, tex_fea_i):
        bs, loc_num, hidden_size = vis_local_support_i.shape
        tex_fea_i = tex_fea_i.expand(bs, loc_num, hidden_size)

        mixed_feature = vis_local_support_i + tex_fea_i
        x = self.fc1(mixed_feature)
        x = self.fc2(x)
        weights = torch.sigmoid(x)

        new_local_feature = vis_local_support_i * weights
        new_local_feature = new_local_feature + vis_local_support_i
        new_local_feature =  self.atten_dropout(new_local_feature)
        return new_local_feature

class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.attn = Attention(config, vis)
        self.ffn = Mlp(config)

        if config.channel_atten:
            self.channel_atten = ChannelAttention(config)

    def forward(self, x, query):
        # h = x
        x = self.attention_norm(x)
        query = self.attention_norm(query) # new
        if self.config.channel_atten:
            x = self.channel_atten(x, query)
        x, weights = self.attn(x, query)

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

class CrossModalAtten(nn.Module):
    def __init__(self, config, use_all_loc = False):
        super(CrossModalAtten, self).__init__()
        self.config = config
        self.use_all_loc = use_all_loc
        self.vis = config.vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.dimension_fc = nn.Conv2d(in_channels=2048, out_channels=config.hidden_size, kernel_size=1)
        for _ in range(config.num_layers):
            layer = Block(config, self.vis)
            self.layer.append(copy.deepcopy(layer))

        self.dropout = Dropout(config.dropout_rate)

    def forward(self, vis_local_support_i, tex_fea_i, scores = None):
        attn_weights = []
        vis_local_support_i = self.dimension_fc(vis_local_support_i)
        vis_local_support_i = vis_local_support_i.flatten(2)
        vis_local_support_i = vis_local_support_i.transpose(-1, -2)
        vis_local_support_i = self.dropout(vis_local_support_i)

        if self.use_all_loc:
            bs, loc_num, cha = vis_local_support_i.size()
            if scores == None:
                vis_local_support_i = vis_local_support_i.contiguous().view(1, bs * loc_num, cha)
            else:
                scores = scores.contiguous().view(bs, loc_num, 1).expand(bs, loc_num, cha)
                vis_local_support_i = vis_local_support_i * scores
                vis_local_support_i = vis_local_support_i.contiguous().view(bs * loc_num, -1)
                sum_cur = torch.sum(vis_local_support_i, dim = 1)
                keep_idx = torch.where(sum_cur != 0)
                keep_local = vis_local_support_i[keep_idx]
                loc_num = keep_local.size()[0]
                vis_local_support_i = keep_local.contiguous().view(1, loc_num, cha)

        bs, loc_num, cha = vis_local_support_i.size()
        tex_fea_i = tex_fea_i.contiguous().view(-1, cha)
        tex_fea_i = tex_fea_i.expand(bs, 1, cha)

        for layer_block in self.layer:
            vis_local_support_i, weights = layer_block(vis_local_support_i, tex_fea_i)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(vis_local_support_i)

        vis_local_support_i = encoded[:, 0]

        return vis_local_support_i, attn_weights

        
class DyCovLayer(nn.Module):
    def __init__(self, d_model, nheads, dropout, dim_feedforward, pre_norm):
        super().__init__()

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.dy_conv = DynamicConvMask(
            in_channels = d_model,
            feat_channels = 64,
            out_channels = d_model,
            input_feat_shape = 8,
            act_cfg=dict(type='ReLU', inplace=True),
            norm_cfg=dict(type='LN'),
        )
        self.transformer_self_attention_layer = SelfAttentionLayer(
            d_model=d_model,
            nhead=nheads,
            dropout=0.0,
            normalize_before=pre_norm,
        )
        self.transformer_ffn_layer = FFNLayer(
            d_model=d_model,
            dim_feedforward=dim_feedforward,
            dropout=0.0,
            normalize_before=pre_norm,
        )


    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory, attn_mask_target_size,
                     memory_mask: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        tgt2 = self.dy_conv(param_feature_all=self.with_pos_embed(tgt, query_pos),
                                   input_feature_all=memory, atten_mask_all=memory_mask, atten_mask_size = attn_mask_target_size)
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward(self, tgt, memory, attn_mask_target_size,
                memory_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):

        output = self.forward_post(
            tgt,
            memory,
            attn_mask_target_size,
            memory_mask,
            query_pos
        )

        output = self.transformer_self_attention_layer(
            output, tgt_mask = None,
            tgt_key_padding_mask=None,
            query_pos=query_pos,
        )

        output = self.transformer_ffn_layer(
            output
        )

        return output


class Model(nn.Module):

    def __init__(self, num_classes, embed_size, phase, text_vector_path, trans_config,
                topk = 1, k_shot = 1, use_all_loc = False, finetune=True, backbone = 'res50'):
        super().__init__()
        self.bb = backbone
        if backbone == 'res50':
            self.backbone = nn.Sequential(*list(models.resnet50(pretrained=True).children())[: -2])
        else:
            self.backbone = nn.Sequential(*list(models.resnet101(pretrained=True).children())[: -2])
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.topk = topk
        self.finetune = finetune
        self.use_all_loc = use_all_loc
        self.k_shot = k_shot
        self.num_classes = num_classes

        self.embed_size = embed_size
        embedding_manager = WordEmbeddings(phase, text_vector_path, add_tokens = False, trainable = False)
        self.embedding = embedding_manager.construct_embeddings()
        word_dim = embedding_manager.get_word_dim()

        self.vis_fc = Linear(2048, self.embed_size)
        self.tex_fc = Linear(word_dim, self.embed_size)

        self.crossmodal_atten = CrossModalAtten(trans_config, self.use_all_loc)
        self.purify_coefficient = torch.nn.Parameter(torch.ones(num_classes * k_shot, 7, 7))
        self.register()
        self.scores = None
        self.iters = 0

    def hook(self, grad):
        temp_alpha = min(1 - 1 / (self.iters + 1), 0.95)
        if self.scores == None:
            self.scores = (grad * self.purify_coefficient).abs()
        else:
            self.scores = temp_alpha * self.scores + (1 - temp_alpha) * (grad * self.purify_coefficient).abs()

    def register(self):
        self.purify_coefficient.register_hook(self.hook)

    def forward(self, batch_images, batch_targets):
        n_way, n_img, *_  = batch_images.size()
        batch_images = batch_images.contiguous().view(n_way * n_img, *batch_images.size()[2:])
        local_fea = self.backbone(batch_images) # local_fea.shape = (bs, 2048, 7, 7)
        _, l_c, l_w, l_h = local_fea.size()
        vis_fea = self.avgpool(local_fea)
        vis_fea = vis_fea.contiguous().view(n_way * n_img, -1)
        vis_fea = self.vis_fc(vis_fea)
        vis_fea = vis_fea.contiguous().view(n_way, n_img, -1)
        local_fea = local_fea.contiguous().view(n_way, n_img, l_c, l_w, l_h)

        k_shot = self.k_shot
        vis_support = vis_fea[:, 0:k_shot].contiguous().view(n_way * k_shot, -1)
        vis_query = vis_fea[:, k_shot:].contiguous().view(n_way * (n_img - k_shot), -1)

        vis_local_support = local_fea[:, 0:k_shot].contiguous().view(n_way * k_shot, l_c, l_w, l_h)

        tex_fea = self.tex_fc(self.embedding)

        support_labels = batch_targets[:, 0:k_shot].contiguous().view(n_way * k_shot, -1)
        query_labels = batch_targets[:, k_shot:].contiguous().view(n_way * (n_img - k_shot), -1)

        support_dist = self.dis(vis_support, tex_fea)
        support_scores = -support_dist
        support_scores = (support_scores + 1) * 20
        loss_support = F.binary_cross_entropy_with_logits(support_scores, support_labels)

        prototypes = []
        for i in range(self.num_classes):
            weight_i = support_scores[:, i]
            if self.topk != 'all':
                topk_vals, topk_inds = torch.topk(weight_i, self.topk)
            else:
                topk_inds = torch.where(support_labels[:, i] > 0)
                topk_inds = topk_inds[0]
                label_num = torch.sum(support_labels[:, i])

            if len(topk_inds) == 0:
                prototype_i = tex_fea[i].contiguous().view(1, self.embed_size)
            else:
                weight_i = weight_i[topk_inds]
                vis_num = len(topk_inds)
                weight_i = F.softmax(weight_i, dim = -1).contiguous().view(vis_num, -1)
                vis_local_support_i = vis_local_support[topk_inds]
                new_vis_support_i, attn_weights = self.crossmodal_atten(vis_local_support_i, tex_fea[i])
                if self.use_all_loc:
                    prototype_i = new_vis_support_i
                else:
                    prototype_i = new_vis_support_i * weight_i
                    prototype_i = prototype_i.sum(0).contiguous().view(-1, self.embed_size)

            prototypes.append(prototype_i)
        prototypes = torch.cat(prototypes, dim = 0)
        query_dist = self.dis(vis_query, prototypes)
        query_scores = -query_dist
        query_scores = (query_scores + 1) * 20
        loss_query = F.binary_cross_entropy_with_logits(query_scores, query_labels)

        loss = loss_query + loss_support
        return ['BCEw/logits', ], [round(loss.item(), 4), ], loss

    def generate_prototypes(self, batch_images, batch_targets = None):
        n_way, *_ = batch_images.size()
        n_way = int(n_way / self.k_shot)
        support_labels = batch_targets
        local_fea = self.backbone(batch_images)
        _, l_c, l_w, l_h = local_fea.size()
        vis_fea = self.avgpool(local_fea)
        vis_fea = vis_fea.contiguous().view(n_way * self.k_shot, -1)
        vis_support = self.vis_fc(vis_fea)

        tex_fea = self.tex_fc(self.embedding)

        support_dist = self.dis(vis_support, tex_fea)
        support_scores = -support_dist
        support_scores = (support_scores + 1) * 20

        prototypes = []
        for i in range(n_way):
            weight_i = support_scores[:, i]
            if self.topk != 'all':
                topk_vals, topk_inds = torch.topk(weight_i, self.topk)
            else:
                topk_inds = torch.where(support_labels[:, i] > 0)
                topk_inds = topk_inds[0]

            assert len(topk_inds) != 0
            if len(topk_inds) == 0:
                prototype_i = torch.rand(1, self.embed_size)
                prototype_i = prototype_i.cuda()
            else:
                weight_i = weight_i[topk_inds]
                vis_num = len(topk_inds)
                weight_i = F.softmax(weight_i, dim = -1).contiguous().view(vis_num, -1)
                vis_local_support_i = local_fea[topk_inds]
                cur_scores =  torch.where(torch.sigmoid(self.scores[topk_inds] * 1e+8) < 0.65, 0, 1)
                cur_scores = cur_scores.cuda()
                new_vis_support_i, attn_weights = self.crossmodal_atten(vis_local_support_i, tex_fea[i], cur_scores)
                if self.use_all_loc:
                    prototype_i = new_vis_support_i
                else:
                    prototype_i = new_vis_support_i * weight_i
                    prototype_i = prototype_i.sum(0).contiguous().view(-1, self.embed_size)

            prototypes.append(prototype_i)
        prototypes = torch.cat(prototypes, dim = 0) # [n_way, embed_size]

        self.prototypes = prototypes


    def generate_prototypes_a(self, batch_images, batch_targets = None):
        n_way, n_img, *_  = batch_images.size()
        batch_images = batch_images.contiguous().view(n_way * n_img, *_)
        support_labels = batch_targets.contiguous().view(n_way * n_img, -1)

        local_fea = self.backbone(batch_images)
        _, l_c, l_w, l_h = local_fea.size()
        vis_fea = self.avgpool(local_fea)
        vis_fea = vis_fea.contiguous().view(n_way * n_img, -1)
        vis_support = self.vis_fc(vis_fea)

        tex_fea = self.tex_fc(self.embedding) #[voc, embed_size]

        support_dist = self.dis(vis_support, tex_fea)
        support_scores = -support_dist
        support_scores = (support_scores + 1) * 20

        prototypes = []
        for i in range(n_way):
            weight_i = support_scores[:, i]
            if self.topk != 'all':
                topk_vals, topk_inds = torch.topk(weight_i, self.topk)
            else:
                topk_inds = torch.where(support_labels[:, i] > 0)
                topk_inds = topk_inds[0]
            if len(topk_inds) == 0:
                prototype_i = torch.rand(1, self.embed_size)
                prototype_i = prototype_i.cuda()
            else:
                weight_i = weight_i[topk_inds]
                vis_num = len(topk_inds)
                weight_i = F.softmax(weight_i, dim = -1).contiguous().view(vis_num, -1)
                vis_local_support_i = local_fea[topk_inds]
                new_vis_support_i, attn_weights = self.crossmodal_atten(vis_local_support_i, tex_fea[i])
                if self.use_all_loc:
                    prototype_i = new_vis_support_i
                else:
                    prototype_i = new_vis_support_i * weight_i
                    prototype_i = prototype_i.sum(0).contiguous().view(-1, self.embed_size)
            prototypes.append(prototype_i)
        prototypes = torch.cat(prototypes, dim = 0)

        return prototypes


    def infer_proto(self, batch_images):
        bs, *_ = batch_images.size()
        vis_fea = self.backbone(batch_images)
        vis_fea = self.avgpool(local_fea)
        vis_fea = vis_fea.contiguous().view(bs, -1)
        vis_query = self.vis_fc(vis_fea)

        query_dist = self.dis(vis_query, self.prototypes)
        query_scores = -query_dist
        query_scores = (query_scores + 1) * 20

        output = query_scores

        return output.sigmoid()

    def meta_train(self, batch_images, batch_targets):
        self.iters += 1
        if self.purify_coefficient[0, 0, 0] == self.purify_coefficient[0, 0, 1] == 1.0:
            pass
        else:
            purify_coefficient = self.purify_coefficient.contiguous().view(self.num_classes * self.k_shot, -1)
            min_coe_all = torch.min(purify_coefficient, dim = 1)[0]
            min_coe = min_coe_all.view(self.num_classes * self.k_shot, -1)
            max_coe_all = torch.max(purify_coefficient, dim = 1)[0]
            max_coe = max_coe_all.view(self.num_classes * self.k_shot, -1)
            purify_coefficient = (purify_coefficient - min_coe) / (max_coe - min_coe)
            sec_min = torch.sort(purify_coefficient, dim = 1)[0][:, 1]
            sec_min = sec_min.view(self.num_classes * self.k_shot, -1)
            sec_min = sec_min.expand(self.num_classes * self.k_shot, 49)
            purify_coefficient = torch.where(purify_coefficient > 0.0, purify_coefficient, sec_min)
            self.purify_coefficient.data = purify_coefficient.contiguous().view(-1, 7, 7)

        if len(batch_images.size()) > 4:
            n_way, n_img, *_  = batch_images.size()
        else:
            n_way, *_ = batch_images.size()
            n_img = 1
        k_shot = self.k_shot

        batch_images = batch_images.contiguous().view(n_way * n_img, *_)
        support_labels = batch_targets.contiguous().view(n_way * n_img, -1)
        vis_fea = self.backbone(batch_images)
        bs, cha, w, h = vis_fea.size()
        vis_fea = vis_fea.permute(0, 2, 3, 1)

        vis_fea = vis_fea * self.purify_coefficient.view(bs, w, h, 1)
        vis_fea = vis_fea.permute(0, 3, 1, 2)
        vis_fea = self.avgpool(vis_fea)
        vis_fea = vis_fea.contiguous().view(n_way * n_img, -1)
        vis_support = self.vis_fc(vis_fea)

        tex_fea = self.tex_fc(self.embedding) #[voc, embed_size]

        support_dist = self.dis(vis_support, tex_fea)
        support_scores = -support_dist
        support_scores = (support_scores + 1) * 20

        loss= F.binary_cross_entropy_with_logits(support_scores, support_labels)

        return ['BCEw/logits', ], [round(loss.item(), 4), ], loss

    def infer(self, batch_images, batch_targets):
        n_way, n_img, *_  = batch_images.size()
        labels = batch_targets.contiguous().view(n_way * n_img, -1)
        batch_images = batch_images.contiguous().view(n_way * n_img, *_)

        vis_fea = self.backbone(batch_images)
        vis_fea = self.avgpool(vis_fea)
        vis_fea = vis_fea.contiguous().view(n_way * n_img, -1)
        vis_query = self.vis_fc(vis_fea)

        query_dist = self.dis(vis_query, self.prototypes)
        query_scores = -query_dist
        query_scores = (query_scores + 1) * 20

        output = query_scores.sigmoid()

        return output, labels

    def train(self, mode=True):
        if self.finetune:
            self.backbone.train(mode)
            self.crossmodal_atten.train(mode)
            self.vis_fc.train(mode)
            self.tex_fc.train(mode)
        else:
            self.backbone.train(False)
            self.crossmodal_atten.train(False)
            self.vis_fc.train(False)
            self.tex_fc.train(False)

    def dis(self, x, y):
        # x: N x D
        # y: M x D
        x = F.normalize(x, p=2, dim=1, eps=1e-12)
        y = F.normalize(y, p=2, dim=1, eps=1e-12)

        n = x.size(0)
        m = y.size(0)
        d = x.size(1)

        assert d == y.size(1)

        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)

        dis_ = torch.pow(x - y, 2).sum(2).sqrt()

        return dis_
