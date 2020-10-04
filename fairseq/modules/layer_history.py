import torch
import torch.nn as nn
from fairseq.models.transformer import LayerNorm
import queue
import fairseq.utils as utils
import torch.nn.functional as F
import numpy as np


def CreateLayerHistory(args, is_encoder):
    history_type = args.encoder_history_type if is_encoder else args.decoder_history_type
    if history_type is None:
        return None
    elif history_type == "learnable_dense":
        return LearnableDenseLayerHistory(args, is_encoder)
    else:
        raise ValueError


class BaseLayerHistory(nn.Module):

    def __init__(self, args, is_encoder):
        super(BaseLayerHistory, self).__init__()
        self.is_encoder = is_encoder
        self.normalize_before = args.encoder_normalize_before if is_encoder else args.decoder_normalize_before

        # the first layer (aka. embedding layer) does not have layer normalization
        # layers = args.encoder_layers if is_encoder else args.decoder_layers
        layers = len(args.k) - 1 if is_encoder else args.decoder_layers
        dim = args.encoder_embed_dim if is_encoder else args.decoder_embed_dim
        self.layer_norms = nn.ModuleList(LayerNorm(dim) for _ in range(layers))

    def add(self, layer):
        raise NotImplemented

    def pop(self):
        raise NotImplemented

    def clean(self):
        raise NotImplemented


class LearnableDenseLayerHistory(BaseLayerHistory):
    """
    x_n = (x_1 + y_1 + y_2 + ... y_{n-1}) / n
    """

    def __init__(self, args, is_encoder):
        super(LearnableDenseLayerHistory, self).__init__(args, is_encoder)
        self.sum = None
        self.count = 0
        # self.layer_num = 1 + (args.encoder_layers if is_encoder else args.decoder_layers)
        self.layer_num = len(args.k) if is_encoder else args.decoder_layers + 1
        self.weight = nn.Parameter(torch.Tensor(self.layer_num, self.layer_num).fill_(1.0).tril())
        self.weight.data = self.weight.data / self.weight.data.sum(1, keepdim=True)

        # print('count:', len(list(self.named_parameters())))
        # for k,v in self.named_parameters():
        #    print('k=%s' %k)

    def extra_repr(self):
        return 'n_layers={layer_num}, '.format(**self.__dict__)

    def add(self, layer):
        self.count += 1

        # first layer
        if self.sum is None:
            self.sum = layer
            self.layers.append(layer)
            return

        # following layer
        if self.normalize_before:
            layer = self.layer_norms[self.count - 2](layer)

        self.layers.append(layer)

    def pop(self):
        assert len(self.layers) > 0
        # print(self.weight)
        # layers_dropout = F.dropout(torch.stack(self.layers, 0), p=self.dense_dropout, training=self.training)
        # ret = (layers_dropout * self.weight[self.count -1, : self.count].view(-1, 1, 1, 1)).sum(0)
        ret = (torch.stack(self.layers, 0) * self.weight[self.count - 1, : self.count].view(-1, 1, 1, 1)).sum(0)
        if self.count == 1 or self.normalize_before:
            return ret
        return self.layer_norms[self.count - 2](ret)

    def clean(self):
        self.sum = None
        self.count = 0
        self.layers = []

    def get_loss(self):
        return (0.5 * (self.weight.sum(1) - 1.0) ** 2).mean()

    def print_weight(self):
        print(self.weight)


