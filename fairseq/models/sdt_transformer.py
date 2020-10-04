# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from fairseq import options, utils
from fairseq.modules import (
    AdaptiveInput, AdaptiveSoftmax, CharacterTokenEmbedder, LayerNorm,
    LearnedPositionalEmbedding, MultiheadAttention, SinusoidalPositionalEmbedding,
    RelativeMultiheadAttention,
)

from . import (
    FairseqIncrementalDecoder, FairseqEncoder, FairseqLanguageModel,
    FairseqModel, register_model, register_model_architecture,
)
from fairseq.modules.layer_history import CreateLayerHistory
import fairseq.utils as util
import numpy as np

np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=np.inf)

@register_model('sdt_transformer')
class SdtTransformerModel(FairseqModel):
    """
    Transformer model from `"Attention Is All You Need" (Vaswani, et al, 2017)
    <https://arxiv.org/abs/1706.03762>`_.

    Args:
        encoder (TransformerEncoder): the encoder
        decoder (TransformerDecoder): the decoder

    The Transformer model provides the following named architectures and
    command-line arguments:

    .. argparse::
        :ref: fairseq.models.transformer_parser
        :prog:
    """

    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--attention-dropout', type=float, metavar='D',
                            help='dropout probability for attention weights')
        parser.add_argument('--relu-dropout', type=float, metavar='D',
                            help='dropout probability after ReLU in FFN')
        parser.add_argument('--encoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained encoder embedding')
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension')
        parser.add_argument('--encoder-ffn-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension for FFN')
        parser.add_argument('--encoder-layers', type=int, metavar='N',
                            help='num encoder layers')
        parser.add_argument('--encoder-attention-heads', type=int, metavar='N',
                            help='num encoder attention heads')
        parser.add_argument('--encoder-normalize-before', action='store_true',
                            help='apply layernorm before each encoder block')
        parser.add_argument('--encoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the encoder')
        parser.add_argument('--decoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained decoder embedding')
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension')
        parser.add_argument('--decoder-ffn-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension for FFN')
        parser.add_argument('--decoder-layers', type=int, metavar='N',
                            help='num decoder layers')
        parser.add_argument('--decoder-attention-heads', type=int, metavar='N',
                            help='num decoder attention heads')
        parser.add_argument('--decoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the decoder')
        parser.add_argument('--decoder-normalize-before', action='store_true',
                            help='apply layernorm before each decoder block')
        parser.add_argument('--share-decoder-input-output-embed', action='store_true',
                            help='share decoder input and output embeddings')
        parser.add_argument('--share-all-embeddings', action='store_true',
                            help='share encoder, decoder and output embeddings'
                                 ' (requires shared dictionary and embed dim)')
        parser.add_argument('--no-token-positional-embeddings', default=False, action='store_true',
                            help='if set, disables positional embeddings (outside self attention)')
        parser.add_argument('--adaptive-softmax-cutoff', metavar='EXPR',
                            help='comma separated list of adaptive softmax cutoff points. '
                                 'Must be used with adaptive_loss criterion'),
        parser.add_argument('--adaptive-softmax-dropout', type=float, metavar='D',
                            help='sets adaptive softmax dropout for the tail projections')
        parser.add_argument('--max-relative-length', type=int, default=-1,
                            help='the max relative length')
        parser.add_argument('--k-only', default=False, action='store_true',
                            help='select the relative mode to map relative position information')
        # fmt: on

        ### dense layer parameters
        parser.add_argument('--encoder-history-type',
                            help='encoder layer history type')
        parser.add_argument('--decoder-history-type',
                            help='decoder layer history type')
        parser.add_argument('--encoder-integration-type', choices=['avg', 'sum'],
                            help='encoder layer integration type')
        parser.add_argument('--decoder-integration-type', choices=['avg', 'sum'],
                            help='decoder layer integration type')

        parser.add_argument('--inspect-grad', default=False, action='store_true',
                            help='inspect intermediate gradient')

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        if not hasattr(args, 'max_source_positions'):
            args.max_source_positions = 1024
        if not hasattr(args, 'max_target_positions'):
            args.max_target_positions = 1024

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        def build_embedding(dictionary, embed_dim, path=None):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx)
            # if provided, load from preloaded dictionaries
            if path:
                embed_dict = utils.parse_embedding(path)
                utils.load_embedding(embed_dict, dictionary, emb)
            return emb

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError('--share-all-embeddings requires a joined dictionary')
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    '--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim')
            if args.decoder_embed_path and (
                    args.decoder_embed_path != args.encoder_embed_path):
                raise ValueError('--share-all-embeddings not compatible with --decoder-embed-path')
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = build_embedding(
                tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
            )

        encoder = TransformerEncoder(args, src_dict, encoder_embed_tokens)
        decoder = TransformerDecoder(args, tgt_dict, decoder_embed_tokens)
        return SdtTransformerModel(encoder, decoder)


class TransformerEncoder(FairseqEncoder):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
        left_pad (bool, optional): whether the input is left-padded
            (default: True).
    """

    def __init__(self, args, dictionary, embed_tokens, left_pad=True):
        super().__init__(dictionary)
        self.dropout = args.dropout
        self.k = args.k
        self.count = 0
        #self.attn_weight = []

        #self.Sigmoid = torch.nn.Sigmoid()
        #self.alpha = nn.Parameter(torch.zeros(1))

        embed_dim = embed_tokens.embedding_dim
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = args.max_source_positions

        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(embed_dim)
        self.embed_positions = PositionalEmbedding(
            args.max_source_positions, embed_dim, self.padding_idx,
            left_pad=left_pad,
            learned=args.encoder_learned_pos,
        ) if not args.no_token_positional_embeddings else None

        # create encoder layer history
        self.history = CreateLayerHistory(args, is_encoder=True)
        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerEncoderLayer(args)
            for i in range(args.encoder_layers)
        ])
        self.register_buffer('version', torch.Tensor([2]))
        self.normalize = args.encoder_normalize_before
        if self.normalize:
            self.layer_norm = LayerNorm(embed_dim)

        self.inspected_grads = OrderedDict() if getattr(args, 'inspect_grad', False) else None
        self.inspected_grads_qkv = OrderedDict() if getattr(args, 'inspect_grad', False) else None
        self.inspected_grads_ffn_out = OrderedDict() if getattr(args, 'inspect_grad', False) else None

    def forward(self, src_tokens, src_lengths):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`

        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
        """
        if self.history is not None:
            self.history.clean()
        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(src_tokens)
        if self.embed_positions is not None:
            x += self.embed_positions(src_tokens)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        util.inspect_grad("encoder_0", x, self.inspected_grads)
        #temp = x

        # add emb into history
        if self.history is not None:
            self.history.add(x)

        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        if not encoder_padding_mask.any():
            encoder_padding_mask = None

        # encoder layers
        #inner_states=[x]
        for layer_id, layer in enumerate(self.layers):
            if layer_id in self.k:
                if self.history is not None:
                    x = self.history.pop()
                x,qkv,ffn_out, attn_weight = layer(x, encoder_padding_mask)
                #inner_states.append(x)
                #self.attn_weight.append(attn_weight)
                #util.inspect_grad("encoder_%d" % (layer_id+1), x, self.inspected_grads)
                #util.inspect_grad("encoder_%d" % (layer_id + 1), qkv, self.inspected_grads_qkv)
                #util.inspect_grad("encoder_%d" % (layer_id + 1), ffn_out, self.inspected_grads_ffn_out)
            else:
                x,qkv,ffn_out, attn_weight = layer(x, encoder_padding_mask)
                #inner_states.append(x)
                #self.attn_weight.append(attn_weight)
                #util.inspect_grad("encoder_%d" % (layer_id+1), x, self.inspected_grads)
                #util.inspect_grad("encoder_%d" % (layer_id + 1), qkv, self.inspected_grads_qkv)
                #util.inspect_grad("encoder_%d" % (layer_id + 1), ffn_out, self.inspected_grads_ffn_out)
                if layer_id + 1 in self.k:
                    if self.history is not None:
                        self.history.add(x)
        #self.count = 0
        #self.history.print_weight()
        if self.history is not None:
            x = self.history.pop()

        if self.normalize:
            x = self.layer_norm(x)

        #util.inspect_grad("encoder_top", x, self.inspected_grads)
        #输出分布
        #self.print_attn_weight()
        #for key,value in dict_intra_sim.items():
            #print('{}'.format(value))

        return {
            'encoder_out': x,  # T x B x C
            'encoder_padding_mask': encoder_padding_mask,  # B x T
        }

    def layer_sim(self,inner_states):
        length, batch, hidden = inner_states[0].size()
        if not self.training and batch ==1:
            word_rep_list = inner_states[0].view(length, hidden)
            word_sim = {}
            cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
            for id, i in enumerate(inner_states[1:]):
                layer_rep_list = i.view(length, hidden)
                word_average = 0
                for i in range(3, 4):
                    cosin = cos(word_rep_list[i], layer_rep_list[i])
                    word_average += cosin
                # word_average = word_average / (length)
                word_sim[id+1] = word_average
        for key, value in word_sim.items():
            # print(value)
            print('{}'.format(value))
    def adj_sim(self,inner_states):
        length, batch, hidden = inner_states[0].size()
        if not self.training and batch ==1:
            word_rep_list = inner_states[0].view(length, hidden)
            word_sim = {}
            cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
            for id, i in enumerate(inner_states[1:]):
                layer_rep_list = i.view(length, hidden)
                word_average = 0
                for i in range(3, 4):
                    cosin = cos(word_rep_list[i], layer_rep_list[i])
                    word_average += cosin
                # word_average = word_average / (length)
                word_sim[id+1] = word_average
                word_rep_list = layer_rep_list
        for key, value in word_sim.items():
            # print(value)
            print('{}'.format(value))

    def print_attn_weight(self):
        for layer_id, tensor in enumerate(self.attn_weight):
            with open('attn_weight_stack', 'a') as f:
                f.write(str(layer_id))
                f.write('\n')
                f.write(str(tensor.cpu().numpy()))
                f.write('\n')

    def reorder_encoder_out(self, encoder_out, new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        if encoder_out['encoder_out'] is not None:
            encoder_out['encoder_out'] = \
                encoder_out['encoder_out'].index_select(1, new_order)
        if encoder_out['encoder_padding_mask'] is not None:
            encoder_out['encoder_padding_mask'] = \
                encoder_out['encoder_padding_mask'].index_select(0, new_order)
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions())

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = '{}.embed_positions.weights'.format(name)
            if weights_key in state_dict:
                del state_dict[weights_key]
            state_dict['{}.embed_positions._float_tensor'.format(name)] = torch.FloatTensor(1)
        version_key = '{}.version'.format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) < 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])
        return state_dict

class TransformerDecoder(FairseqIncrementalDecoder):
    """
    Transformer decoder consisting of *args.decoder_layers* layers. Each layer
    is a :class:`TransformerDecoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
        left_pad (bool, optional): whether the input is left-padded
            (default: False).
        final_norm (bool, optional): apply layer norm to the output of the
            final decoder layer (default: True).
    """

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False, left_pad=False, final_norm=True):
        super().__init__(dictionary)
        self.dropout = args.dropout
        self.share_input_output_embed = args.share_decoder_input_output_embed

        input_embed_dim = embed_tokens.embedding_dim
        embed_dim = args.decoder_embed_dim
        output_embed_dim = args.decoder_output_dim

        padding_idx = embed_tokens.padding_idx
        self.max_target_positions = args.max_target_positions

        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(embed_dim)  # todo: try with input_embed_dim

        self.project_in_dim = Linear(input_embed_dim, embed_dim, bias=False) if embed_dim != input_embed_dim else None

        self.embed_positions = PositionalEmbedding(
            args.max_target_positions, embed_dim, padding_idx,
            left_pad=left_pad,
            learned=args.decoder_learned_pos,
        ) if not args.no_token_positional_embeddings else None

        # create decoder layer history
        #self.history = CreateLayerHistory(args, is_encoder=False)
        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerDecoderLayer(args, no_encoder_attn)
            for _ in range(args.decoder_layers)
        ])

        self.adaptive_softmax = None

        self.project_out_dim = Linear(embed_dim, output_embed_dim, bias=False) \
            if embed_dim != output_embed_dim and not args.tie_adaptive_weights else None

        if args.adaptive_softmax_cutoff is not None:
            self.adaptive_softmax = AdaptiveSoftmax(
                len(dictionary),
                output_embed_dim,
                options.eval_str_list(args.adaptive_softmax_cutoff, type=int),
                dropout=args.adaptive_softmax_dropout,
                adaptive_inputs=embed_tokens if args.tie_adaptive_weights else None,
                factor=args.adaptive_softmax_factor,
                tie_proj=args.tie_adaptive_proj,
            )
        elif not self.share_input_output_embed:
            self.embed_out = nn.Parameter(torch.Tensor(len(dictionary), output_embed_dim))
            nn.init.normal_(self.embed_out, mean=0, std=output_embed_dim ** -0.5)
        self.register_buffer('version', torch.Tensor([2]))
        self.normalize = args.decoder_normalize_before and final_norm
        if self.normalize:
            self.layer_norm = LayerNorm(embed_dim)

    def forward(self, prev_output_tokens, encoder_out=None, incremental_state=None):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for input feeding/teacher forcing
            encoder_out (Tensor, optional): output from the encoder, used for
                encoder-side attention
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`

        Returns:
            tuple:
                - the last decoder layer's output of shape `(batch, tgt_len,
                  vocab)`
                - the last decoder layer's attention weights of shape `(batch,
                  tgt_len, src_len)`
        """
        # embed positions
        #if self.history is not None:
            #self.history.clean()
        # embed positions
        positions = self.embed_positions(
            prev_output_tokens,
            incremental_state=incremental_state,
        ) if self.embed_positions is not None else None

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        attn = None

        inner_states = [x]

        # add emb into history
        #if self.history is not None:
            #self.history.add(x)

        # decoder layers
        for layer in self.layers:
            #if self.history is not None:
                #x = self.history.pop()
            x, attn = layer(
                x,
                encoder_out['encoder_out'] if encoder_out is not None else None,
                encoder_out['encoder_padding_mask'] if encoder_out is not None else None,
                incremental_state,
                self_attn_mask=self.buffered_future_mask(x) if incremental_state is None else None,
            )
            inner_states.append(x)
            #if self.history is not None:
                #self.history.add(x)

        #if self.history is not None:
            #x = self.history.pop()

        if self.normalize:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        if self.adaptive_softmax is None:
            # project back to size of vocabulary
            if self.share_input_output_embed:
                x = F.linear(x, self.embed_tokens.weight)
            else:
                x = F.linear(x, self.embed_out)

        return x, {'attn': attn, 'inner_states': inner_states}

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        if self.embed_positions is None:
            return self.max_target_positions
        return min(self.max_target_positions, self.embed_positions.max_positions())

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        if not hasattr(self, '_future_mask') or self._future_mask is None or self._future_mask.device != tensor.device:
            self._future_mask = torch.triu(utils.fill_with_neg_inf(tensor.new(dim, dim)), 1)
        if self._future_mask.size(0) < dim:
            self._future_mask = torch.triu(utils.fill_with_neg_inf(self._future_mask.resize_(dim, dim)), 1)
        return self._future_mask[:dim, :dim]

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = '{}.embed_positions.weights'.format(name)
            if weights_key in state_dict:
                del state_dict[weights_key]
            state_dict['{}.embed_positions._float_tensor'.format(name)] = torch.FloatTensor(1)

        for i in range(len(self.layers)):
            # update layer norms
            layer_norm_map = {
                '0': 'self_attn_layer_norm',
                '1': 'encoder_attn_layer_norm',
                '2': 'final_layer_norm'
            }
            for old, new in layer_norm_map.items():
                for m in ('weight', 'bias'):
                    k = '{}.layers.{}.layer_norms.{}.{}'.format(name, i, old, m)
                    if k in state_dict:
                        state_dict['{}.layers.{}.{}.{}'.format(name, i, new, m)] = state_dict[k]
                        del state_dict[k]
        if utils.item(state_dict.get('{}.version'.format(name), torch.Tensor([1]))[0]) < 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict['{}.version'.format(name)] = torch.Tensor([1])

        return state_dict


class TransformerEncoderLayer(nn.Module):
    """Encoder layer block.

    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: `dropout -> add residual -> layernorm`. In the
    tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.encoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
    """

    def __init__(self, args):
        super().__init__()
        self.embed_dim = args.encoder_embed_dim
        if args.max_relative_length==-1:
            self.self_attn = MultiheadAttention(
                self.embed_dim, args.encoder_attention_heads,
                dropout=args.attention_dropout,
            )
        else:
            self.self_attn = RelativeMultiheadAttention(
                self.embed_dim, args.encoder_attention_heads,
                args.max_relative_length, dropout=args.attention_dropout, k_only=args.k_only,
            )

        self.dropout = args.dropout
        self.relu_dropout = args.relu_dropout
        self.normalize_before = args.encoder_normalize_before
        self.fc1 = Linear(self.embed_dim, args.encoder_ffn_embed_dim)
        self.fc2 = Linear(args.encoder_ffn_embed_dim, self.embed_dim)
        self.layer_norms = nn.ModuleList([LayerNorm(self.embed_dim) for i in range(2)])

    def forward(self, x, encoder_padding_mask):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.

        Returns:
            encoded output of shape `(batch, src_len, embed_dim)`
        """
        residual = x
        x = self.maybe_layer_norm(0, x, before=True)
        x, attn_weight = self.self_attn(query=x, key=x, value=x, key_padding_mask=encoder_padding_mask)
        x = F.dropout(x, p=self.dropout, training=self.training)
        qkv = x
        x = residual + x
        x = self.maybe_layer_norm(0, x, after=True)

        residual = x
        x = self.maybe_layer_norm(1, x, before=True)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        ffn_out = x
        x = residual + x
        x = self.maybe_layer_norm(1, x, after=True)
        return x,qkv,ffn_out,attn_weight

    def maybe_layer_norm(self, i, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return self.layer_norms[i](x)
        else:
            return x


class TransformerDecoderLayer(nn.Module):
    """Decoder layer block.

    In the original paper each operation (multi-head attention, encoder
    attention or FFN) is postprocessed with: `dropout -> add residual ->
    layernorm`. In the tensor2tensor code they suggest that learning is more
    robust when preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.decoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(self, args, no_encoder_attn=False):
        super().__init__()
        self.embed_dim = args.decoder_embed_dim
        if args.max_relative_length == -1:
            self.self_attn = MultiheadAttention(
                self.embed_dim, args.decoder_attention_heads,
                dropout=args.attention_dropout,
            )
        else:
            self.self_attn = RelativeMultiheadAttention(
                self.embed_dim, args.decoder_attention_heads,
                args.max_relative_length, dropout=args.attention_dropout, k_only=args.k_only,
            )

        self.dropout = args.dropout
        self.relu_dropout = args.relu_dropout
        self.normalize_before = args.decoder_normalize_before

        self.self_attn_layer_norm = LayerNorm(self.embed_dim)

        if no_encoder_attn:
            self.encoder_attn = None
            self.encoder_attn_layer_norm = None
        else:
            self.encoder_attn = MultiheadAttention(
                self.embed_dim, args.decoder_attention_heads,
                dropout=args.attention_dropout,
            )
            self.encoder_attn_layer_norm = LayerNorm(self.embed_dim)

        self.fc1 = Linear(self.embed_dim, args.decoder_ffn_embed_dim)
        self.fc2 = Linear(args.decoder_ffn_embed_dim, self.embed_dim)

        self.final_layer_norm = LayerNorm(self.embed_dim)
        self.need_attn = True

        self.onnx_trace = False

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def forward(self, x, encoder_out, encoder_padding_mask, incremental_state,
                prev_self_attn_state=None, prev_attn_state=None, self_attn_mask=None,
                self_attn_padding_mask=None):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.

        Returns:
            encoded output of shape `(batch, src_len, embed_dim)`
        """
        residual = x
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, before=True)
        if prev_self_attn_state is not None:
            if incremental_state is None:
                incremental_state = {}
            prev_key, prev_value = prev_self_attn_state
            saved_state = {"prev_key": prev_key, "prev_value": prev_value}
            self.self_attn._set_input_buffer(incremental_state, saved_state)
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, after=True)

        attn = None
        if self.encoder_attn is not None:
            residual = x
            x = self.maybe_layer_norm(self.encoder_attn_layer_norm, x, before=True)
            if prev_attn_state is not None:
                if incremental_state is None:
                    incremental_state = {}
                prev_key, prev_value = prev_attn_state
                saved_state = {"prev_key": prev_key, "prev_value": prev_value}
                self.encoder_attn._set_input_buffer(incremental_state, saved_state)
            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=(not self.training and self.need_attn),
            )
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = residual + x
            x = self.maybe_layer_norm(self.encoder_attn_layer_norm, x, after=True)

        residual = x
        x = self.maybe_layer_norm(self.final_layer_norm, x, before=True)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(self.final_layer_norm, x, after=True)
        if self.onnx_trace:
            saved_state = self.self_attn._get_input_buffer(incremental_state)
            self_attn_state = saved_state["prev_key"], saved_state["prev_value"]
            return x, attn, self_attn_state
        return x, attn

    def maybe_layer_norm(self, layer_norm, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return layer_norm(x)
        else:
            return x

    def make_generation_fast_(self, need_attn=False, **kwargs):
        self.need_attn = need_attn


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m


def PositionalEmbedding(num_embeddings, embedding_dim, padding_idx, left_pad, learned=False):
    if learned:
        m = LearnedPositionalEmbedding(num_embeddings + padding_idx + 1, embedding_dim, padding_idx, left_pad)
        nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
        nn.init.constant_(m.weight[padding_idx], 0)
    else:
        m = SinusoidalPositionalEmbedding(embedding_dim, padding_idx, left_pad, num_embeddings + padding_idx + 1)
    return m



@register_model_architecture('sdt_transformer', 'sdt_transformer')
def base_architecture(args):
    args.encoder_embed_path = getattr(args, 'encoder_embed_path', None)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 2048)
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 8)
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', False)
    args.encoder_learned_pos = getattr(args, 'encoder_learned_pos', False)
    args.decoder_embed_path = getattr(args, 'decoder_embed_path', None)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', args.encoder_ffn_embed_dim)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 8)
    args.decoder_normalize_before = getattr(args, 'decoder_normalize_before', False)
    args.decoder_learned_pos = getattr(args, 'decoder_learned_pos', False)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.)
    args.relu_dropout = getattr(args, 'relu_dropout', 0.)
    args.dropout = getattr(args, 'dropout', 0.1)
    args.adaptive_softmax_cutoff = getattr(args, 'adaptive_softmax_cutoff', None)
    args.adaptive_softmax_dropout = getattr(args, 'adaptive_softmax_dropout', 0)
    args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', False)
    args.share_all_embeddings = getattr(args, 'share_all_embeddings', False)
    args.no_token_positional_embeddings = getattr(args, 'no_token_positional_embeddings', False)
    args.adaptive_input = getattr(args, 'adaptive_input', False)

    args.decoder_output_dim = getattr(args, 'decoder_output_dim', args.decoder_embed_dim)
    args.decoder_input_dim = getattr(args, 'decoder_input_dim', args.decoder_embed_dim)
    args.max_relative_length = getattr(args, 'max_relative_length', args.max_relative_length)
    args.k_only = getattr(args, 'k_only', args.k_only)
    args.inspect_grad = getattr(args, 'inspect_grad', False)

@register_model_architecture('sdt_transformer', 'sdt_transformer_wmt_en_de')
def transformer_wmt_en_de(args):
    base_architecture(args)


@register_model_architecture('sdt_transformer', 'sdt_transformer_t2t_wmt_en_de')
def transformer_t2t_wmt_en_de(args):
    args.encoder_normalize_before = True
    args.decoder_normalize_before = True
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.relu_dropout = getattr(args, 'relu_dropout', 0.1)
    base_architecture(args)


@register_model_architecture('sdt_transformer', 'sdt_relative_transformer_wmt_en_de')
def relative_transformer_wmt_en_de(args):
    args.max_relative_length = 8
    args.k_only = True
    base_architecture(args)


@register_model_architecture('sdt_transformer', 'sdt_relative_transformer_t2t_wmt_en_de')
def relative_transformer_t2t_wmt_en_de(args):
    args.encoder_normalize_before = True
    args.decoder_normalize_before = True
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.relu_dropout = getattr(args, 'relu_dropout', 0.1)
    args.encoder_layers = 40
    args.max_relative_length = 8
    args.k_only = True
    base_architecture(args)


# parameters used in the "Attention Is All You Need" paper (Vaswani, et al, 2017)
@register_model_architecture('sdt_transformer', 'sdt_transformer_vaswani_wmt_en_de_big')
def transformer_vaswani_wmt_en_de_big(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 1024)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 4096)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 16)
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', False)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 1024)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 4096)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 16)
    args.dropout = getattr(args, 'dropout', 0.3)
    base_architecture(args)


@register_model_architecture('fusion_transformer', 'sdt_transformer_vaswani_wmt_en_fr_big')
def transformer_vaswani_wmt_en_fr_big(args):
    args.dropout = getattr(args, 'dropout', 0.1)
    transformer_vaswani_wmt_en_de_big(args)


@register_model_architecture('sdt_transformer', 'sdt_transformer_wmt_en_de_big')
def transformer_wmt_en_de_big(args):
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    transformer_vaswani_wmt_en_de_big(args)


# default parameters used in tensor2tensor implementation
@register_model_architecture('sdt_transformer', 'sdt_transformer_wmt_en_de_big_t2t')
def transformer_wmt_en_de_big_t2t(args):
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', True)
    args.decoder_normalize_before = getattr(args, 'decoder_normalize_before', True)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.relu_dropout = getattr(args, 'relu_dropout', 0.1)
    transformer_vaswani_wmt_en_de_big(args)



@register_model_architecture('sdt_transformer', 'sdt_transformer_t2t_wmt_en_de_6l')
def transformer_t2t_wmt_en_de_6l(args):
    args.encoder_normalize_before = True
    args.decoder_normalize_before = True
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.relu_dropout = getattr(args, 'relu_dropout', 0.1)
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.encoder_history_type = getattr(args, 'encoder_history_type', 'learnable_dense')
    args.decoder_history_type = getattr(args, 'decoder_history_type', 'learnable_dense')
    args.k=[0,6]
    args.max_relative_length = 8
    args.k_only = True
    args.inspect_grad=True
    base_architecture(args)

@register_model_architecture('sdt_transformer', 'sdt_transformer_t2t_wmt_en_de_8l')
def transformer_t2t_wmt_en_de_8l(args):
    args.encoder_normalize_before = True
    args.decoder_normalize_before = True
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.relu_dropout = getattr(args, 'relu_dropout', 0.1)
    args.encoder_layers = getattr(args, 'encoder_layers', 8)
    args.encoder_history_type = getattr(args, 'encoder_history_type', 'learnable_dense')
    args.decoder_history_type = getattr(args, 'decoder_history_type', 'learnable_dense')
    args.k=[0,8]
    args.max_relative_length = 8
    args.k_only = True
    args.inspect_grad=True
    base_architecture(args)

@register_model_architecture('sdt_transformer', 'sdt_transformer_t2t_wmt_en_de_9l')
def transformer_t2t_wmt_en_de_9l(args):
    args.encoder_normalize_before = True
    args.decoder_normalize_before = True
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.relu_dropout = getattr(args, 'relu_dropout', 0.1)
    args.encoder_layers = getattr(args, 'encoder_layers', 9)
    args.encoder_history_type = getattr(args, 'encoder_history_type', 'learnable_dense')
    args.decoder_history_type = getattr(args, 'decoder_history_type', 'learnable_dense')
    args.k=[0,9]
    args.max_relative_length = 8
    args.k_only = True
    args.inspect_grad=True
    base_architecture(args)

@register_model_architecture('sdt_transformer', 'sdt_transformer_t2t_wmt_en_de_12l')
def transformer_t2t_wmt_en_de_12l(args):
    args.encoder_normalize_before = True
    args.decoder_normalize_before = True
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.relu_dropout = getattr(args, 'relu_dropout', 0.1)
    args.encoder_layers = getattr(args, 'encoder_layers', 12)
    args.encoder_history_type = getattr(args, 'encoder_history_type', 'learnable_dense')
    args.decoder_history_type = getattr(args, 'decoder_history_type', 'learnable_dense')
    args.k = [0,6,12]
    args.max_relative_length = 8
    args.k_only = True
    args.inspect_grad=True
    base_architecture(args)

@register_model_architecture('sdt_transformer', 'sdt_transformer_t2t_wmt_en_de_15l')
def transformer_t2t_wmt_en_de_15l(args):
    args.encoder_normalize_before = True
    args.decoder_normalize_before = True
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.relu_dropout = getattr(args, 'relu_dropout', 0.1)
    args.encoder_layers = getattr(args, 'encoder_layers', 15)
    args.encoder_history_type = getattr(args, 'encoder_history_type', 'learnable_dense')
    args.decoder_history_type = getattr(args, 'decoder_history_type', 'learnable_dense')
    args.k = [0,6,9,12,15]
    args.max_relative_length = 8
    args.k_only = True
    args.inspect_grad=True
    base_architecture(args)

@register_model_architecture('sdt_transformer', 'sdt_transformer_t2t_wmt_en_de_16l')
def transformer_t2t_wmt_en_de_16l(args):
    args.encoder_normalize_before = True
    args.decoder_normalize_before = True
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.relu_dropout = getattr(args, 'relu_dropout', 0.1)
    args.encoder_layers = getattr(args, 'encoder_layers', 16)
    args.encoder_history_type = getattr(args, 'encoder_history_type', 'learnable_dense')
    args.decoder_history_type = getattr(args, 'decoder_history_type', 'learnable_dense')
    args.k=[0,8,16]
    args.max_relative_length = 8
    args.k_only = True
    args.inspect_grad=True
    base_architecture(args)

@register_model_architecture('sdt_transformer', 'sdt_transformer_t2t_wmt_en_de_18l')
def transformer_t2t_wmt_en_de_18l(args):
    args.encoder_normalize_before = True
    args.decoder_normalize_before = True
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.relu_dropout = getattr(args, 'relu_dropout', 0.1)
    args.encoder_layers = getattr(args, 'encoder_layers', 18)
    args.encoder_history_type = getattr(args, 'encoder_history_type', 'learnable_dense')
    args.decoder_history_type = getattr(args, 'decoder_history_type', 'learnable_dense')
    args.k = [0,6,12,18]
    args.max_relative_length = 8
    args.k_only = True
    args.inspect_grad=True
    base_architecture(args)

@register_model_architecture('sdt_transformer', 'sdt_transformer_t2t_wmt_en_de_21l')
def transformer_t2t_wmt_en_de_21l(args):
    args.encoder_normalize_before = True
    args.decoder_normalize_before = True
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.relu_dropout = getattr(args, 'relu_dropout', 0.1)
    args.encoder_layers = getattr(args, 'encoder_layers', 21)
    args.encoder_history_type = getattr(args, 'encoder_history_type', 'learnable_dense')
    args.decoder_history_type = getattr(args, 'decoder_history_type', 'learnable_dense')
    args.k = [0,6,12,21]
    args.max_relative_length = 8
    args.k_only = True
    args.inspect_grad=True
    base_architecture(args)

@register_model_architecture('sdt_transformer', 'sdt_transformer_t2t_wmt_en_de_24l')
def transformer_t2t_wmt_en_de_24l(args):
    args.encoder_normalize_before = True
    args.decoder_normalize_before = True
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.relu_dropout = getattr(args, 'relu_dropout', 0.1)
    args.encoder_layers = getattr(args, 'encoder_layers', 24)
    args.encoder_history_type = getattr(args, 'encoder_history_type', 'learnable_dense')
    args.decoder_history_type = getattr(args, 'decoder_history_type', 'learnable_dense')
    args.k = [0,6,12,18,24]
    args.max_relative_length = 8
    args.k_only = True
    args.inspect_grad=True
    base_architecture(args)

@register_model_architecture('sdt_transformer', 'sdt_transformer_t2t_wmt_en_de_27l')
def transformer_t2t_wmt_en_de_27l(args):
    args.encoder_normalize_before = True
    args.decoder_normalize_before = True
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.relu_dropout = getattr(args, 'relu_dropout', 0.1)
    args.encoder_layers = getattr(args, 'encoder_layers', 27)
    args.encoder_history_type = getattr(args, 'encoder_history_type', 'learnable_dense')
    args.decoder_history_type = getattr(args, 'decoder_history_type', 'learnable_dense')
    args.k = [0,9,18,27]
    args.max_relative_length = 8
    args.k_only = True
    args.inspect_grad=True
    base_architecture(args)

@register_model_architecture('sdt_transformer', 'sdt_transformer_t2t_wmt_en_de_30l')
def transformer_t2t_wmt_en_de_30l(args):
    args.encoder_normalize_before = True
    args.decoder_normalize_before = True
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.relu_dropout = getattr(args, 'relu_dropout', 0.1)
    args.encoder_layers = getattr(args, 'encoder_layers', 30)
    args.encoder_history_type = getattr(args, 'encoder_history_type', 'learnable_dense')
    args.decoder_history_type = getattr(args, 'decoder_history_type', 'learnable_dense')
    args.k = [0,6,12,18,24,30]
    args.max_relative_length = 8
    args.k_only = True
    args.inspect_grad=True
    base_architecture(args)

@register_model_architecture('sdt_transformer', 'sdt_transformer_t2t_wmt_en_de_33l')
def transformer_t2t_wmt_en_de_33l(args):
    args.encoder_normalize_before = True
    args.decoder_normalize_before = True
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.relu_dropout = getattr(args, 'relu_dropout', 0.1)
    args.encoder_layers = getattr(args, 'encoder_layers', 33)
    args.encoder_history_type = getattr(args, 'encoder_history_type', 'learnable_dense')
    args.decoder_history_type = getattr(args, 'decoder_history_type', 'learnable_dense')
    args.k = [0,6,12,21,33]
    args.max_relative_length = 8
    args.k_only = True
    args.inspect_grad=True
    base_architecture(args)

@register_model_architecture('sdt_transformer', 'sdt_transformer_t2t_wmt_en_de_36l')
def transformer_t2t_wmt_en_de_36l(args):
    args.encoder_normalize_before = True
    args.decoder_normalize_before = True
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.relu_dropout = getattr(args, 'relu_dropout', 0.1)
    args.encoder_layers = getattr(args, 'encoder_layers', 36)
    args.encoder_history_type = getattr(args, 'encoder_history_type', 'learnable_dense')
    args.decoder_history_type = getattr(args, 'decoder_history_type', 'learnable_dense')
    args.k = [0,6,12,18,24,30,36]
    args.max_relative_length = 8
    args.k_only = True
    args.inspect_grad=True
    base_architecture(args)


@register_model_architecture('sdt_transformer', 'sdt_transformer_t2t_wmt_en_de_39l')
def transformer_t2t_wmt_en_de_39l(args):
    args.encoder_normalize_before = True
    args.decoder_normalize_before = True
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.relu_dropout = getattr(args, 'relu_dropout', 0.1)
    args.encoder_layers = getattr(args, 'encoder_layers', 39)
    args.encoder_history_type = getattr(args, 'encoder_history_type', 'learnable_dense')
    args.decoder_history_type = getattr(args, 'decoder_history_type', 'learnable_dense')
    args.k = [0,6,9,12,15,18,21,24,27,30,33,36,39]
    args.max_relative_length = 8
    args.k_only = True
    args.inspect_grad=True
    base_architecture(args)

@register_model_architecture('sdt_transformer', 'sdt_transformer_t2t_wmt_en_de_40l')
def transformer_t2t_wmt_en_de_40l(args):
    args.encoder_normalize_before = True
    args.decoder_normalize_before = True
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.relu_dropout = getattr(args, 'relu_dropout', 0.1)
    args.encoder_layers = getattr(args, 'encoder_layers', 40)
    base_architecture(args)

@register_model_architecture('sdt_transformer', 'sdt_transformer_t2t_wmt_en_de_42l')
def transformer_t2t_wmt_en_de_42l(args):
    args.encoder_normalize_before = True
    args.decoder_normalize_before = True
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.relu_dropout = getattr(args, 'relu_dropout', 0.1)
    args.encoder_layers = getattr(args, 'encoder_layers', 42)
    args.encoder_history_type = getattr(args, 'encoder_history_type', 'learnable_dense')
    args.decoder_history_type = getattr(args, 'decoder_history_type', 'learnable_dense')
    args.k = [0,6,12,18,24,30,36,42]
    args.max_relative_length = 8
    args.k_only = True
    args.inspect_grad=True
    base_architecture(args)

@register_model_architecture('sdt_transformer', 'sdt_transformer_t2t_wmt_en_de_45l')
def transformer_t2t_wmt_en_de_45l(args):
    args.encoder_normalize_before = True
    args.decoder_normalize_before = True
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.relu_dropout = getattr(args, 'relu_dropout', 0.1)
    args.encoder_layers = getattr(args, 'encoder_layers', 45)
    args.encoder_history_type = getattr(args, 'encoder_history_type', 'learnable_dense')
    args.decoder_history_type = getattr(args, 'decoder_history_type', 'learnable_dense')
    args.k = [0,6,9,12,15,18,21,24,27,30,33,36,39,42,45]
    args.max_relative_length = 8
    args.k_only = True
    args.inspect_grad=True
    base_architecture(args)

@register_model_architecture('sdt_transformer', 'sdt_transformer_t2t_wmt_en_de_48l')
def transformer_t2t_wmt_en_de_48l(args):
    args.encoder_normalize_before = True
    args.decoder_normalize_before = True
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.relu_dropout = getattr(args, 'relu_dropout', 0.1)
    args.encoder_layers = getattr(args, 'encoder_layers', 48)
    args.encoder_history_type = getattr(args, 'encoder_history_type', 'learnable_dense')
    args.decoder_history_type = getattr(args, 'decoder_history_type', 'learnable_dense')
    args.k = [0,6,12,18,24,30,36,42,48]
    args.max_relative_length = 8
    args.k_only = True
    args.inspect_grad=True
    base_architecture(args)

@register_model_architecture('sdt_transformer', 'sdt_transformer_t2t_wmt_en_de_54l')
def transformer_t2t_wmt_en_de_54l(args):
    args.encoder_normalize_before = True
    args.decoder_normalize_before = True
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.relu_dropout = getattr(args, 'relu_dropout', 0.1)
    args.encoder_layers = getattr(args, 'encoder_layers', 54)
    args.encoder_history_type = getattr(args, 'encoder_history_type', 'learnable_dense')
    args.decoder_history_type = getattr(args, 'decoder_history_type', 'learnable_dense')
    args.k=[0,6,12,18,24,30,36,42,48,54]
    args.max_relative_length = 8
    args.k_only = True
    args.inspect_grad=True
    base_architecture(args)

@register_model_architecture('sdt_transformer', 'sdt_transformer_t2t_wmt_en_de_60l')
def transformer_t2t_wmt_en_de_60l(args):
    args.encoder_normalize_before = True
    args.decoder_normalize_before = True
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.relu_dropout = getattr(args, 'relu_dropout', 0.1)
    args.encoder_layers = getattr(args, 'encoder_layers', 60)
    args.encoder_history_type = getattr(args, 'encoder_history_type', 'learnable_dense')
    args.decoder_history_type = getattr(args, 'decoder_history_type', 'learnable_dense')
    args.k=[0,6,12,18,24,30,36,42,48,54,60]
    args.max_relative_length = 8
    args.k_only = True
    args.inspect_grad=True
    base_architecture(args)

@register_model_architecture('sdt_transformer', 'sdt_transformer_t2t_wmt_en_de_63l')
def transformer_t2t_wmt_en_de_63l(args):
    args.encoder_normalize_before = True
    args.decoder_normalize_before = True
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.relu_dropout = getattr(args, 'relu_dropout', 0.1)
    args.encoder_layers = getattr(args, 'encoder_layers', 63)
    args.encoder_history_type = getattr(args, 'encoder_history_type', 'learnable_dense')
    args.decoder_history_type = getattr(args, 'decoder_history_type', 'learnable_dense')
    args.k=[0,9,18,27,36,45,54,63]
    #args.max_relative_length = 8
    #args.k_only = True
    base_architecture(args)

@register_model_architecture('sdt_transformer', 'sdt_transformer_t2t_wmt_en_de_66l')
def transformer_t2t_wmt_en_de_66l(args):
    args.encoder_normalize_before = True
    args.decoder_normalize_before = True
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.relu_dropout = getattr(args, 'relu_dropout', 0.1)
    args.encoder_layers = getattr(args, 'encoder_layers', 66)
    args.encoder_history_type = getattr(args, 'encoder_history_type', 'learnable_dense')
    args.decoder_history_type = getattr(args, 'decoder_history_type', 'learnable_dense')
    args.k=[0,6,12,18,24,30,36,42,48,54,60,66]
    args.max_relative_length = 8
    args.k_only = True
    base_architecture(args)

@register_model_architecture('sdt_transformer', 'sdt_transformer_t2t_wmt_en_de_72l')
def transformer_t2t_wmt_en_de_72l(args):
    args.encoder_normalize_before = True
    args.decoder_normalize_before = True
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.relu_dropout = getattr(args, 'relu_dropout', 0.1)
    args.encoder_layers = getattr(args, 'encoder_layers', 72)
    args.encoder_history_type = getattr(args, 'encoder_history_type', 'learnable_dense')
    args.decoder_history_type = getattr(args, 'decoder_history_type', 'learnable_dense')
    args.k=[0,6,12,18,24,30,36,42,48,54,60,66,72]
    args.max_relative_length = 8
    args.k_only = True
    args.inspect_grad=True
    base_architecture(args)

@register_model_architecture('sdt_transformer', 'sdt_transformer_t2t_wmt_en_de_96l')
def transformer_t2t_wmt_en_de_96l(args):
    args.encoder_normalize_before = True
    args.decoder_normalize_before = True
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.relu_dropout = getattr(args, 'relu_dropout', 0.1)
    args.encoder_layers = getattr(args, 'encoder_layers', 96)
    args.encoder_history_type = getattr(args, 'encoder_history_type', 'learnable_dense')
    args.decoder_history_type = getattr(args, 'decoder_history_type', 'learnable_dense')
    args.k = [0,6,12,18,24,30,36,42,48,96]
    base_architecture(args)