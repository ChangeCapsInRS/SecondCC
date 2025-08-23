import torch,os
from torch import nn
import math
from torch.nn.init import xavier_uniform_
import numpy as np
from torch import Tensor
from typing import Optional
from torch.nn import functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 1
torch.manual_seed(seed)

class CrossTransformer(nn.Module):
    def __init__(self, dropout, d_model=512, n_head=4):
        """
        :param dropout: dropout rate
        :param d_model: dimension of hidden state
        :param n_head: number of heads in multi head attention
        """
        super(CrossTransformer, self).__init__()
        self.attention = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
        self.attention2 = nn.MultiheadAttention(d_model, n_head, dropout=dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)
        self.dropout5 = nn.Dropout(dropout)
        self.dropout6 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.linear2 = nn.Linear(d_model * 4, d_model)
        self.linear3 = nn.Linear(d_model, d_model * 4)
        self.linear4 = nn.Linear(d_model * 4, d_model)

    def forward(self, In1, In2, Sn1, Sn2):
        """
        Forward pass of the CrossTransformer block.

        :param In1: input feature set 1 (e.g., image A features), shape (seq_len, batch, d_model)
        :param In2: input feature set 2 (e.g., image B features), shape (seq_len, batch, d_model)
        :param Sn1: secondary feature set 1 (e.g., semantic A features), shape (seq_len, batch, d_model)
        :param Sn2: secondary feature set 2 (e.g., semantic B features), shape (seq_len, batch, d_model)

        The idea:
          1. Perform cross-attention between input features and semantic features.
          2. Perform cross-attention between semantic features and input features.
          3. Introduce difference features (In2 - In1, Sn2 - Sn1) and let each stream attend to them.
        This encourages the model to explicitly capture "change" signals.
        """

        # Cross-attention: image features attend to semantic features
        I1, attw1 = self.cross(In1, Sn1, Sn1)  # Query = In1, Keys = Sn1, Values = Sn1
        I2, attw2 = self.cross(In2, Sn2, Sn2)  # Query = In2, Keys = Sn2, Values = Sn2

        # Cross-attention: semantic features attend to image features
        S1, attw3 = self.cross(Sn1, In1, In1)  # Query = Sn1, Keys = In1, Values = In1
        S2, attw4 = self.cross(Sn2, In2, In2)  # Query = Sn2, Keys = In2, Values = In2

        # Difference features between images (explicit change signal)
        diff = In2 - In1
        I1, attw1 = self.cross(I1, diff, diff)  # Image A attends to difference
        I2, attw2 = self.cross(I2, diff, diff)  # Image B attends to difference

        # Difference features between semantics
        diff = Sn2 - Sn1
        S1, attw3 = self.cross(S1, diff, diff)  # Semantic A attends to difference
        S2, attw4 = self.cross(S2, diff, diff)  # Semantic B attends to difference

        # Return updated features for both image and semantic streams
        return I1, I2, S1, S2

    def cross(self, input, difK, diffV):
        attn_output, attn_weight = self.attention2(input, difK,
                                                   diffV)  # Cross-attention: Query=input, Key=difK, Value=diffV

        # Residual connection + dropout + layer normalization (post-attention)
        output = input + self.dropout4(attn_output)
        output = self.norm3(output)

        # Feed-forward network (FFN): Linear → ReLU → Dropout → Linear
        ff_output = self.linear4(
            self.dropout5(
                self.activation(
                    self.linear3(output)
                )
            )
        )

        # Residual connection + dropout + layer normalization (post-FFN)
        output = output + self.dropout6(ff_output)
        output = self.norm4(output)

        # Return updated output and attention weights
        return output, attn_weight


class resblock(nn.Module):
    '''
    module: Residual Block
    '''
    def __init__(self, inchannel, outchannel, stride=1, shortcut=None):
        super(resblock, self).__init__()
        self.left = nn.Sequential(
                nn.Conv2d(inchannel,int(outchannel/2),kernel_size = 1),
                nn.BatchNorm2d(int(outchannel/2)),
                nn.ReLU(inplace=True),
                nn.Conv2d(int(outchannel/2), int(outchannel / 2), kernel_size = 3,stride=1,padding=1),
                nn.BatchNorm2d(int(outchannel / 2)),
                nn.ReLU(inplace=True),
                nn.Conv2d(int(outchannel/2),outchannel,kernel_size = 1),
                nn.BatchNorm2d(outchannel)
        )
        self.right = shortcut

    def forward(self, x):
        out = self.left(x)
        residual = x
        out += residual
        return F.relu(out)


class MCCFormers_diff_as_Q(nn.Module):
    """
    RSICCFormers_diff
    """

    def __init__(self, feature_dim, dropout, h, w, d_model=512, n_head=4, n_layers=2):
        """
        :param feature_dim: dimension of input features
        :param dropout: dropout rate
        :param d_model: dimension of hidden state
        :param n_head: number of heads in multi head attention
        :param n_layer: number of layers of transformer layer
        """
        super(MCCFormers_diff_as_Q, self).__init__()
        self.d_model = d_model

        # n_layers = 3
        print("encoder_n_layers=", n_layers)

        self.n_layers = n_layers

        self.w_embedding = nn.Embedding(w, int(d_model / 2))
        self.h_embedding = nn.Embedding(h, int(d_model / 2))
        self.embedding_1D = nn.Embedding(h*w, int(d_model))
        self.projection = nn.Conv2d(feature_dim, d_model, kernel_size=1)
        self.projection2 = nn.Conv2d(feature_dim, d_model, kernel_size=1)
        self.transformer = nn.ModuleList([CrossTransformer(dropout, d_model, n_head) for i in range(n_layers)])
        self.resblock = nn.ModuleList([resblock(d_model*2, d_model*2) for i in range(n_layers)])
        self.LN = nn.ModuleList([nn.LayerNorm(d_model * 2) for i in range(n_layers)])
        self._reset_parameters()

    def _reset_parameters(self):
        """Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def forward(self, img_feat1, sem_feat1, img_feat2, sem_feat2):

        # img_feat1 (batch_size, feature_dim, h, w)
        batch = img_feat1.size(0)
        feature_dim = img_feat1.size(1)
        w, h = img_feat1.size(2), img_feat1.size(3)

        img_feat1 = self.projection(img_feat1)
        img_feat2 = self.projection(img_feat2)
        sem_feat1 = self.projection2(sem_feat1)
        sem_feat2 = self.projection2(sem_feat2)

        pos_w = torch.arange(w, device=device).to(device)
        pos_h = torch.arange(h, device=device).to(device)
        embed_w = self.w_embedding(pos_w)
        embed_h = self.h_embedding(pos_h)
        position_embedding = torch.cat([embed_w.unsqueeze(0).repeat(h, 1, 1),
                                        embed_h.unsqueeze(1).repeat(1, w, 1)],
                                       dim=-1)

        position_embedding = position_embedding.permute(2, 0, 1).unsqueeze(0).repeat(batch, 1, 1,1)  # (batch, d_model, h, w)
        # Fuse Type : ADDITION
        # Add positional encodings to image and semantic features
        img_f1 = img_feat1 + position_embedding
        img_f2 = img_feat2 + position_embedding
        semf1 = sem_feat1 + position_embedding
        semf2 = sem_feat2 + position_embedding

        # Flatten spatial dimensions (h*w) and rearrange to (sequence_length, batch, d_model)
        # Needed because nn.Transformer expects inputs in (S, N, E) format
        encoder_output3 = img_f1.view(batch, self.d_model, -1).permute(2, 0, 1)  # image A
        encoder_output4 = img_f2.view(batch, self.d_model, -1).permute(2, 0, 1)  # image B
        encoder_output5 = semf1.view(batch, self.d_model, -1).permute(2, 0, 1)  # semantic A
        encoder_output6 = semf2.view(batch, self.d_model, -1).permute(2, 0, 1)  # semantic B

        # Prepare lists to store outputs from each transformer block
        output1_list = list()  # updated image A features
        output2_list = list()  # updated image B features
        output3_list = list()  # updated semantic A features
        output4_list = list()  # updated semantic B features

        # Pass through the stack of transformer layers
        # Each layer processes the four inputs jointly (cross-attention between A/B + semantics)
        for l in self.transformer:
            output1, output2, output3, output4 = l(
                encoder_output3, encoder_output4, encoder_output5, encoder_output6
            )
            # Collect outputs from this layer
            output1_list.append(output1)
            output2_list.append(output2)
            output3_list.append(output3)
            output4_list.append(output4)

        i = 0
        output = torch.zeros((196, batch, self.d_model * 2)).to(device)
        output = output.permute(1, 2, 0).view(batch, self.d_model * 2, 14, 14)

        for res in self.resblock:
            inputZ = torch.cat([output1_list[i], output2_list[i]], dim=-1)
            inputZ = inputZ.permute(1, 2, 0).view(batch, self.d_model * 2, 14, 14)
            output = output + inputZ
            output = res(output)
            output = output.view(batch, self.d_model * 2, -1).permute(2, 0, 1)
            output = self.LN[i](output)
            output = output.permute(1, 2, 0).view(batch, self.d_model * 2, 14, 14)
            i = i + 1

        output = output.view(batch, self.d_model * 2, -1).permute(2, 0, 1)


        i = 0
        output2 = torch.zeros((196, batch, self.d_model * 2)).to(device)
        output2 = output2.permute(1, 2, 0).view(batch, self.d_model * 2, 14, 14)
        for res in self.resblock:
            inputZ = torch.cat([output3_list[i], output4_list[i]], dim=-1)
            inputZ = inputZ.permute(1, 2, 0).view(batch, self.d_model * 2, 14, 14)
           # input = self.Net1[i](inputZ)
            output2 = output2 + inputZ
            output2 = res(output2)
            output2 = output2.view(batch, self.d_model * 2, -1).permute(2, 0, 1)
            output2 = self.LN[i](output2)
            output2 = output2.permute(1, 2, 0).view(batch, self.d_model * 2, 14, 14)
            i = i + 1

        output2 = output2.view(batch, self.d_model * 2, -1).permute(2, 0, 1)

        return output, output2


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

        self.embedding_1D = nn.Embedding(52, int(d_model))
    def forward(self, x):
        # fixed
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class Mesh_TransformerDecoderLayer(nn.Module):

    __constants__ = ['batch_first', 'norm_first']
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 layer_norm_eps=1e-5, batch_first=False, norm_first=False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(Mesh_TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(int(d_model), nhead, dropout=dropout)
        self.multihead_attn2 = nn.MultiheadAttention(int(d_model), nhead, dropout=dropout)

        self.gate = nn.Sequential(
            nn.Linear(d_model*2, d_model),
            nn.Sigmoid()
        )
        self.gate2 = nn.Sequential(
            nn.Linear(d_model*2, d_model),
            nn.Sigmoid()
        )

        self.gate3 = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm4 = nn.LayerNorm(d_model, eps=layer_norm_eps)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)
        self.dropout5 = nn.Dropout(dropout)

        self.activation = nn.ReLU()
        self.activation2 = nn.Softmax(dim=-1)


    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:

        mem1, mem2 = torch.split(memory, [1024, 1024], dim=2)
        self_att_tgt = self.norm1(tgt + self._sa_block(tgt, tgt_mask, tgt_key_padding_mask))
        # # cross self-attention
        enc_att, attention_weights = self._mha_block((self_att_tgt),
                                               mem1, memory_mask,
                                               memory_key_padding_mask)

        x_enc_att, att_weight2 = self._mha_block((self_att_tgt),
                                               mem2, memory_mask,
                                               memory_key_padding_mask)
        dec_s1 = torch.cat([self_att_tgt, enc_att], dim=2)
        w1 = self.gate(dec_s1)

        dec_s = torch.cat([self_att_tgt, x_enc_att], dim=2)
        w2 = self.gate2(dec_s)
        w3 = self.gate3(self_att_tgt)

        vect = self.norm3(enc_att*w1 + x_enc_att*w2 + self_att_tgt*w3)
        x = self.norm4(vect + self._ff_block(vect))

        return x

    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout1(x)

    # multihead attention block
    def _mha_block(self, x: Tensor, mem: Tensor,
                   attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x,att_weight = self.multihead_attn(x, mem, mem,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask,
                                need_weights=True)
        return self.dropout2(x),att_weight

    def _mha_block2(self, x: Tensor, mem: Tensor,
                   attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x ,att_weight= self.multihead_attn2(x, mem, mem,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask,
                                need_weights=True)
        return self.dropout3(x),att_weight

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout4(x)



class DecoderTransformer(nn.Module):
    """
    Decoder with Transformer.
    """

    def __init__(self, feature_dim, vocab_size, n_head, n_layers, dropout):
        """
        :param n_head: the number of heads in Transformer
        :param n_layers: the number of layers of Transformer
        """
        super(DecoderTransformer, self).__init__()

        # n_layers = 1
        print("decoder_n_layers=",n_layers)

        self.feature_dim = feature_dim
        self.embed_dim = feature_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        # embedding layer
        self.vocab_embedding = nn.Embedding(vocab_size, self.embed_dim)  # vocaburaly embedding

        # Transformer layer
        decoder_layer = Mesh_TransformerDecoderLayer(feature_dim, n_head, dim_feedforward=feature_dim * 4,
                                                   dropout=self.dropout)
        self.transformer = nn.TransformerDecoder(decoder_layer, n_layers)
        self.position_encoding = PositionalEncoding(feature_dim)

        # Linear layer to find scores over vocabulary
        self.wdc = nn.Linear(feature_dim, vocab_size)
        self.dropout = nn.Dropout(p=self.dropout)
        self.init_weights()  # initialize some layers with the uniform distribution

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence
        """
        self.vocab_embedding.weight.data.uniform_(-0.1, 0.1)

        self.wdc.bias.data.fill_(0)
        self.wdc.weight.data.uniform_(-0.1, 0.1)

    def forward(self, memory, memory2, encoded_captions, caption_lengths):
        """
        :param memory:   encoder output 1, shape (S, batch, feature_dim)   e.g., image A features
        :param memory2:  encoder output 2, shape (S, batch, feature_dim)   e.g., image B / semantic features
        :param encoded_captions: token indices, shape (batch, seq_len)
        :param caption_lengths:  lengths for each caption (including <start>/<end>), shape (batch, 1)
        :return:
            pred:             logits over vocabulary, shape (batch, seq_len, vocab_size)
            encoded_captions: captions reordered by descending length (for masking/alignment)
            decode_lengths:   list of effective decode lengths (length-1 to ignore last step)
            sort_ind:         indices used to sort the batch
        """

        # Transformer expects target as (seq_len, batch), so permute from (batch, seq_len)
        tgt = encoded_captions.permute(1, 0)
        tgt_length = tgt.size(0)

        # Create a causal mask so each position can only attend to <= its own index (no look-ahead)
        mask = (torch.triu(torch.ones(tgt_length, tgt_length)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        mask = mask.to(device)

        # Embed target tokens and add positional encodings → (seq_len, batch, d_model)
        tgt_embedding = self.vocab_embedding(tgt)
        tgt_embedding = self.position_encoding(tgt_embedding)

        # Concatenate two encoder memories along the feature dimension → (S, batch, 2*d_model)
        images = torch.cat([memory, memory2], dim=2)

        # Transformer decoder forward pass with causal mask
        # Output is hidden states for each target position → (seq_len, batch, d_model)
        pred = self.transformer(tgt_embedding, images, tgt_mask=mask)

        # Project hidden states to vocabulary logits and apply dropout → (seq_len, batch, vocab_size)
        pred = self.wdc(self.dropout(pred))

        # Revert to (batch, seq_len, vocab_size) for loss computation
        pred = pred.permute(1, 0, 2)

        # ---- Length-aware reordering for consistent batching ----
        # Sort by decreasing caption length (common pattern for sequence models)
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoded_captions = encoded_captions[sort_ind]
        pred = pred[sort_ind]

        # Effective decode lengths (exclude the last step, typically accounts for <end>)
        decode_lengths = (caption_lengths - 1).tolist()

        return pred, encoded_captions, decode_lengths, sort_ind



