# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import Tensor, nn
from torch.distributions.normal import Normal

import math
from typing import Tuple, Type

from .common import MLPBlock, Adapter
from .common import moe_forward#, _gates_to_load, _prob_in_top_k, cv_squared

import copy

class TwoWayTransformer(nn.Module):
    def __init__(
        self,
        args,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        moe = -1,
        k = -1,
    ) -> None:
        """
        A transformer decoder that attends to an input image using
        queries whose positional embedding is supplied.

        Args:
          depth (int): number of layers in the transformer
          embedding_dim (int): the channel dimension for the input embeddings
          num_heads (int): the number of heads for multihead attention. Must
            divide embedding_dim
          mlp_dim (int): the channel dimension internal to the MLP block
          activation (nn.Module): the activation to use in the MLP block
        """
        super().__init__()
        self.args = args
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = nn.ModuleList()
        for i in range(depth):
            if i<args.decoder_adapt_depth:
                if_adapter = args.if_mask_decoder_adapter
            else:
                if_adapter = False
            self.layers.append(
                TwoWayAttentionBlock(
                    args = self.args,
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    attention_downsample_rate=attention_downsample_rate,
                    if_adapter=if_adapter,
                    skip_first_layer_pe=(i == 0),
                    moe=moe,
                    k=k,
                    depth=i,
                )
            )

        self.final_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm_final_attn = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        image_embedding: Tensor,
        image_pe: Tensor,
        point_embedding: Tensor,
        training=False,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
          image_embedding (torch.Tensor): image to attend to. Should be shape
            B x embedding_dim x h x w for any h and w.
          image_pe (torch.Tensor): the positional encoding to add to the image. Must
            have the same shape as image_embedding.
          point_embedding (torch.Tensor): the embedding to add to the query points.
            Must have shape B x N_points x embedding_dim for any N_points.

        Returns:
          torch.Tensor: the processed point_embedding
          torch.Tensor: the processed image_embedding
        """
        # BxCxHxW -> BxHWxC == B x N_image_tokens x C
        bs, c, h, w = image_embedding.shape
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
        image_pe = image_pe.flatten(2).permute(0, 2, 1)

        # Prepare queries
        queries = point_embedding
        keys = image_embedding
        
        moe_total_loss = 0
        gates_total = []

        # Apply transformer blocks and final layernorm
        for layer in self.layers:
            queries, keys, moe_loss, gates = layer(
                queries=queries,
                keys=keys,
                query_pe=point_embedding,
                key_pe=image_pe,
                training=training,
            )
            moe_total_loss += moe_loss
            gates_total.append(gates)

        # Apply the final attention layer from the points to the image
        q = queries + point_embedding
        k = keys + image_pe
        attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)

        return queries, keys, moe_total_loss, gates_total


class TwoWayAttentionBlock(nn.Module):
    def __init__(
        self,
        args,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        if_adapter:bool = False,
        skip_first_layer_pe: bool = False,
        moe = -1,
        k = -1,
        depth = 0,
    ) -> None:
        """
        A transformer block with four layers: (1) self-attention of sparse
        inputs, (2) cross attention of sparse inputs to dense inputs, (3) mlp
        block on sparse inputs, and (4) cross attention of dense inputs to sparse
        inputs.

        Arguments:
          embedding_dim (int): the channel dimension of the embeddings
          num_heads (int): the number of heads in the attention layers
          mlp_dim (int): the hidden dimension of the mlp block
          activation (nn.Module): the activation of the mlp block
          skip_first_layer_pe (bool): skip the PE on the first layer
        """
        super().__init__()
        self.args = args
        self.if_adapter = if_adapter
        self.self_attn = Attention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)
        #||print(if_adapter)

        self.cross_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.mlp = MLPBlock(embedding_dim, mlp_dim, activation)
        self.norm3 = nn.LayerNorm(embedding_dim)

        self.norm4 = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        if self.if_adapter:
            self.scale = 0.5
            self.moe = moe
            
            if moe > 0:
                #self.centroids1 = torch.zeros([moe, 5, embedding_dim]).cuda()
                #self.centroids_mlp = torch.zeros([moe, 5, embedding_dim]).cuda()
                #self.centroids2 = torch.zeros([moe, 4096, embedding_dim]).cuda()

                #self.centroids1 = torch.zeros([moe, embedding_dim]).cuda()
                #self.centroids_mlp = torch.zeros([moe, embedding_dim]).cuda()
                #self.centroids2 = torch.zeros([moe, embedding_dim]).cuda()

                #self.centroids1 = nn.Parameter(self.centroids1)
                #self.centroids_mlp = nn.Parameter(self.centroids_mlp)
                #self.centroids2 = nn.Parameter(self.centroids2)

                self.k = k
                print('Use Dn-MOE with %s experts and %s selection' % (self.moe, self.k))
                self.MLP_Adapter = nn.ModuleList([Adapter(embedding_dim, skip_connect=False) for _ in range(moe)])  # MLP-adapter, no skip connection
                self.Adapter  = nn.ModuleList([Adapter(embedding_dim) for _ in range(moe)])  # with skip connection
                self.Adapter2 = nn.ModuleList([Adapter(embedding_dim) for _ in range(moe)])  # with skip connection

                self.gater = nn.ModuleList([nn.Linear(embedding_dim, self.moe, bias=False) for _ in range(1)])
                self.noise = nn.Linear(embedding_dim, self.moe, bias = False)

                self.softplus = nn.Softplus()

                self.register_buffer("mean", torch.tensor([0.0]))
                self.register_buffer("std", torch.tensor([1.0]))
            else:
                self.MLP_Adapter = Adapter(embedding_dim, skip_connect=False)  # MLP-adapter, no skip connection
                self.Adapter  = Adapter(embedding_dim)  # with skip connection
                self.Adapter2 = Adapter(embedding_dim)  # with skip connection


        self.skip_first_layer_pe = skip_first_layer_pe

    def _gates_to_load(self, gates):
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)

        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in)/noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out)/noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def cv_squared(self, x):
        eps = 1e-10
        return x.float().var() / (x.float().mean()**2 + eps)

    def forward(
        self, queries: Tensor, keys: Tensor, query_pe: Tensor, key_pe: Tensor, labels=None, training=False,
    ) -> Tuple[Tensor, Tensor]:
        moe_loss_total = 0
        gates_total = []

        # Self attention block
        if self.skip_first_layer_pe:
            queries = self.self_attn(q=queries, k=queries, v=queries)
        else:
            q = queries + query_pe
            attn_out = self.self_attn(q=q, k=q, v=queries)
            queries = queries + attn_out
        queries = self.norm1(queries)

        # Cross attention block, tokens attending to image embedding
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out

        # add adapter layer
        if self.if_adapter:
            if self.moe <= 0:
                queries = self.Adapter(queries)
            else:
                moe_out, moe_loss, gates = moe_forward(self, queries, self.Adapter, training)
                queries = moe_out

                moe_loss_total += moe_loss
                gates_total.append(gates)

        queries = self.norm2(queries)

        # MLP block
        mlp_out = self.mlp(queries)
        if self.if_adapter:
            if self.moe <= 0:
                queries = queries + mlp_out + self.scale * self.MLP_Adapter(queries)
            else:
                moe_out, moe_loss, gates = moe_forward(self, queries, self.MLP_Adapter, training)
                queries = queries + mlp_out + self.scale * moe_out

                moe_loss_total += moe_loss
                gates_total.append(gates)
        else:
            queries = queries + mlp_out 
        queries = self.norm3(queries)

        # Cross attention block, image embedding attending to tokens
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)
        keys = keys + attn_out
        
        if self.if_adapter:
            if self.moe <= 0:
                keys = self.Adapter2(keys)
            else:
                moe_out, moe_loss, gates = moe_forward(self, keys, self.Adapter2, training)

                moe_loss_total += moe_loss
                gates_total.append(gates)
                keys = moe_out

        keys = self.norm4(keys)

        return queries, keys, moe_loss_total, gates_total


class Attention(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)

        # Get output
        out = attn @ v
        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out

