# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from models.tts.naturalspeech2.diffusion import Diffusion
from models.tts.naturalspeech2.wavenet import DiffWaveNet
from models.tts.naturalspeech2.prior_encoder import PriorEncoder
from modules.naturalpseech2.transformers import TransformerEncoder, DiffTransformer


class ReferenceEncoder(nn.Module):
    def __init__(
        self,
        encoder_layer=None,
        encoder_hidden=None,
        encoder_head=None,
        conv_filter_size=None,
        conv_kernel_size=None,
        encoder_dropout=None,
        use_skip_connection=None,
        use_new_ffn=None,
        ref_in_dim=None,
        ref_out_dim=None,
        use_query_emb=None,
        num_query_emb=None,
        cfg=None,
    ):
        super().__init__()

        self.encoder_layer = (
            encoder_layer if encoder_layer is not None else cfg.encoder_layer
        )
        self.encoder_hidden = (
            encoder_hidden if encoder_hidden is not None else cfg.encoder_hidden
        )
        self.encoder_head = (
            encoder_head if encoder_head is not None else cfg.encoder_head
        )
        self.conv_filter_size = (
            conv_filter_size if conv_filter_size is not None else cfg.conv_filter_size
        )
        self.conv_kernel_size = (
            conv_kernel_size if conv_kernel_size is not None else cfg.conv_kernel_size
        )
        self.encoder_dropout = (
            encoder_dropout if encoder_dropout is not None else cfg.encoder_dropout
        )
        self.use_skip_connection = (
            use_skip_connection
            if use_skip_connection is not None
            else cfg.use_skip_connection
        )
        self.use_new_ffn = use_new_ffn if use_new_ffn is not None else cfg.use_new_ffn
        self.in_dim = ref_in_dim if ref_in_dim is not None else cfg.ref_in_dim
        self.out_dim = ref_out_dim if ref_out_dim is not None else cfg.ref_out_dim
        self.use_query_emb = (
            use_query_emb if use_query_emb is not None else cfg.use_query_emb
        )
        self.num_query_emb = (
            num_query_emb if num_query_emb is not None else cfg.num_query_emb
        )

        if self.in_dim != self.encoder_hidden:
            self.in_linear = nn.Linear(self.in_dim, self.encoder_hidden)
            self.in_linear.weight.data.normal_(0.0, 0.02)
        else:
            self.in_dim = None

        if self.out_dim != self.encoder_hidden:
            self.out_linear = nn.Linear(self.encoder_hidden, self.out_dim)
            self.out_linear.weight.data.normal_(0.0, 0.02)
        else:
            self.out_linear = None

        self.transformer_encoder = TransformerEncoder(
            encoder_layer=self.encoder_layer,
            encoder_hidden=self.encoder_hidden,
            encoder_head=self.encoder_head,
            conv_kernel_size=self.conv_kernel_size,
            conv_filter_size=self.conv_filter_size,
            encoder_dropout=self.encoder_dropout,
            use_new_ffn=self.use_new_ffn,
            use_cln=False,
            use_skip_connection=False,
            add_diff_step=False,
        )

        if self.use_query_emb:
            self.query_embs = nn.Embedding(self.num_query_emb, self.encoder_hidden)
            self.query_attn = nn.MultiheadAttention(
                self.encoder_hidden, self.encoder_hidden // 64, batch_first=True
            )

    def forward(self, x_ref, key_padding_mask=None):
        # x_ref: (B, T, d_ref)
        # key_padding_mask: (B, T)
        # return speaker embedding: x_spk
        # if self.use_query_embs: shape is (B, N_query, d_out)
        # else: shape is (B, 1, d_out)

        if self.in_linear != None:
            x = self.in_linear(x_ref)

        x = self.transformer_encoder(
            x, key_padding_mask=key_padding_mask, condition=None, diffusion_step=None
        )

        if self.use_query_emb:
            spk_query_emb = self.query_embs(
                torch.arange(self.num_query_emb).to(x.device)
            ).repeat(x.shape[0], 1, 1)
            spk_embs, _ = self.query_attn(
                query=spk_query_emb,
                key=x,
                value=x,
                key_padding_mask=(
                    ~(key_padding_mask.bool()) if key_padding_mask is not None else None
                ),
            )

            if self.out_linear != None:
                spk_embs = self.out_linear(spk_embs)

        else:
            spk_query_emb = None

        return spk_embs, x


class NaturalSpeech2(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        self.reference_encoder = ReferenceEncoder(cfg=cfg.reference_encoder)
        if cfg.diffusion.diff_model_type == "Transformer":
            self.diffusion = Diffusion(
                cfg=cfg.diffusion,
                diff_model=DiffTransformer(cfg=cfg.diffusion.diff_transformer),
            )
        elif cfg.diffusion.diff_model_type == "WaveNet":
            self.diffusion = Diffusion(
                cfg=cfg.diffusion,
                diff_model=DiffWaveNet(cfg=cfg.diffusion.diff_wavenet),
            )
        else:
            raise NotImplementedError()

        self.prior_encoder = PriorEncoder(cfg=cfg.prior_encoder)

        self.reset_parameters()

    def forward(
        self,
        x=None,
        pitch=None,
        duration=None,
        phone_id=None,
        x_ref=None,
        phone_mask=None,
        x_mask=None,
        x_ref_mask=None,
    ):
        reference_embedding, reference_latent = self.reference_encoder(
            x_ref=x_ref, key_padding_mask=x_ref_mask
        )

        prior_out = self.prior_encoder(
            phone_id=phone_id,
            duration=duration,
            pitch=pitch,
            phone_mask=phone_mask,
            mask=x_mask,
            ref_emb=reference_latent,
            ref_mask=x_ref_mask,
            is_inference=False,
        )

        condition_embedding = prior_out["prior_out"]

        diff_out = self.diffusion(
            x=x,
            condition_embedding=condition_embedding,
            x_mask=x_mask,
            reference_embedding=reference_embedding,
        )

        return diff_out, prior_out

    @torch.no_grad()
    def inference(
        self,
        phone_id=None,
        x_ref=None,
        x_ref_mask=None,
        inference_steps=1000,
        sigma=1.2,
    ):
        reference_embedding, reference_latent = self.reference_encoder(
            x_ref=x_ref, key_padding_mask=x_ref_mask
        )

        prior_out = self.prior_encoder(
            phone_id=phone_id,
            duration=None,
            pitch=None,
            phone_mask=None,
            mask=None,
            ref_emb=reference_latent,
            ref_mask=x_ref_mask,
            is_inference=True,
        )

        condition_embedding = prior_out["prior_out"]

        bsz, l, _ = condition_embedding.shape
        if self.cfg.diffusion.diff_model_type == "Transformer":
            z = (
                torch.randn(bsz, l, self.cfg.diffusion.diff_transformer.in_dim).to(
                    condition_embedding.device
                )
                / sigma
            )
        elif self.cfg.diffusion.diff_model_type == "WaveNet":
            z = (
                torch.randn(bsz, l, self.cfg.diffusion.diff_wavenet.input_size).to(
                    condition_embedding.device
                )
                / sigma
            )

        x0 = self.diffusion.reverse_diffusion(
            z=z,
            condition_embedding=condition_embedding,
            x_mask=None,
            reference_embedding=reference_embedding,
            n_timesteps=inference_steps,
        )

        return x0, prior_out

    @torch.no_grad()
    def reverse_diffusion_from_t(
        self,
        x,
        pitch=None,
        duration=None,
        phone_id=None,
        x_ref=None,
        phone_mask=None,
        x_mask=None,
        x_ref_mask=None,
        inference_steps=None,
        t=None,
    ):
        reference_embedding, reference_latent = self.reference_encoder(
            x_ref=x_ref, key_padding_mask=x_ref_mask
        )

        diffusion_step = (
            torch.ones(
                x.shape[0],
                dtype=x.dtype,
                device=x.device,
                requires_grad=False,
            )
            * t
        )

        xt, _ = self.diffusion.forward_diffusion(x0=x, diffusion_step=diffusion_step)

        prior_out = self.prior_encoder(
            phone_id=phone_id,
            duration=duration,
            pitch=pitch,
            phone_mask=phone_mask,
            mask=x_mask,
            ref_emb=reference_latent,
            ref_mask=x_ref_mask,
            is_inference=False,
        )

        condition_embedding = prior_out["prior_out"]

        x0 = self.diffusion.reverse_diffusion_from_t(
            z=xt,
            condition_embedding=condition_embedding,
            x_mask=x_mask,
            reference_embedding=reference_embedding,
            n_timesteps=inference_steps,
            t_start=t,
        )

        return x0

    def reset_parameters(self):
        def _reset_parameters(m):
            if isinstance(m, nn.MultiheadAttention):
                if m._qkv_same_embed_dim:
                    nn.init.normal_(m.in_proj_weight, std=0.02)
                else:
                    nn.init.normal_(m.q_proj_weight, std=0.02)
                    nn.init.normal_(m.k_proj_weight, std=0.02)
                    nn.init.normal_(m.v_proj_weight, std=0.02)

                if m.in_proj_bias is not None:
                    nn.init.constant_(m.in_proj_bias, 0.0)
                    nn.init.constant_(m.out_proj.bias, 0.0)
                if m.bias_k is not None:
                    nn.init.xavier_normal_(m.bias_k)
                if m.bias_v is not None:
                    nn.init.xavier_normal_(m.bias_v)

            elif (
                isinstance(m, nn.Conv1d)
                or isinstance(m, nn.ConvTranspose1d)
                or isinstance(m, nn.Conv2d)
                or isinstance(m, nn.ConvTranspose2d)
            ):
                m.weight.data.normal_(0.0, 0.02)

            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(mean=0.0, std=0.02)
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.Embedding):
                m.weight.data.normal_(mean=0.0, std=0.02)
                if m.padding_idx is not None:
                    m.weight.data[m.padding_idx].zero_()

        self.apply(_reset_parameters)
