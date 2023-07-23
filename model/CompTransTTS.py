import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import PostNet, VarianceAdaptor, MLVAEencoder, Condional_LayerNorm
from utils.tools import get_mask_from_lengths
from .transformers.blocks import LinearNorm
from vector_quantize_pytorch import VectorQuantize



class CompTransTTS(nn.Module):
    """ CompTransTTS """

    def __init__(self, preprocess_config, model_config, train_config):
        super(CompTransTTS, self).__init__()
        self.model_config = model_config
        self.preprocess_config = preprocess_config
        self.train_config = train_config
        
        self.use_vq = model_config["vector_quantizer"]["learn_codebook"]
        self.use_cln = model_config["conditional_layer_norm"]["use_cln"]

        
        if self.use_cln:
            self.layer_norm = Condional_LayerNorm(preprocess_config["preprocessing"]["mel"]["n_mel_channels"])

        if model_config["block_type"] == "transformer_fs2":
            from .transformers.transformer_fs2 import TextEncoder, Decoder
        elif model_config["block_type"] == "transformer":
            from .transformers.transformer import TextEncoder, Decoder
        elif model_config["block_type"] == "lstransformer":
            from .transformers.lstransformer import TextEncoder, Decoder
        elif model_config["block_type"] == "fastformer":
            from .transformers.fastformer import TextEncoder, Decoder
        elif model_config["block_type"] == "conformer":
            from .transformers.conformer import TextEncoder, Decoder
        elif model_config["block_type"] == "reformer":
            from .transformers.reformer import TextEncoder, Decoder
        else:
            raise NotImplementedError

        self.encoder = TextEncoder(model_config)
        # MLVAE Encoder
        if model_config["VAE"]["type"] == "MLVAE":
            self.mlvae_encoder = MLVAEencoder(model_config,preprocess_config)
            
        # Vector Quantizer Layer
        
        if self.use_vq:
            
            self.mlvae_acc_vq = VectorQuantize(
                dim = model_config["vector_quantizer"]["accent"]["dim"],
                codebook_size =  model_config["vector_quantizer"]["accent"]["codebook_size"],    # codebook size
                decay= model_config["vector_quantizer"]["accent"]["decay"],
                commitment_weight= model_config["vector_quantizer"]["accent"]["commitment_weight"]
                )
            
            self.mlvae_spk_vq = VectorQuantize(
                dim = model_config["vector_quantizer"]["speaker"]["dim"],     # specify number of quantizers
                codebook_size =  model_config["vector_quantizer"]["speaker"]["codebook_size"],    # codebook size
                decay= model_config["vector_quantizer"]["speaker"]["decay"],
                commitment_weight= model_config["vector_quantizer"]["speaker"]["commitment_weight"]
                )
                        
                                
        self.variance_adaptor = VarianceAdaptor(preprocess_config, model_config, train_config, self.encoder.d_model)
        self.decoder = Decoder(model_config)
        self.mel_linear = nn.Linear(
            self.decoder.d_model,
            preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
        )
        self.postnet = PostNet()
        
        self.lin_proj = LinearNorm(model_config["lin_proj"]["in_dim"], model_config["lin_proj"]["out_dim"])

        self.speaker_emb = None
                
    def forward(
        self,
        speakers,
        texts,
        src_lens,
        max_src_len,
        mels=None,
        mel_lens=None,
        max_mel_len=None,
        p_targets=None,
        e_targets=None,
        d_targets=None,
        attn_priors=None,
        spker_embeds=None,
        accents=None,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
        step=None,
    ):
        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        mel_masks = (
            get_mask_from_lengths(mel_lens, max_mel_len)
            if mel_lens is not None
            else None
        )

        texts, text_embeds = self.encoder(texts, src_masks)
        # MLVAE Module
        if self.model_config["VAE"]["type"] == "MLVAE":
            (z_acc, z_spk, z_acc_sg, mlvae_stats) = self.mlvae_encoder(mels,acc_labels=accents)
        else:
            speaker_embeds = spker_embeds
        
        # Vetor Quantization
        vq_loss = 0.0
        commit_loss_acc = 0.0
        commit_loss_spk = 0.0
        
        if self.model_config["vector_quantizer"]["learn_codebook"]:
            quantized_acc, indices_acc, commit_loss_acc = self.mlvae_acc_vq(z_acc)
            quantized_spk, indices_spk, commit_loss_spk = self.mlvae_spk_vq(z_spk)
            vae_outs=torch.cat([quantized_acc,quantized_spk],axis=1)
            # vae_outs_=torch.cat([quantized_acc,quantized_spk],axis=1).unsqueeze(1).expand(-1,max_mel_len,-1)
            speaker_embeds = vae_outs
            vq_loss = (commit_loss_acc,commit_loss_spk)
            mlvae_stats = (vq_loss,mlvae_stats)
        else:
            # vae_outs=torch.cat([z_acc,z_spk],axis=1).unsqueeze(1).expand(-1,max_mel_len,-1)
            vae_outs=torch.cat([z_acc,z_spk],axis=1)
            # vae_outs=torch.cat([quantized_acc,quantized_spk],axis=1)
            vq_loss = (commit_loss_acc,commit_loss_spk)
            mlvae_stats = (vq_loss,mlvae_stats)
            speaker_embeds = vae_outs

        (
            output,
            p_targets,
            p_predictions,
            e_targets,
            e_predictions,
            log_d_predictions,
            d_rounded,
            mel_lens,
            mel_masks,
            attn_outs,
            prosody_info,
        ) = self.variance_adaptor(
            speaker_embeds,
            texts,
            text_embeds,
            src_lens,
            src_masks,
            mels,
            mel_lens,
            mel_masks,
            max_mel_len,
            p_targets,
            e_targets,
            d_targets,
            attn_priors,
            p_control,
            e_control,
            d_control,
            step,
        )

        output, mel_masks = self.decoder(output, mel_masks)
        output = self.mel_linear(output)
        
        if self.use_cln:
            output = self.layer_norm(output, speaker_embeds)

        postnet_output = self.postnet(output) + output

        return (
            output,
            mlvae_stats,
            postnet_output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            src_masks,
            mel_masks,
            src_lens,
            mel_lens,
            attn_outs,
            prosody_info,
            p_targets,
            e_targets,
        )


    def inference_stats(
        self,
        speakers,
        texts,
        src_lens,
        max_src_len,
        mels=None,
        mel_lens=None,
        max_mel_len=None,
        p_targets=None,
        e_targets=None,
        d_targets=None,
        attn_priors=None,
        spker_embeds=None,
        accents=None,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
        step=None,
        z_acc=None,
        z_spk=None,
    ):
        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        mel_masks = (
            get_mask_from_lengths(mel_lens, max_mel_len)
            if mel_lens is not None
            else None
        )

        texts, text_embeds = self.encoder(texts, src_masks)
        # MLVAE Module
        # if self.model_config["VAE"]["type"] == "MLVAE":
        #     (z_acc, z_spk, z_acc_sg, mlvae_stats) = self.mlvae_encoder(mels,acc_labels=accents)
        # else:
        #     speaker_embeds = spker_embeds
        
        # if self.model_config["VAE"]["type"] == "MLVAE":
        #     (z_acc, z_spk, z_acc_sg, mlvae_stats) = self.mlvae_encoder(mels,acc_labels=accents)
        # else:
        #     speaker_embeds = spker_embeds

        
        mlvae_stats=(z_acc,z_acc,z_spk,z_spk,z_acc,z_acc)
        # mlvae_stats=(class_latent_embeddings, style_latent_embeddings, accent_latent_embeddings, (grouped_mu, grouped_logvar, style_latent_space_mu, style_latent_space_logvar, accent_latent_space_mu, accent_latent_space_logvar))

        # Vetor Quantization
        vq_loss = 0.0
        commit_loss_acc = 0.0
        commit_loss_spk = 0.0
        
        if self.model_config["vector_quantizer"]["learn_codebook"]:
            quantized_acc, indices_acc, commit_loss_acc = self.mlvae_acc_vq(z_acc)
            quantized_spk, indices_spk, commit_loss_spk = self.mlvae_spk_vq(z_spk)
            vae_outs=torch.cat([quantized_acc,quantized_spk],axis=1)
            # vae_outs_=torch.cat([quantized_acc,quantized_spk],axis=1).unsqueeze(1).expand(-1,max_mel_len,-1)
            speaker_embeds = vae_outs
            vq_loss = (commit_loss_acc,commit_loss_spk)
            mlvae_stats = (vq_loss,mlvae_stats)
        else:
            # vae_outs=torch.cat([z_acc,z_spk],axis=1).unsqueeze(1).expand(-1,max_mel_len,-1)
            # vae_outs=torch.cat([quantized_acc,quantized_spk],axis=1)
            vae_outs=torch.cat([z_acc,z_spk],axis=1)
            vq_loss = (commit_loss_acc,commit_loss_spk)
            mlvae_stats = (vq_loss,mlvae_stats)
            speaker_embeds = vae_outs

        (
            output,
            p_targets,
            p_predictions,
            e_targets,
            e_predictions,
            log_d_predictions,
            d_rounded,
            mel_lens,
            mel_masks,
            attn_outs,
            prosody_info,
        ) = self.variance_adaptor(
            speaker_embeds,
            texts,
            text_embeds,
            src_lens,
            src_masks,
            mels,
            mel_lens,
            mel_masks,
            max_mel_len,
            p_targets,
            e_targets,
            d_targets,
            attn_priors,
            p_control,
            e_control,
            d_control,
            step,
        )

        output, mel_masks = self.decoder(output, mel_masks)
        output = self.mel_linear(output)
        
        if self.use_cln:
            output = self.layer_norm(output, speaker_embeds)

        postnet_output = self.postnet(output) + output

        return (
            output,
            mlvae_stats,
            postnet_output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            src_masks,
            mel_masks,
            src_lens,
            mel_lens,
            attn_outs,
            prosody_info,
            p_targets,
            e_targets,
        )
