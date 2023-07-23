import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math
import numpy
from utils.tools import get_variance_level, ssim
from text import sil_phonemes_ids

from collections import OrderedDict


def buildupfc(n_iter, Btype='expo', n_stop=25000, n_up=5000, start=-5.0, stop=0.0):
    if Btype=='expo':
        Llow = numpy.ones(n_up)*(10**start)
        Lhigh = numpy.ones(n_iter-n_stop)*(10**stop)
        Lramp = numpy.ones(n_stop-n_up)*(10**(numpy.linspace(start,stop,n_stop-n_up)))
    else:
        Llow = numpy.ones(n_up)*start
        Lhigh = numpy.ones(n_iter-n_stop)*stop
        Lramp = numpy.linspace(start,stop,n_stop-n_up)
    return numpy.concatenate((Llow,Lramp,Lhigh))


class CompTransTTSLoss(nn.Module):
    """ CompTransTTS Loss """

    def __init__(self, preprocess_config, model_config, train_config):
        super(CompTransTTSLoss, self).__init__()
        _, self.energy_feature_level = \
                get_variance_level(preprocess_config, model_config, data_loading=False)
        self.loss_config = train_config["loss"]
        self.pitch_config = preprocess_config["preprocessing"]["pitch"]
        self.pitch_type = self.pitch_config["pitch_type"]
        self.use_pitch_embed = model_config["variance_embedding"]["use_pitch_embed"]
        self.use_energy_embed = model_config["variance_embedding"]["use_energy_embed"]
        self.model_type = model_config["prosody_modeling"]["model_type"]
        self.learn_alignment = model_config["duration_modeling"]["learn_alignment"]
        self.learn_codebook = model_config["vector_quantizer"]["learn_codebook"]
        self.binarization_loss_enable_steps = train_config["duration"]["binarization_loss_enable_steps"]
        self.binarization_loss_warmup_steps = train_config["duration"]["binarization_loss_warmup_steps"]
        self.gmm_mdn_beta = train_config["prosody"]["gmm_mdn_beta"]
        self.prosody_loss_enable_steps = train_config["prosody"]["prosody_loss_enable_steps"]
        self.var_start_steps = train_config["step"]["var_start_steps"]
        self.sum_loss = ForwardSumLoss()
        self.bin_loss = BinLoss()
        self.sil_ph_ids = sil_phonemes_ids()
        
        self.encoder_type = model_config["VAE"]["type"]
        
        self.restore_step = train_config["step"]["restore_step"]
        self.constant_steps = train_config["step"]["constant_steps"]
        
        
        self.n_iter = train_config["step"]["total_step"]
        self.n_stopKL = train_config["linbuildkl"]["n_stop"]
        self.n_upKL = train_config["linbuildkl"]["n_up"]
        self.stopKL = train_config["linbuildkl"]["stop"]
        self.startKL = train_config["linbuildkl"]["start"]
        self.KLBtype = train_config["linbuildkl"]["type"]
        self.n_stopadv = train_config["linbuildadv"]["n_stop"]
        self.n_upadv = train_config["linbuildadv"]["n_up"]
        self.stopadv = train_config["linbuildadv"]["stop"]
        self.startadv = train_config["linbuildadv"]["start"]
        self.advBtype = train_config["linbuildadv"]["type"]

        self.acc_kl_coef = train_config["coeffs"]["acc_kl"]
        self.spk_kl_coef = train_config["coeffs"]["spk_kl"]
        self.acc_adv_coef = train_config["coeffs"]["acc_adv"]
        self.spk_adv_coef = train_config["coeffs"]["spk_adv"]
        self.reco_coef = train_config["coeffs"]["reco"]
        self.n_accent_classes = train_config["n_accent_classes"]
        self.LKL = buildupfc(self.n_iter,self.KLBtype,self.n_stopKL,self.n_upKL,start=self.startKL,stop=self.stopKL)


        # self.learn_speaker_emb = preprocess_config["preprocessing"]["learn_speaker_emb"]
        # self.criterion = torch.nn.CrossEntropyLoss(reduction='mean')

    # def gaussian_probability(self, sigma, mu, target, mask=None):
    #     target = target.unsqueeze(2).expand_as(sigma)
    #     ret = (1.0 / math.sqrt(2 * math.pi)) * torch.exp(-0.5 * ((target - mu) / sigma)**2) / sigma
    #     if mask is not None:
    #         ret = ret.masked_fill(mask.unsqueeze(-1).unsqueeze(-1), 0)
    #     return torch.prod(ret, 3)

    # def mdn_loss(self, w, sigma, mu, target, mask=None):
    #     """
    #     w -- [B, src_len, num_gaussians]
    #     sigma -- [B, src_len, num_gaussians, out_features]
    #     mu -- [B, src_len, num_gaussians, out_features]
    #     target -- [B, src_len, out_features]
    #     mask -- [B, src_len]
    #     """
    #     prob = w * self.gaussian_probability(sigma, mu, target, mask)
    #     nll = -torch.log(torch.sum(prob, dim=2))
    #     if mask is not None:
    #         nll = nll.masked_fill(mask, 0)
    #     l_pp = torch.sum(nll, dim=1)
    #     return torch.mean(l_pp)

    def log_gaussian_probability(self, sigma, mu, target, mask=None):
        """
        prob -- [B, src_len, num_gaussians]
        """
        target = target.unsqueeze(2).expand_as(sigma)
        prob = torch.log((1.0 / (math.sqrt(2 * math.pi)*sigma))) -0.5 * ((target - mu) / sigma)**2
        if mask is not None:
            prob = prob.masked_fill(mask.unsqueeze(-1).unsqueeze(-1), 0)
        prob = torch.sum(prob, dim=3)

        return prob

    def mdn_loss(self, w, sigma, mu, target, mask=None):
        """
        w -- [B, src_len, num_gaussians]
        sigma -- [B, src_len, num_gaussians, out_features]
        mu -- [B, src_len, num_gaussians, out_features]
        target -- [B, src_len, out_features]
        mask -- [B, src_len]
        """
        prob = torch.log(w) + self.log_gaussian_probability(sigma, mu, target, mask)
        nll = -torch.logsumexp(prob, 2)
        if mask is not None:
            nll = nll.masked_fill(mask, 0)
        return torch.mean(nll)

        # prob = w * self.gaussian_probability(sigma, mu, target, mask)
        # # prob = w.unsqueeze(-1) * self.gaussian_probability(sigma, mu, target, mask)
        # nll = -torch.log(torch.sum(prob, dim=2))
        # # nll = -torch.sum(prob, dim=2)
        # if mask is not None:
        #     nll = nll.masked_fill(mask, 0)
        #     # nll = nll.masked_fill(mask.unsqueeze(-1), 0)
        # l_pp = torch.sum(nll, dim=1)
        # return torch.mean(l_pp)

    def get_mel_loss(self, mel_predictions, mel_targets):
        mel_targets.requires_grad = False
        mel_predictions = mel_predictions.masked_fill(self.mel_masks_fill.unsqueeze(-1), 0)
        mel_targets = mel_targets.masked_fill(self.mel_masks_fill.unsqueeze(-1), 0)
        mel_loss = self.l1_loss(mel_predictions, mel_targets)
        return mel_loss

    def l1_loss(self, decoder_output, target):
        # decoder_output : B x T x n_mel
        # target : B x T x n_mel
        l1_loss = F.l1_loss(decoder_output, target, reduction="none")
        weights = self.weights_nonzero_speech(target)
        l1_loss = (l1_loss * weights).sum() / weights.sum()
        return l1_loss

    def ssim_loss(self, decoder_output, target, bias=6.0):
        # decoder_output : B x T x n_mel
        # target : B x T x n_mel
        assert decoder_output.shape == target.shape
        weights = self.weights_nonzero_speech(target)
        decoder_output = decoder_output[:, None] + bias
        target = target[:, None] + bias
        ssim_loss = 1 - ssim(decoder_output, target, size_average=False)
        ssim_loss = (ssim_loss * weights).sum() / weights.sum()
        return ssim_loss

    def weights_nonzero_speech(self, target):
        # target : B x T x mel
        # Assign weight 1.0 to all labels except for padding (id=0).
        dim = target.size(-1)
        return target.abs().sum(-1, keepdim=True).ne(0).float().repeat(1, 1, dim)

    def get_duration_loss(self, dur_pred, dur_gt, txt_tokens):
        """
        :param dur_pred: [B, T], float, log scale
        :param txt_tokens: [B, T]
        :return:
        """
        dur_gt.requires_grad = False
        losses = {}
        B, T = txt_tokens.shape
        nonpadding = self.src_masks.float()
        dur_gt = dur_gt.float() * nonpadding
        is_sil = torch.zeros_like(txt_tokens).bool()
        for p_id in self.sil_ph_ids:
            is_sil = is_sil | (txt_tokens == p_id)
        is_sil = is_sil.float()  # [B, T_txt]

        # phone duration loss
        if self.loss_config["dur_loss"] == "mse":
            losses["pdur"] = F.mse_loss(dur_pred, (dur_gt + 1).log(), reduction="none")
            losses["pdur"] = (losses["pdur"] * nonpadding).sum() / nonpadding.sum()
            dur_pred = (dur_pred.exp() - 1).clamp(min=0)
        elif self.loss_config["dur_loss"] == "mog":
            return NotImplementedError
        elif self.loss_config["dur_loss"] == "crf":
            # losses["pdur"] = -self.model.dur_predictor.crf(
            #     dur_pred, dur_gt.long().clamp(min=0, max=31), mask=nonpadding > 0, reduction="mean")
            return NotImplementedError
        losses["pdur"] = losses["pdur"] * self.loss_config["lambda_ph_dur"]

        # use linear scale for sent and word duration
        if self.loss_config["lambda_word_dur"] > 0:
            word_id = (is_sil.cumsum(-1) * (1 - is_sil)).long()
            word_dur_p = dur_pred.new_zeros([B, word_id.max() + 1]).scatter_add(1, word_id, dur_pred)[:, 1:]
            word_dur_g = dur_gt.new_zeros([B, word_id.max() + 1]).scatter_add(1, word_id, dur_gt)[:, 1:]
            wdur_loss = F.mse_loss((word_dur_p + 1).log(), (word_dur_g + 1).log(), reduction="none")
            word_nonpadding = (word_dur_g > 0).float()
            wdur_loss = (wdur_loss * word_nonpadding).sum() / word_nonpadding.sum()
            losses["wdur"] = wdur_loss * self.loss_config["lambda_word_dur"]
        if self.loss_config["lambda_sent_dur"] > 0:
            sent_dur_p = dur_pred.sum(-1)
            sent_dur_g = dur_gt.sum(-1)
            sdur_loss = F.mse_loss((sent_dur_p + 1).log(), (sent_dur_g + 1).log(), reduction="mean")
            losses["sdur"] = sdur_loss.mean() * self.loss_config["lambda_sent_dur"]
        return losses

    def get_pitch_loss(self, pitch_predictions, pitch_targets):
        for _, pitch_target in pitch_targets.items():
            if pitch_target is not None:
                pitch_target.requires_grad = False
        losses = {}
        if self.pitch_type == "ph":
            nonpadding = self.src_masks.float()
            pitch_loss_fn = F.l1_loss if self.loss_config["pitch_loss"] == "l1" else F.mse_loss
            losses["f0"] = (pitch_loss_fn(pitch_predictions["pitch_pred"][:, :, 0], pitch_targets["f0"],
                                          reduction="none") * nonpadding).sum() \
                           / nonpadding.sum() * self.loss_config["lambda_f0"]
        else:
            mel2ph = pitch_targets["mel2ph"]  # [B, T_s]
            f0 = pitch_targets["f0"]
            uv = pitch_targets["uv"]
            nonpadding = self.mel_masks.float()
            if self.pitch_type == "cwt":
                cwt_spec = pitch_targets[f"cwt_spec"]
                f0_mean = pitch_targets["f0_mean"]
                f0_std = pitch_targets["f0_std"]
                cwt_pred = pitch_predictions["cwt"][:, :, :10]
                f0_mean_pred = pitch_predictions["f0_mean"]
                f0_std_pred = pitch_predictions["f0_std"]
                losses["C"] = self.cwt_loss(cwt_pred, cwt_spec) * self.loss_config["lambda_f0"]
                if self.pitch_config["use_uv"]:
                    assert pitch_predictions["cwt"].shape[-1] == 11
                    uv_pred = pitch_predictions["cwt"][:, :, -1]
                    losses["uv"] = (F.binary_cross_entropy_with_logits(uv_pred, uv, reduction="none") * nonpadding) \
                                    .sum() / nonpadding.sum() * self.loss_config["lambda_uv"]
                losses["f0_mean"] = F.l1_loss(f0_mean_pred, f0_mean) * self.loss_config["lambda_f0"]
                losses["f0_std"] = F.l1_loss(f0_std_pred, f0_std) * self.loss_config["lambda_f0"]
                # if self.loss_config["cwt_add_f0_loss"]:
                #     f0_cwt_ = cwt2f0_norm(cwt_pred, f0_mean_pred, f0_std_pred, mel2ph, self.pitch_config)
                #     self.add_f0_loss(f0_cwt_[:, :, None], f0, uv, losses, nonpadding=nonpadding)
            elif self.pitch_type == "frame":
                self.add_f0_loss(pitch_predictions["pitch_pred"], f0, uv, losses, nonpadding=nonpadding)
        return losses

    def add_f0_loss(self, p_pred, f0, uv, losses, nonpadding):
        assert p_pred[..., 0].shape == f0.shape
        if self.pitch_config["use_uv"]:
            assert p_pred[..., 1].shape == uv.shape
            losses["uv"] = (F.binary_cross_entropy_with_logits(
                p_pred[:, :, 1], uv, reduction="none") * nonpadding).sum() \
                           / nonpadding.sum() * self.loss_config["lambda_uv"]
            nonpadding = nonpadding * (uv == 0).float()

        f0_pred = p_pred[:, :, 0]
        if self.loss_config["pitch_loss"] in ["l1", "l2"]:
            pitch_loss_fn = F.l1_loss if self.loss_config["pitch_loss"] == "l1" else F.mse_loss
            losses["f0"] = (pitch_loss_fn(f0_pred, f0, reduction="none") * nonpadding).sum() \
                           / nonpadding.sum() * self.loss_config["lambda_f0"]
        elif self.loss_config["pitch_loss"] == "ssim":
            return NotImplementedError

    def cwt_loss(self, cwt_p, cwt_g):
        if self.loss_config["cwt_loss"] == "l1":
            return F.l1_loss(cwt_p, cwt_g)
        if self.loss_config["cwt_loss"] == "l2":
            return F.mse_loss(cwt_p, cwt_g)
        if self.loss_config["cwt_loss"] == "ssim":
            return self.ssim_loss(cwt_p, cwt_g, 20)

    def get_energy_loss(self, energy_predictions, energy_targets):
        energy_targets.requires_grad = False
        if self.energy_feature_level == "phoneme_level":
            energy_predictions = energy_predictions.masked_select(self.src_masks)
            energy_targets = energy_targets.masked_select(self.src_masks)
        if self.energy_feature_level == "frame_level":
            energy_predictions = energy_predictions.masked_select(self.mel_masks)
            energy_targets = energy_targets.masked_select(self.mel_masks)
        energy_loss = F.l1_loss(energy_predictions, energy_targets)
        return energy_loss

    def get_init_losses(self, device):
        duration_loss = {
            "pdur": torch.zeros(1).to(device),
            "wdur": torch.zeros(1).to(device),
            "sdur": torch.zeros(1).to(device),
        }
        pitch_loss = {}
        if self.pitch_type == "ph":
            pitch_loss["f0"] = torch.zeros(1).to(device)
        else:
            if self.pitch_type == "cwt":
                pitch_loss["C"] = torch.zeros(1).to(device)
                if self.pitch_config["use_uv"]:
                    pitch_loss["uv"] = torch.zeros(1).to(device)
                pitch_loss["f0_mean"] = torch.zeros(1).to(device)
                pitch_loss["f0_std"] = torch.zeros(1).to(device)
            elif self.pitch_type == "frame":
                if self.pitch_config["use_uv"]:
                    pitch_loss["uv"] = torch.zeros(1).to(device)
                if self.loss_config["pitch_loss"] in ["l1", "l2"]:
                    pitch_loss["f0"] = torch.zeros(1).to(device)
        energy_loss = torch.zeros(1).to(device)
        return duration_loss, pitch_loss, energy_loss
    
    def KL_loss(self, mu, var):
        return torch.mean(0.5 * torch.sum(torch.exp(var) + mu ** 2 - 1. - var, 1))
    
    
    def get_encoder_loss(self, id_, vae_stats, classes_, acc_kl_lambda, spk_kl_lambda, encoder_type):
        
        if (encoder_type == 'MLVAE') and (acc_kl_lambda != 0.0 or spk_kl_lambda != 0.0):
            loss_class =  acc_kl_lambda * self.KL_loss(vae_stats[0], vae_stats[1])
            loss_style =  spk_kl_lambda * self.KL_loss(vae_stats[2], vae_stats[3])

            loss = loss_style + loss_class
            
        return loss, loss_class, loss_style

    def forward(self, inputs, predictions, step):
        (
            speakers,
            texts,
            _,
            _,
            mel_targets,
            _,
            _,
            pitch_targets,
            energy_targets,
            duration_targets,
            _,
            _,
            accents,
        ) = inputs[2:]
        (
            mel_predictions,
            mlvae_stats,
            postnet_mel_predictions,
            pitch_predictions,
            energy_predictions,
            log_duration_predictions,
            _,
            src_masks,
            mel_masks,
            src_lens,
            mel_lens,
            attn_outs,
            prosody_info,
        ) = predictions
        vq_loss,mlvae_stats = mlvae_stats
        self.src_masks = ~src_masks
        mel_masks = ~mel_masks
        if self.learn_alignment:
            attn_soft, attn_hard, attn_hard_dur, attn_logprob = attn_outs
            duration_targets = attn_hard_dur
        mel_targets = mel_targets[:, : mel_masks.shape[1], :]
        self.mel_masks = mel_masks[:, :mel_masks.shape[1]]
        self.mel_masks_fill = ~self.mel_masks

        mel_loss = self.get_mel_loss(mel_predictions, mel_targets)
        postnet_mel_loss = self.get_mel_loss(postnet_mel_predictions, mel_targets)
        if self.constant_steps:
            acc_kl_lambda = self.acc_kl_coef
            spk_kl_lambda = self.spk_kl_coef
        else:
            acc_kl_lambda = self.LKL[step-self.restore_step] * self.acc_kl_coef
            spk_kl_lambda = self.LKL[step-self.restore_step] * self.spk_kl_coef

        
        
        encoder_loss, acc_kl_loss, spk_kl_loss = self.get_encoder_loss(accents, 
                                                                       mlvae_stats, 
                                                                       self.n_accent_classes, 
                                                                       acc_kl_lambda,
                                                                       spk_kl_lambda,
                                                                       self.encoder_type,
                                                                       )
        
        acc_vq_loss = spk_vq_loss = torch.zeros(1).to(mel_targets.device)
        
        if self.learn_codebook:
            acc_vq_loss,spk_vq_loss = vq_loss        

        ctc_loss = bin_loss = torch.zeros(1).to(mel_targets.device)
        if self.learn_alignment:
            ctc_loss = self.sum_loss(attn_logprob=attn_logprob, in_lens=src_lens, out_lens=mel_lens)
            if step < self.binarization_loss_enable_steps:
                bin_loss_weight = 0.
            else:
                bin_loss_weight = min((step-self.binarization_loss_enable_steps) / self.binarization_loss_warmup_steps, 1.0) * 1.0
            bin_loss = self.bin_loss(hard_attention=attn_hard, soft_attention=attn_soft) * bin_loss_weight

        prosody_loss = torch.zeros(1).to(mel_targets.device)
        if self.training and self.model_type == "du2021" and step > self.prosody_loss_enable_steps:
            w, sigma, mu, prosody_embeddings = prosody_info
            prosody_loss = self.gmm_mdn_beta * self.mdn_loss(w, sigma, mu, prosody_embeddings.detach(), ~src_masks)
        elif self.training and self.model_type == "liu2021" and step > self.prosody_loss_enable_steps:
            up_tgt, pp_tgt, up_vec, pp_vec, _ = prosody_info
            prosody_loss = F.l1_loss(up_tgt, up_vec)
            # prosody_loss = F.l1_loss(
            prosody_loss += F.l1_loss(
                pp_tgt.masked_select(src_masks.unsqueeze(-1)), pp_vec.masked_select(src_masks.unsqueeze(-1)))

        total_loss = mel_loss + postnet_mel_loss + ctc_loss + bin_loss + prosody_loss + encoder_loss \
            + acc_vq_loss + spk_vq_loss

        duration_loss, pitch_loss, energy_loss = self.get_init_losses(mel_targets.device)
        if step > self.var_start_steps:
            duration_loss = self.get_duration_loss(log_duration_predictions, duration_targets, texts)
            if self.use_pitch_embed:
                pitch_loss = self.get_pitch_loss(pitch_predictions, pitch_targets)
            if self.use_energy_embed:
                energy_loss = self.get_energy_loss(energy_predictions, energy_targets)
            total_loss += sum(duration_loss.values()) + sum(pitch_loss.values()) + energy_loss
            
        # speaker_loss=torch.zeros(1)    
        # if self.learn_speaker_emb:
        #     speaker_loss = self.criterion(logits[0],speakers)
        #     total_loss += speaker_loss

        return (
            total_loss,
            mel_loss,
            postnet_mel_loss,
            pitch_loss,
            energy_loss,
            duration_loss,
            ctc_loss,
            bin_loss,
            prosody_loss,
            encoder_loss,
            acc_kl_loss,
            spk_kl_loss,
            acc_vq_loss,
            spk_vq_loss,
        )


class ForwardSumLoss(nn.Module):
    def __init__(self, blank_logprob=-1):
        super().__init__()
        self.log_softmax = nn.LogSoftmax(dim=3)
        self.ctc_loss = nn.CTCLoss(zero_infinity=True)
        self.blank_logprob = blank_logprob

    def forward(self, attn_logprob, in_lens, out_lens):
        key_lens = in_lens
        query_lens = out_lens
        attn_logprob_padded = F.pad(input=attn_logprob, pad=(1, 0), value=self.blank_logprob)

        total_loss = 0.0
        for bid in range(attn_logprob.shape[0]):
            target_seq = torch.arange(1, key_lens[bid] + 1).unsqueeze(0)
            curr_logprob = attn_logprob_padded[bid].permute(1, 0, 2)[: query_lens[bid], :, : key_lens[bid] + 1]

            curr_logprob = self.log_softmax(curr_logprob[None])[0]
            loss = self.ctc_loss(
                curr_logprob,
                target_seq,
                input_lengths=query_lens[bid : bid + 1],
                target_lengths=key_lens[bid : bid + 1],
            )
            total_loss += loss

        total_loss /= attn_logprob.shape[0]
        return total_loss


class BinLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hard_attention, soft_attention):
        log_sum = torch.log(torch.clamp(soft_attention[hard_attention == 1], min=1e-12)).sum()
        return -log_sum / hard_attention.sum()


class ArcMarginModel(torch.nn.Module):
    """

    """
    def __init__(self, args):
        super(ArcMarginModel, self).__init__()

        self.weight = Parameter(torch.FloatTensor(num_classes, args.emb_size))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = args.easy_margin
        self.m = args.margin_m
        self.s = args.margin_s

        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

    def forward(self, input, label):
        """

        :param input:
        :param label:
        :return:
        """
        x = F.normalize(input)
        W = F.normalize(self.weight)
        cosine = F.linear(x, W)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m  # cos(theta + m)
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros(cosine.size(), device=device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output


def l2_norm(input, axis=1):
    """

    :param input:
    :param axis:
    :return:
    """
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


class ArcFace(torch.nn.Module):
    """

    """
    # implementation of additive margin softmax loss in https://arxiv.org/abs/1801.05599
    def __init__(self, embedding_size, classnum, s=64., m=0.5):
        super(ArcFace, self).__init__()
        self.classnum = classnum
        self.kernel = Parameter(torch.Tensor(embedding_size, classnum))
        # initial kernel
        self.kernel.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.m = m  # the margin value, default is 0.5
        self.s = s  # scalar value default is 64, see normface https://arxiv.org/abs/1704.06369
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.mm = self.sin_m * m  # issue 1
        self.threshold = math.cos(math.pi - m)

    def forward(self, embbedings, target):
        """

        :param embbedings:
        :param target:
        :return:
        """
        # weights norm
        nB = len(embbedings)
        kernel_norm = l2_norm(self.kernel, axis=0)
        # cos(theta+m)
        cos_theta = torch.mm(embbedings, kernel_norm)
        #         output = torch.mm(embbedings,kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
        cos_theta_2 = torch.pow(cos_theta, 2)
        sin_theta_2 = 1 - cos_theta_2
        sin_theta = torch.sqrt(sin_theta_2)
        cos_theta_m = (cos_theta * self.cos_m - sin_theta * self.sin_m)
        # this condition controls the theta+m should in range [0, pi]
        #      0<=theta+m<=pi
        #     -m<=theta<=pi-m
        cond_v = cos_theta - self.threshold
        cond_mask = cond_v <= 0
        # when theta not in [0,pi], use cosface instead
        keep_val = (cos_theta - self.mm)
        cos_theta_m[cond_mask] = keep_val[cond_mask]
        # a little bit hacky way to prevent in_place operation on cos_theta
        output = cos_theta * 1.0
        idx_ = torch.arange(0, nB, dtype=torch.long)
        output[idx_, target] = cos_theta_m[idx_, target]
        output *= self.s  # scale up in order to make softmax work, first introduced in normface
        return output


##################################  Cosface head #############################################################

class Am_softmax(torch.nn.Module):
    """

    """
    # implementation of additive margin softmax loss in https://arxiv.org/abs/1801.05599
    def __init__(self, embedding_size=512, classnum=51332):
        super(Am_softmax, self).__init__()
        self.classnum = classnum
        self.kernel = Parameter(torch.Tensor(embedding_size, classnum))
        # initial kernel
        self.kernel.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.m = 0.35  # additive margin recommended by the paper
        self.s = 30.  # see normface https://arxiv.org/abs/1704.06369

    def forward(self, embbedings, label):
        """

        :param embbedings:
        :param label:
        :return:
        """
        kernel_norm = l2_norm(self.kernel, axis=0)
        cos_theta = torch.mm(embbedings, kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
        phi = cos_theta - self.m
        label = label.view(-1, 1)  # size=(B,1)
        index = cos_theta.data * 0.0  # size=(B,Classnum)
        index.scatter_(1, label.data.view(-1, 1), 1)
        index = index.byte()
        output = cos_theta * 1.0
        output[index] = phi[index]  # only change the correct predicted output
        output *= self.s  # scale up in order to make softmax work, first introduced in normface
        return output


class ArcLinear(torch.nn.Module):
    """Additive Angular Margin linear module (ArcFace)

    Parameters
    ----------
    nfeat : int
        Embedding dimension
    nclass : int
        Number of classes
    margin : float
        Angular margin to penalize distances between embeddings and centers
    s : float
        Scaling factor for the logits
    """

    def __init__(self, nfeat, nclass, margin, s):
        super(ArcLinear, self).__init__()
        eps = 1e-4
        self.min_cos = eps - 1
        self.max_cos = 1 - eps
        self.nclass = nclass
        self.margin = margin
        self.s = s
        self.W = torch.nn.Parameter(torch.Tensor(nclass, nfeat))
        torch.nn.init.xavier_uniform_(self.W)

    def forward(self, x, target=None):
        """Apply the angular margin transformation

        Parameters
        ----------
        x : `torch.Tensor`
            an embedding batch
        target : `torch.Tensor`
            a non one-hot label batch

        Returns
        -------
        fX : `torch.Tensor`
            logits after the angular margin transformation
        """
        # the feature vectors has been normalized before calling this layer
        #xnorm = torch.nn.functional.normalize(x)
        xnorm = x
        # normalize W
        Wnorm = torch.nn.functional.normalize(self.W)
        target = target.long().view(-1, 1)
        # calculate cosθj (the logits)
        cos_theta_j = torch.matmul(xnorm, torch.transpose(Wnorm, 0, 1))
        # get the cosθ corresponding to the classes
        cos_theta_yi = cos_theta_j.gather(1, target)
        # for numerical stability
        cos_theta_yi = cos_theta_yi.clamp(min=self.min_cos, max=self.max_cos)
        # get the angle separating xi and Wyi
        theta_yi = torch.acos(cos_theta_yi)
        # apply the margin to the angle
        cos_theta_yi_margin = torch.cos(theta_yi + self.margin)
        # one hot encode  y
        one_hot = torch.zeros_like(cos_theta_j)
        one_hot.scatter_(1, target, 1.0)
        # project margin differences into cosθj
        return self.s * (cos_theta_j + one_hot * (cos_theta_yi_margin - cos_theta_yi))


class ArcMarginProduct(torch.nn.Module):
    """
    Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
    """

    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        torch.nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

    def change_params(self, s=None, m=None):
        """

        :param s:
        :param m:
        """
        if s is None:
            s = self.s
        if m is None:
            m = self.m
        self.s = s
        self.m = m
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

    def forward(self, input, target=None):
        """

        :param input:
        :param target:
        :return:
        """
        # cos(theta)
        cosine = torch.nn.functional.linear(torch.nn.functional.normalize(input),
                                        torch.nn.functional.normalize(self.weight))
        if target == None:
            return cosine * self.s
        # cos(theta + m)
        sine = torch.sqrt((1.0 - torch.mul(cosine, cosine)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)

        #one_hot = torch.zeros(cosine.size(), device='cuda' if torch.cuda.is_available() else 'cpu')
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, target.view(-1, 1), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.s

        return output, cosine * self.s


class SoftmaxAngularProto(torch.nn.Module):
    """

    from https://github.com/clovaai/voxceleb_trainer/blob/3bfd557fab5a3e6cd59d717f5029b3a20d22a281/loss/angleproto.py
    """
    def __init__(self, spk_count, emb_dim=256, init_w=10.0, init_b=-5.0, **kwargs):
        super(SoftmaxAngularProto, self).__init__()

        self.test_normalize = True

        self.w = torch.nn.Parameter(torch.tensor(init_w))
        self.b = torch.nn.Parameter(torch.tensor(init_b))
        self.criterion  = torch.nn.CrossEntropyLoss()

        self.cce_backend = torch.nn.Sequential(OrderedDict([
                    ("linear8", torch.nn.Linear(emb_dim, spk_count))
                ]))

    def forward(self, x, target=None):
        """

        :param x:
        :param target:
        :return:
        """
        assert x.size()[1] >= 2

        cce_prediction = self.cce_backend(x)

        if target is None:
            return cce_prediction

        x = x.reshape(-1, 2, x.size()[-1]).squeeze(1)

        out_anchor = torch.mean(x[:, 1:, :], 1)
        out_positive = x[:,0,:]

        cos_sim_matrix = torch.nn.functional.cosine_similarity(out_positive.unsqueeze(-1),
                                                               out_anchor.unsqueeze(-1).transpose(0, 2))
        torch.clamp(self.w, 1e-6)
        cos_sim_matrix = cos_sim_matrix * self.w + self.b
        loss = self.criterion(cos_sim_matrix, torch.arange(0,
                                                           cos_sim_matrix.shape[0],
                                                           device=x.device)) + self.criterion(cce_prediction, target)
        return loss, cce_prediction


class AngularProximityMagnet(torch.nn.Module):
    """
    from https://github.com/clovaai/voxceleb_trainer/blob/3bfd557fab5a3e6cd59d717f5029b3a20d22a281/loss/angleproto.py
    """
    def __init__(self, spk_count, emb_dim=256, batch_size=512, init_w=10.0, init_b=-5.0, **kwargs):
        super(AngularProximityMagnet, self).__init__()

        self.test_normalize = True

        self.w = torch.nn.Parameter(torch.tensor(init_w))
        self.b1 = torch.nn.Parameter(torch.tensor(init_b))
        self.b2 = torch.nn.Parameter(torch.tensor(+5.54))

        #last_linear = torch.nn.Linear(512, 1)
        #last_linear.bias.data += 1

        #self.magnitude = torch.nn.Sequential(OrderedDict([
        #            ("linear9", torch.nn.Linear(emb_dim, 512)),
        #            ("relu9", torch.nn.ReLU()),
        #            ("linear10", torch.nn.Linear(512, 512)),
        #            ("relu10", torch.nn.ReLU()),
        #            ("linear11", last_linear),
        #            ("relu11", torch.nn.ReLU())
        #        ]))

        self.cce_backend = torch.nn.Sequential(OrderedDict([
                    ("linear8", torch.nn.Linear(emb_dim, spk_count))
                ]))

        self.criterion  = torch.nn.CrossEntropyLoss()
        self.magnet_criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')

    def forward(self, x, target=None):
        """

        :param x:
        :param target:
        :return:
        """
        assert x.size()[1] >= 2

        cce_prediction = self.cce_backend(x)

        if target is None:
            return x, cce_prediction

        x = x.reshape(-1, 2, x.size()[-1]).squeeze(1)
        out_anchor = torch.mean(x[:, 1:, :], 1)
        out_positive = x[:, 0, :]

        ap_sim_matrix  = torch.nn.functional.cosine_similarity(out_positive.unsqueeze(-1),out_anchor.unsqueeze(-1).transpose(0,2))
        torch.clamp(self.w, 1e-6)
        ap_sim_matrix = ap_sim_matrix * self.w + self.b1

        labels = torch.arange(0, int(out_positive.shape[0]), device=torch.device("cuda:0")).unsqueeze(1)
        cos_sim_matrix  = torch.mm(out_positive, out_anchor.T)
        cos_sim_matrix = cos_sim_matrix + self.b2
        cos_sim_matrix = cos_sim_matrix + numpy.log(1/out_positive.shape[0] / (1 - 1/out_positive.shape[0]))
        mask = (torch.tile(labels, (1, labels.shape[0])) == labels.T).float()
        batch_loss = self.criterion(ap_sim_matrix, torch.arange(0, int(out_positive.shape[0]), device=torch.device("cuda:0"))) \
            + self.magnet_criterion(cos_sim_matrix.flatten().unsqueeze(1), mask.flatten().unsqueeze(1))
        return batch_loss, cce_prediction


class CircleMargin(torch.nn.Module):
    """

    """
    def __init__(self, in_features, out_features, s=256, m=0.25) -> None:
        super(CircleMargin, self).__init__()
        self.margin = m
        self.gamma = s
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, x, target=None):
        """

        :param x:
        :param target:
        :return:
        """
        cosine = torch.nn.functional.linear(torch.nn.functional.normalize(x),
                                            torch.nn.functional.normalize(self.weight))

        if target is None:
            return cosine * self.gamma

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, target.view(-1, 1), 1)

        output = (one_hot * (self.margin ** 2 - (1 - cosine) ** 2)) +\
                 ((1.0 - one_hot) * (cosine ** 2 - self.margin ** 2))
        output = output * self.gamma

        return output, cosine * self.gamma

