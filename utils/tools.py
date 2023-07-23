import os
import json
import yaml

import torch
import torch.nn.functional as F
from torch.cuda import amp
import numpy as np
import librosa
import struct
import matplotlib
matplotlib.use("Agg")
from scipy.io import wavfile
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
import webrtcvad
from typing import Optional, Union
from pathlib import Path
from torch.autograd import Variable


from utils.pitch_tools import denorm_f0, expand_f0_ph, cwt2f0
from scipy.ndimage.morphology import binary_dilation

int16_max = (2 ** 15) - 1

def get_configs_of(dataset):
    config_dir = os.path.join("./config", dataset)
    preprocess_config = yaml.load(open(
        os.path.join(config_dir, "preprocess.yaml"), "r"), Loader=yaml.FullLoader)
    model_config = yaml.load(open(
        os.path.join(config_dir, "model.yaml"), "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(
        os.path.join(config_dir, "train.yaml"), "r"), Loader=yaml.FullLoader)
    return preprocess_config, model_config, train_config

def preprocess_wav(fpath_or_wav: Union[str, Path, np.ndarray],
                   source_sr: Optional[int] = None,
                   normalize: Optional[bool] = True,
                   trim_silence: Optional[bool] = True):
    """
    Applies the preprocessing operations used in training the Speaker Encoder to a waveform 
    either on disk or in memory. The waveform will be resampled to match the data hyperparameters.

    :param fpath_or_wav: either a filepath to an audio file (many extensions are supported, not 
    just .wav), either the waveform as a numpy array of floats.
    :param source_sr: if passing an audio waveform, the sampling rate of the waveform before 
    preprocessing. After preprocessing, the waveform's sampling rate will match the data 
    hyperparameters. If passing a filepath, the sampling rate will be automatically detected and 
    this argument will be ignored.
    """
    audio_norm_target_dBFS = -30
    sampling_rate = 16000
    
    # Load the wav from disk if needed
    if isinstance(fpath_or_wav, str) or isinstance(fpath_or_wav, Path):
        wav, source_sr = librosa.load(str(fpath_or_wav), sr=None)
    else:
        wav = fpath_or_wav
    
    # Resample the wav if needed
    if source_sr is not None and source_sr != sampling_rate:
        wav = librosa.resample(wav, source_sr, sampling_rate)
        
    # Apply the preprocessing: normalize volume and shorten long silences 
    if normalize:
        wav = normalize_volume(wav, audio_norm_target_dBFS, increase_only=True)
    if webrtcvad and trim_silence:
        wav = trim_long_silences(wav)
    
    return wav

def trim_long_silences(wav):
    """
    Ensures that segments without voice in the waveform remain no longer than a 
    threshold determined by the VAD parameters in params.py.

    :param wav: the raw waveform as a numpy array of floats 
    :return: the same waveform with silences trimmed away (length <= original wav length)
    """
    sampling_rate = 16000
    ## Voice Activation Detection
    # # Window size of the VAD. Must be either 10, 20 or 30 milliseconds.
    # # This sets the granularity of the VAD. Should not need to be changed.
    vad_window_length = 30  # In milliseconds
    # Number of frames to average together when performing the moving average smoothing.
    # # The larger this value, the larger the VAD variations must be to not get smoothed out. 
    vad_moving_average_width = 8
    # Maximum number of consecutive silent frames a segment can have.
    vad_max_silence_length = 6
    # Compute the voice detection window size
    samples_per_window = (vad_window_length * sampling_rate) // 1000
    
    # Trim the end of the audio to have a multiple of the window size
    wav = wav[:len(wav) - (len(wav) % samples_per_window)]
    
    # Convert the float waveform to 16-bit mono PCM
    pcm_wave = struct.pack("%dh" % len(wav), *(np.round(wav * int16_max)).astype(np.int16))
    
    # Perform voice activation detection
    voice_flags = []
    vad = webrtcvad.Vad(mode=3)
    for window_start in range(0, len(wav), samples_per_window):
        window_end = window_start + samples_per_window
        voice_flags.append(vad.is_speech(pcm_wave[window_start * 2:window_end * 2],
                                         sample_rate=sampling_rate))
    voice_flags = np.array(voice_flags)
    
    # Smooth the voice detection with a moving average
    def moving_average(array, width):
        array_padded = np.concatenate((np.zeros((width - 1) // 2), array, np.zeros(width // 2)))
        ret = np.cumsum(array_padded, dtype=float)
        ret[width:] = ret[width:] - ret[:-width]
        return ret[width - 1:] / width
    
    audio_mask = moving_average(voice_flags, vad_moving_average_width)
    audio_mask = np.round(audio_mask).astype(np.bool)
    
    # Dilate the voiced regions
    audio_mask = binary_dilation(audio_mask, np.ones(vad_max_silence_length + 1))
    audio_mask = np.repeat(audio_mask, samples_per_window)
    
    return wav[audio_mask == True]


def normalize_volume(wav, target_dBFS, increase_only=False, decrease_only=False):
    if increase_only and decrease_only:
        raise ValueError("Both increase only and decrease only are set")
    dBFS_change = target_dBFS - 10 * np.log10(np.mean(wav ** 2))
    if (dBFS_change < 0 and increase_only) or (dBFS_change > 0 and decrease_only):
        return wav
    return wav * (10 ** (dBFS_change / 20))

def get_variance_level(preprocess_config, model_config, data_loading=True):
    """
    Consider the fact that there is no pre-extracted phoneme-level variance features in unsupervised duration modeling.
    Outputs:
        energy_level_tag: ["frame", "phone"]
            If data_loading is set True, then it will only be the "frame" for unsupervised duration modeling. 
            Otherwise, it will be aligned with the feature_level in config.
        energy_feature_level: ["frame_level", "phoneme_level"]
            The feature_level in config where the model will learn each variance in this level regardless of the input level.
    """
    learn_alignment = model_config["duration_modeling"]["learn_alignment"] if data_loading else False
    energy_feature_level = preprocess_config["preprocessing"]["energy"]["feature"]
    assert energy_feature_level in ["frame_level", "phoneme_level"]
    energy_level_tag = "phone" if (not learn_alignment and energy_feature_level == "phoneme_level") else "frame"
    return energy_level_tag, energy_feature_level


def get_phoneme_level_pitch(phone, mel2ph, pitch):
    pitch_phlevel_sum = torch.zeros(phone.shape[:-1]).float().scatter_add(
        0, torch.from_numpy(mel2ph).long() - 1, torch.from_numpy(pitch).float())
    pitch_phlevel_num = torch.zeros(phone.shape[:-1]).float().scatter_add(
        0, torch.from_numpy(mel2ph).long() - 1, torch.ones(pitch.shape)).clamp_min(1)
    pitch = (pitch_phlevel_sum / pitch_phlevel_num).numpy()
    return pitch


def get_phoneme_level_energy(duration, energy):
    # Phoneme-level average
    pos = 0
    for i, d in enumerate(duration):
        if d > 0:
            energy[i] = np.mean(energy[pos : pos + d])
        else:
            energy[i] = 0
        pos += d
    energy = energy[: len(duration)]
    return energy


def to_device(data, device):
    if len(data) == 21:
        (
            ids,
            raw_texts,
            speakers,
            texts,
            src_lens,
            max_src_len,
            mels,
            mel_lens,
            max_mel_len,
            pitches,
            f0s,
            uvs,
            cwt_specs,
            f0_means,
            f0_stds,
            energies,
            durations,
            mel2phs,
            attn_priors,
            spker_embeds,
            accents,
        ) = data

        speakers = torch.from_numpy(speakers).long().to(device)
        accents = torch.LongTensor(accents).to(device)        
        texts = torch.from_numpy(texts).long().to(device)
        src_lens = torch.from_numpy(src_lens).to(device)
        # raw_wav = torch.from_numpy(raw_wav).to(device)
        mels = torch.from_numpy(mels).float().to(device)
        mel_lens = torch.from_numpy(mel_lens).to(device)
        pitches = torch.from_numpy(pitches).long().to(device)
        f0s = torch.from_numpy(f0s).float().to(device)
        uvs = torch.from_numpy(uvs).float().to(device)
        cwt_specs = torch.from_numpy(cwt_specs).float().to(device) if cwt_specs is not None else cwt_specs
        f0_means = torch.from_numpy(f0_means).float().to(device) if f0_means is not None else f0_means
        f0_stds = torch.from_numpy(f0_stds).float().to(device) if f0_stds is not None else f0_stds
        energies = torch.from_numpy(energies).to(device)
        durations = torch.from_numpy(durations).long().to(device) if durations is not None else durations
        mel2phs = torch.from_numpy(mel2phs).long().to(device) if mel2phs is not None else mel2phs
        attn_priors = torch.from_numpy(attn_priors).float().to(device) if attn_priors is not None else attn_priors
        spker_embeds = torch.from_numpy(spker_embeds).float().to(device) if spker_embeds is not None else spker_embeds

        pitch_data = {
            "pitch": pitches,
            "f0": f0s,
            "uv": uvs,
            "cwt_spec": cwt_specs,
            "f0_mean": f0_means,
            "f0_std": f0_stds,
            "mel2ph": mel2phs,
        }

        return [
            ids,
            raw_texts,
            speakers,
            texts,
            src_lens,
            max_src_len,
            mels,
            mel_lens,
            max_mel_len,
            pitch_data,
            energies,
            durations,
            attn_priors,
            spker_embeds,
            accents,
        ]

    if len(data) == 8:
        (ids, raw_texts, speakers, texts, src_lens, max_src_len, spker_embeds, accents) = data

        speakers = torch.from_numpy(speakers).long().to(device)
        accents = torch.LongTensor(accents).to(device)
        texts = torch.from_numpy(texts).long().to(device)
        src_lens = torch.from_numpy(src_lens).to(device)
        if spker_embeds is not None:
            spker_embeds = torch.from_numpy(spker_embeds).float().to(device)
        # hidden_states = torch.from_numpy(hidden_states).to(device)

        return (ids, raw_texts, speakers, texts, src_lens, max_src_len, spker_embeds,accents)


def log(
    logger, step=None, losses=None, lr=None, fig=None, figs=None, img=None, audio=None, sampling_rate=22050, tag=""
):
    if losses is not None:
        logger.add_scalar("Loss/total_loss", losses[0], step)
        logger.add_scalar("Loss/mel_loss", losses[1], step)
        logger.add_scalar("Loss/mel_postnet_loss", losses[2], step)
        for k, v in losses[3].items():
            logger.add_scalar("Loss/{}_loss".format(k), v, step)
        logger.add_scalar("Loss/energy_loss", losses[4], step)
        for k, v in losses[5].items():
            logger.add_scalar("Loss/{}_loss".format(k), v, step)
        logger.add_scalar("Loss/ctc_loss", losses[6], step)
        logger.add_scalar("Loss/bin_loss", losses[7], step)
        logger.add_scalar("Loss/prosody_loss", losses[8], step)

    if lr is not None:
        logger.add_scalar("Training/learning_rate", lr, step)

    if fig is not None:
        logger.add_figure(tag, fig, step)

    if figs is not None:
        for k, v in figs.items():
            logger.add_figure("{}/{}".format(tag, k), v, step)

    if img is not None:
        logger.add_image(tag, img, step, dataformats='HWC')

    if audio is not None:
        logger.add_audio(
            tag,
            audio / max(abs(audio)),
            step,
            sample_rate=sampling_rate,
        )


def get_mask_from_lengths(lengths, max_len=None):
    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = torch.max(lengths).item()

    ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1).to(lengths.device)
    mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)

    return mask


def expand(values, durations):
    out = list()
    for value, d in zip(values, durations):
        out += [value] * max(0, int(d))
    return np.array(out)


def synth_one_sample(targets, predictions, vocoder, model_config, preprocess_config):

    pitch_config = preprocess_config["preprocessing"]["pitch"]
    pitch_type = pitch_config["pitch_type"]
    use_pitch_embed = model_config["variance_embedding"]["use_pitch_embed"]
    use_energy_embed = model_config["variance_embedding"]["use_energy_embed"]
    learn_alignment = model_config["duration_modeling"]["learn_alignment"]
    pitch_level_tag, energy_level_tag, *_ = get_variance_level(preprocess_config, model_config)
    basename = targets[0][0]
    src_len = predictions[9][0].item()
    mel_len = predictions[10][0].item()
    mel_target = targets[6][0, :mel_len].float().detach().transpose(0, 1)
    mel_prediction = predictions[2][0, :mel_len].float().detach().transpose(0, 1)
    duration = predictions[6][0, :src_len].int().detach().cpu().numpy()

    fig_attn = None
    if learn_alignment:
        attn_prior, attn_soft, attn_hard, attn_hard_dur, attn_logprob = targets[12], *predictions[11]
        attn_prior = attn_prior[0, :src_len, :mel_len].squeeze().detach().cpu().numpy() # text_len x mel_len
        attn_soft = attn_soft[0, 0, :mel_len, :src_len].detach().cpu().transpose(0, 1).numpy() # text_len x mel_len
        attn_hard = attn_hard[0, 0, :mel_len, :src_len].detach().cpu().transpose(0, 1).numpy() # text_len x mel_len
        fig_attn = plot_alignment(
            [
                attn_soft,
                attn_hard,
                attn_prior,
            ],
            ["Soft Attention", "Hard Attention", "Prior"]
        )

    phoneme_prosody_attn = None
    if model_config["prosody_modeling"]["model_type"] == "liu2021":
        if predictions[12][-1] is not None:
            phoneme_prosody_attn = predictions[12][-1][0][:src_len, :mel_len].detach()

    figs = {}
    if use_pitch_embed:
        pitch_prediction, pitch_target = predictions[3], targets[9]
        f0 = pitch_target["f0"]
        if pitch_type == "ph":
            mel2ph = pitch_target["mel2ph"]
            f0 = expand_f0_ph(f0, mel2ph, pitch_config)
            f0_pred = expand_f0_ph(pitch_prediction["pitch_pred"][:, :, 0], mel2ph, pitch_config)
            figs["f0"] = f0_to_figure(f0[0, :mel_len], None, f0_pred[0, :mel_len])
        else:
            f0 = denorm_f0(f0, pitch_target["uv"], pitch_config)
            if pitch_type == "cwt":
                # cwt
                cwt_out = pitch_prediction["cwt"]
                cwt_spec = cwt_out[:, :, :10]
                cwt = torch.cat([cwt_spec, pitch_target["cwt_spec"]], -1)
                figs["cwt"] = spec_to_figure(cwt[0, :mel_len])
                # f0
                f0_pred = cwt2f0(cwt_spec, pitch_prediction["f0_mean"], pitch_prediction["f0_std"], pitch_config["cwt_scales"])
                if pitch_config["use_uv"]:
                    assert cwt_out.shape[-1] == 11
                    uv_pred = cwt_out[:, :, -1] > 0
                    f0_pred[uv_pred > 0] = 0
                f0_cwt = denorm_f0(pitch_target["f0_cwt"], pitch_target["uv"], pitch_config)
                figs["f0"] = f0_to_figure(f0[0, :mel_len], f0_cwt[0, :mel_len], f0_pred[0, :mel_len])
            elif pitch_type == "frame":
                # f0
                uv_pred = pitch_prediction["pitch_pred"][:, :, 1] > 0
                pitch_pred = denorm_f0(pitch_prediction["pitch_pred"][:, :, 0], uv_pred, pitch_config)
                figs["f0"] = f0_to_figure(f0[0, :mel_len], None, pitch_pred[0, :mel_len])
    if use_energy_embed:
        if preprocess_config["preprocessing"]["energy"]["feature"] == "phoneme_level":
            energy_prediction = predictions[4][0, :src_len].detach().cpu().numpy()
            energy_prediction = expand(energy_prediction, duration)
            energy_target = targets[10][0, :src_len].detach().cpu().numpy()
            energy_target = expand(energy_target, duration)
        else:
            energy_prediction = predictions[4][0, :mel_len].detach().cpu().numpy()
            energy_target = targets[10][0, :mel_len].detach().cpu().numpy()
        figs["energy"] = energy_to_figure(energy_target, energy_prediction)

    figs["mel"] = plot_mel(
        [
            mel_prediction.cpu().numpy(),
            mel_target.cpu().numpy(),
            phoneme_prosody_attn.cpu().numpy(),
        ] if phoneme_prosody_attn is not None else [
            mel_prediction.cpu().numpy(),
            mel_target.cpu().numpy(),
        ],
        ["Synthetized Spectrogram", "Ground-Truth Spectrogram", "Prosody Alignment"],
        n_attn=1 if phoneme_prosody_attn is not None else 0,
    )

    if vocoder is not None:
        from .model import vocoder_infer

        wav_reconstruction = vocoder_infer(
            mel_target.unsqueeze(0),
            vocoder,
            model_config,
            preprocess_config,
        )[0]
        wav_prediction = vocoder_infer(
            mel_prediction.unsqueeze(0),
            vocoder,
            model_config,
            preprocess_config,
        )[0]
    else:
        wav_reconstruction = wav_prediction = None

    return figs, fig_attn, wav_reconstruction, wav_prediction, basename

def evaluation_synth_fc(targets, targets2, predictions, predictions2, vocoder, model_config, preprocess_config):
    pitch_config = preprocess_config["preprocessing"]["pitch"]
    pitch_type = pitch_config["pitch_type"]
    use_pitch_embed = model_config["variance_embedding"]["use_pitch_embed"]
    use_energy_embed = model_config["variance_embedding"]["use_energy_embed"]
    learn_alignment = model_config["duration_modeling"]["learn_alignment"]
    pitch_level_tag, energy_level_tag, *_ = get_variance_level(preprocess_config, model_config)
    src_len = predictions[9][0].item()
    src_len2 = predictions2[9][0].item()
    mel_len = predictions[10][0].item()
    mel_len2 = predictions2[10][0].item()
    mel_prediction = predictions[2][0, :mel_len].float().detach().transpose(0, 1)
    mel_prediction2 = predictions2[2][0, :mel_len2].float().detach().transpose(0, 1)
    duration = predictions[6][0, :src_len].int().detach().cpu().numpy()
    duration2 = predictions2[6][0, :src_len].int().detach().cpu().numpy()

    phoneme_prosody_attn = None

    figs = {}
    figs["mel_inference"] = plot_mel(
        [
            mel_prediction.cpu().numpy(),
            mel_prediction2.cpu().numpy(),
            phoneme_prosody_attn.cpu().numpy(),
        ] if phoneme_prosody_attn is not None else [
            mel_prediction.cpu().numpy(),
            mel_prediction2.cpu().numpy(),
        ],
        ["Inference1", "Inference2", "Prosody Alignment"],
        n_attn=1 if phoneme_prosody_attn is not None else 0,
    )

    if vocoder is not None:
        from .model import vocoder_infer

        wav1 = vocoder_infer(
            mel_prediction.unsqueeze(0),
            vocoder,
            model_config,
            preprocess_config,
        )[0]
        wav2 = vocoder_infer(
            mel_prediction2.unsqueeze(0),
            vocoder,
            model_config,
            preprocess_config,
        )[0]
    else:
        wav1 = wav2 = None

    return figs, wav1, wav2



def synth_samples(targets, predictions, vocoder, model_config, preprocess_config, path, args):

    multi_speaker = model_config["multi_speaker"]
    learn_alignment = model_config["duration_modeling"]["learn_alignment"]
    pitch_level_tag, energy_level_tag, *_ = get_variance_level(preprocess_config, model_config)
    basenames = targets[0]
    for i in range(len(predictions[0])):
        basename = basenames[i]
        src_len = predictions[9][i].item()
        mel_len = predictions[10][i].item()
        mel_prediction = predictions[2][i, :mel_len].detach().transpose(0, 1)
        duration = predictions[6][i, :src_len].int().detach().cpu().numpy()
        attn_soft = attn_hard = None

        fig_save_dir = os.path.join(
            path, str(args.restore_step), "{}_{}.png".format(basename, args.speaker_id)\
                if multi_speaker and args.mode == "single" else "{}.png".format(basename))
        fig = plot_mel(
            [
                mel_prediction.cpu().numpy(),
            ],
            ["Synthetized Spectrogram"],
            save_dir=fig_save_dir,
        )
        plt.close()

    from .model import vocoder_infer

    mel_predictions = predictions[2].transpose(1, 2)
    lengths = predictions[10] * preprocess_config["preprocessing"]["stft"]["hop_length"]
    wav_predictions = vocoder_infer(
        mel_predictions, vocoder, model_config, preprocess_config, lengths=lengths
    )

    sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
    for wav, basename in zip(wav_predictions, basenames):
#        wav = preprocess_wav(wav)
        wavfile.write(os.path.join(
            path, str(args.restore_step), "{}_{}.wav".format(basename, args.speaker_id)\
                if multi_speaker and args.mode == "single" else "{}.wav".format(basename)),
            sampling_rate, wav)

def synth_samples_conversion(targets, predictions, vocoder, model_config, preprocess_config, path, args):

    multi_speaker = model_config["multi_speaker"]
    learn_alignment = model_config["duration_modeling"]["learn_alignment"]
    pitch_level_tag, energy_level_tag, *_ = get_variance_level(preprocess_config, model_config)
    basenames = targets[0]
    for i in range(len(predictions[0])):
        basename = basenames[i]
        src_len = predictions[9][i].item()
        mel_len = predictions[10][i].item()
        mel_prediction = predictions[2][i, :mel_len].detach().transpose(0, 1)
        duration = predictions[6][i, :src_len].int().detach().cpu().numpy()
        attn_soft = attn_hard = None

        # fig_save_dir = os.path.join(
        #     path, str(args.restore_step), "{}_{}.png".format(basename, args.speaker_id)\
        #         if multi_speaker and args.mode == "single" else "{}.png".format(basename))
        # fig = plot_mel(
        #     [
        #         mel_prediction.cpu().numpy(),
        #     ],
        #     ["Synthetized Spectrogram"],
        #     save_dir=fig_save_dir,
        # )
        # plt.close()

    from .model import vocoder_infer

    mel_predictions = predictions[2].transpose(1, 2)
    lengths = predictions[10] * preprocess_config["preprocessing"]["stft"]["hop_length"]
    wav_predictions = vocoder_infer(
        mel_predictions, vocoder, model_config, preprocess_config, lengths=lengths
    )

    sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
    for wav, basename in zip(wav_predictions, basenames):
#        wav = preprocess_wav(wav)
        wavfile.write(os.path.join(
            path, str(args.restore_step), "{}_{}_{}.wav".format(args.speaker_id, args.accent, basename, )\
                if multi_speaker and args.mode == "single" else "{}.wav".format(basename)),
            sampling_rate, wav)

def synth_samples_recoset(targets, predictions, vocoder, model_config, preprocess_config, path, args):

    multi_speaker = model_config["multi_speaker"]
    learn_alignment = model_config["duration_modeling"]["learn_alignment"]
    pitch_level_tag, energy_level_tag, *_ = get_variance_level(preprocess_config, model_config)
    basenames = targets[0]
    spk_lab = ["RRBI", "ABA", "SKA", "EBVS", "TNI", "NCC", "BWC", "HQTV", "TXHC", "ERMS", "PNV", "LXC", "HKK", "ASI", "THV", "MBMPS", "SVBI", "ZHAA", "HJK", "TLV", "NJS", "YBAA", "YDCK", "YKWK"]
    for i in range(len(predictions[0])):
        basename = basenames[i]
        src_len = predictions[9][i].item()
        mel_len = predictions[10][i].item()
        mel_prediction = predictions[2][i, :mel_len].detach().transpose(0, 1)
        duration = predictions[6][i, :src_len].int().detach().cpu().numpy()
        attn_soft = attn_hard = None
        
        # fig_save_dir = os.path.join(
        #     path, str(args.restore_step), "{}_{}.png".format(basename, args.speaker_id)\
        #         if multi_speaker and args.mode == "single" else "{}.png".format(basename))
        # fig = plot_mel(
        #     [
        #         mel_prediction.cpu().numpy(),
        #     ],
        #     ["Synthetized Spectrogram"],
        #     save_dir=fig_save_dir,
        # )
        # plt.close()

    from .model import vocoder_infer
    # filename=args.filename
    # filename=targets[0][0]
    # spk_name=spk_lab[targets[2]]
    mel_predictions = predictions[2].transpose(1, 2)
    lengths = predictions[10] * preprocess_config["preprocessing"]["stft"]["hop_length"]
    wav_predictions = vocoder_infer(
        mel_predictions, vocoder, model_config, preprocess_config, lengths=lengths
    )


    sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
    # for wav, basename in zip(wav_predictions, basenames):
    for wav, basename, spk_id in zip(wav_predictions, basenames, targets[2]):
        spk_name=spk_lab[spk_id]
        os.makedirs(os.path.join(path, str(args.restore_step), 'obj_eval', spk_name), exist_ok=True) #create dir with speaker name
#        wav = preprocess_wav(wav)
        # wavfile.write(os.path.join(
        #     path, str(args.restore_step), "{}_{}.wav".format(basename, args.speaker_id)\
        #         if multi_speaker and args.mode == "single" else "{}.wav".format(basename)),
        #     sampling_rate, wav)
        wavfile.write(os.path.join(
            path, str(args.restore_step), 'obj_eval', spk_name, "{}.wav".format(basename)\
                if multi_speaker and args.mode == "single" else "{}.wav".format(basename)),
            sampling_rate, wav)

def infer_one_sample_recoset(targets, predictions, vocoder, mel_stats, model_config, preprocess_config, path, args):

    normalize = preprocess_config["preprocessing"]["mel"]["normalize"]
    n_frames_per_step = model_config["decoder"]["n_frames_per_step"]
    multi_speaker = model_config["multi_speaker"]
    mel_predictions = predictions[1]
    if normalize:
        mel_predictions = Audio.tools.mel_denormalize(predictions[1], *mel_stats)

    basename = targets[0][0]
    src_len = targets[5][0].item()
    mel_len = mel_predictions[0].shape[0]
    reduced_mel_len = mel_len // n_frames_per_step
    mel_prediction = mel_predictions[0].detach().transpose(0, 1)
    attention = predictions[3][0, :reduced_mel_len, :src_len].detach().transpose(0, 1) # [seq_len, mel_len]
    filename=args.filename

    from .model import vocoder_infer

    lengths = mel_len * preprocess_config["preprocessing"]["stft"]["hop_length"]
    wav_predictions = vocoder_infer(
        mel_predictions.transpose(1, 2), vocoder, model_config, preprocess_config, lengths=[lengths]
    )
    os.makedirs(os.path.join(path, str(args.restore_step), args.speaker_id), exist_ok=True) #create dir with speaker name

    sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
    wavfile.write(os.path.join(
        path, str(args.restore_step), args.speaker_id, "{}.wav".format(filename)\
             if multi_speaker and (args.mode == "sample_stats") else "{}.wav".format(basename)),
        sampling_rate, wav_predictions[0])


def plot_mel(data, titles, n_attn=0, save_dir=None):
    fig, axes = plt.subplots(len(data), 1, squeeze=False)
    if titles is None:
        titles = [None for i in range(len(data))]

    if n_attn > 0:
        # Plot Mel Spectrogram
        plot_mel_(fig, axes, data[:-n_attn], titles)

        # Plot Alignment
        xlim = data[0].shape[1]
        for i in range(-n_attn, 0):
            im = axes[i][0].imshow(data[i], origin='lower', aspect='auto')
            axes[i][0].set_xlabel('Decoder timestep')
            axes[i][0].set_ylabel('Encoder timestep')
            axes[i][0].set_xlim(0, xlim)
            axes[i][0].set_title(titles[i], fontsize="medium")
            axes[i][0].tick_params(labelsize="x-small")
            axes[i][0].set_anchor("W")
            fig.colorbar(im, ax=axes[i][0])
    else:
        # Plot Mel Spectrogram
        plot_mel_(fig, axes, data, titles)

    fig.canvas.draw()
    # data = save_figure_to_numpy(fig)
    if save_dir is not None:
        plt.savefig(save_dir)
    # plt.close()
    return fig #, data


def plot_mel_(fig, axes, data, titles, tight_layout=True):
    if tight_layout:
        fig.tight_layout()

    for i in range(len(data)):
        mel = data[i]
        if isinstance(mel, torch.Tensor):
            mel = mel.detach().cpu().numpy()
        axes[i][0].imshow(mel, origin="lower")
        axes[i][0].set_aspect(2.5, adjustable="box")
        axes[i][0].set_ylim(0, mel.shape[0])
        axes[i][0].set_title(titles[i], fontsize="medium")
        axes[i][0].tick_params(labelsize="x-small", left=False, labelleft=False)
        axes[i][0].set_anchor("W")


def spec_to_figure(spec, vmin=None, vmax=None, filename=None):
    if isinstance(spec, torch.Tensor):
        spec = spec.detach().cpu().numpy()
    fig = plt.figure(figsize=(12, 6))
    plt.pcolor(spec.T, vmin=vmin, vmax=vmax)
    if filename is not None:
        plt.savefig(filename)
    return fig


def spec_f0_to_figure(spec, f0s, figsize=None, line_colors=['w', 'r', 'y', 'cyan', 'm', 'b', 'lime'], filename=None):
    max_y = spec.shape[1]
    if isinstance(spec, torch.Tensor):
        spec = spec.detach().cpu().numpy()
        f0s = {k: f0.detach().cpu().numpy() for k, f0 in f0s.items()}
    f0s = {k: f0 / 10 for k, f0 in f0s.items()}
    fig = plt.figure(figsize=(12, 6) if figsize is None else figsize)
    plt.pcolor(spec.T)
    for i, (k, f0) in enumerate(f0s.items()):
        plt.plot(f0.clip(0, max_y), label=k, c=line_colors[i], linewidth=1, alpha=0.8)
    plt.legend()
    if filename is not None:
        plt.savefig(filename)
    return fig


def f0_to_figure(f0_gt, f0_cwt=None, f0_pred=None):
    fig = plt.figure()
    if isinstance(f0_gt, torch.Tensor):
        f0_gt = f0_gt.detach().cpu().numpy()
    plt.plot(f0_gt, color="r", label="gt")
    if f0_cwt is not None:
        if isinstance(f0_cwt, torch.Tensor):
            f0_cwt = f0_cwt.detach().cpu().numpy()
        plt.plot(f0_cwt, color="b", label="cwt")
    if f0_pred is not None:
        if isinstance(f0_pred, torch.Tensor):
            f0_pred = f0_pred.detach().cpu().numpy()
        plt.plot(f0_pred, color="green", label="pred")
    plt.legend()
    return fig


def energy_to_figure(energy_gt, energy_pred=None):
    fig = plt.figure()
    if isinstance(energy_gt, torch.Tensor):
        energy_gt = energy_gt.detach().cpu().numpy()
    plt.plot(energy_gt, color="r", label="gt")
    if energy_pred is not None:
        if isinstance(energy_pred, torch.Tensor):
            energy_pred = energy_pred.detach().cpu().numpy()
        plt.plot(energy_pred, color="green", label="pred")
    plt.legend()
    return fig


# def plot_single_alignment(alignment, info=None, save_dir=None):
#     fig, ax = plt.subplots(figsize=(6, 4))
#     im = ax.imshow(alignment, aspect='auto', origin='lower', interpolation='none')
#     fig.colorbar(im, ax=ax)
#     xlabel = 'Decoder timestep'
#     if info is not None:
#         xlabel += '\n\n' + info
#     plt.xlabel(xlabel)
#     plt.ylabel('Encoder timestep')
#     plt.tight_layout()

#     fig.canvas.draw()
#     data = save_figure_to_numpy(fig)
#     if save_dir is not None:
#         plt.savefig(save_dir)
#     plt.close()
#     return data


def plot_alignment(data, titles=None, save_dir=None):
    fig, axes = plt.subplots(len(data), 1, figsize=[6,4],dpi=300)
    plt.subplots_adjust(top = 0.9, bottom = 0.1, right = 0.95, left = 0.05)
    if titles is None:
        titles = [None for i in range(len(data))]

    for i in range(len(data)):
        im = data[i]
        axes[i].imshow(im, origin='lower')
        axes[i].set_xlabel('Audio timestep')
        axes[i].set_ylabel('Text timestep')
        axes[i].set_ylim(0, im.shape[0])
        axes[i].set_xlim(0, im.shape[1])
        axes[i].set_title(titles[i], fontsize='medium')
        axes[i].tick_params(labelsize='x-small') 
        axes[i].set_anchor('W')
    plt.tight_layout()

    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    if save_dir is not None:
        plt.savefig(save_dir)
    plt.close()
    return data


def plot_embedding(out_dir, embedding, embedding_accent_id,colors,labels,filename='embedding.png'):
#    colors = 'r','b','g','y'
#    labels = 'Female','Male'

    data_x = embedding
    data_y = embedding_accent_id
#    data_y = np.array([gender_dict[spk_id] == 'M' for spk_id in embedding_speaker_id], dtype=np.int)
    tsne_model = TSNE(n_components=2, random_state=0, init='random', n_iter=350, n_iter_without_progress=100)
    tsne_all_data = tsne_model.fit_transform(data_x)
    tsne_all_y_data = data_y

    plt.figure(figsize=(10,10))
    for i, (c, label) in enumerate(zip(colors, labels)):
        plt.scatter(tsne_all_data[tsne_all_y_data==i,0], tsne_all_data[tsne_all_y_data==i,1], c=c, label=label, alpha=0.5)

    plt.grid(True)
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, filename))
    plt.close()


def save_figure_to_numpy(fig):
    # save it to a numpy array.
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data


def pad_1D(inputs, PAD=0):
    def pad_data(x, length, PAD):
        x_padded = np.pad(
            x, (0, length - x.shape[0]), mode="constant", constant_values=PAD
        )
        return x_padded

    max_len = max((len(x) for x in inputs))
    padded = np.stack([pad_data(x, max_len, PAD) for x in inputs])

    return padded


def pad_2D(inputs, maxlen=None):
    def pad(x, max_len):
        PAD = 0
        if np.shape(x)[0] > max_len:
            raise ValueError("not max_len")

        s = np.shape(x)[1]
        x_padded = np.pad(
            x, (0, max_len - np.shape(x)[0]), mode="constant", constant_values=PAD
        )
        return x_padded[:, :s]

    if maxlen:
        output = np.stack([pad(x, maxlen) for x in inputs])
    else:
        max_len = max(np.shape(x)[0] for x in inputs)
        output = np.stack([pad(x, max_len) for x in inputs])

    return output


def pad_3D(inputs, B, T, L):
    inputs_padded = np.zeros((B, T, L), dtype=np.float32)
    for i, input_ in enumerate(inputs):
        inputs_padded[i, :np.shape(input_)[0], :np.shape(input_)[1]] = input_
    return inputs_padded


def pad(input_ele, mel_max_length=None):
    if mel_max_length:
        max_len = mel_max_length
    else:
        max_len = max([input_ele[i].size(0) for i in range(len(input_ele))])

    out_list = list()
    for i, batch in enumerate(input_ele):
        if len(batch.shape) == 1:
            one_batch_padded = F.pad(
                batch, (0, max_len - batch.size(0)), "constant", 0.0
            )
        elif len(batch.shape) == 2:
            one_batch_padded = F.pad(
                batch, (0, 0, 0, max_len - batch.size(0)), "constant", 0.0
            )
        out_list.append(one_batch_padded)
    out_padded = torch.stack(out_list)
    return out_padded


def dur_to_mel2ph(dur, dur_padding=None, alpha=1.0):
    """
    Example (no batch dim version):
        1. dur = [2,2,3]
        2. token_idx = [[1],[2],[3]], dur_cumsum = [2,4,7], dur_cumsum_prev = [0,2,4]
        3. token_mask = [[1,1,0,0,0,0,0],
                            [0,0,1,1,0,0,0],
                            [0,0,0,0,1,1,1]]
        4. token_idx * token_mask = [[1,1,0,0,0,0,0],
                                        [0,0,2,2,0,0,0],
                                        [0,0,0,0,3,3,3]]
        5. (token_idx * token_mask).sum(0) = [1,1,2,2,3,3,3]

    :param dur: Batch of durations of each frame (B, T_txt)
    :param dur_padding: Batch of padding of each frame (B, T_txt)
    :param alpha: duration rescale coefficient
    :return:
        mel2ph (B, T_speech)
    """
    assert alpha > 0
    dur = torch.round(dur.float() * alpha).long()
    if dur_padding is not None:
        dur = dur * (1 - dur_padding.long())
    token_idx = torch.arange(1, dur.shape[1] + 1)[None, :, None].to(dur.device)
    dur_cumsum = torch.cumsum(dur, 1)
    dur_cumsum_prev = F.pad(dur_cumsum, [1, -1], mode="constant", value=0)

    pos_idx = torch.arange(dur.sum(-1).max())[None, None].to(dur.device)
    token_mask = (pos_idx >= dur_cumsum_prev[:, :, None]) & (pos_idx < dur_cumsum[:, :, None])
    mel2ph = (token_idx * token_mask.long()).sum(1)
    return mel2ph


def mel2ph_to_dur(mel2ph, T_txt, max_dur=None):
    B, _ = mel2ph.shape
    dur = mel2ph.new_zeros(B, T_txt + 1).scatter_add(1, mel2ph, torch.ones_like(mel2ph))
    dur = dur[:, 1:]
    if max_dur is not None:
        dur = dur.clamp(max=max_dur)
    return dur


def make_positions(tensor, padding_idx):
    """Replace non-padding symbols with their position numbers.

    Position numbers begin at padding_idx+1. Padding symbols are ignored.
    """
    # The series of casts and type-conversions here are carefully
    # balanced to both work with ONNX export and XLA. In particular XLA
    # prefers ints, cumsum defaults to output longs, and ONNX doesn"t know
    # how to handle the dtype kwarg in cumsum.
    mask = tensor.ne(padding_idx).int()
    return (
                   torch.cumsum(mask, dim=1).type_as(mask) * mask
           ).long() + padding_idx


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1)


def ssim(img1, img2, window_size=11, size_average=True):
        (_, channel, _, _) = img1.size()
        global window
        if window is None:
            window = create_window(window_size, channel)
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
        return _ssim(img1, img2, window, window_size, channel, size_average)
    
    
# Tools for MLVAE    

def accumulate_group_evidence(class_mu, class_logvar, labels_batch, is_cuda, device):
    """
    :param class_mu: mu values for class latent embeddings of each sample in the mini-batch
    :param class_logvar: logvar values for class latent embeddings for each sample in the mini-batch
    :param labels_batch: class labels of each sample (the operation of accumulating class evidence can also
        be performed using group labels instead of actual class labels)
    :param is_cuda:
    :return:
    """
    var_dict = {}
    mu_dict = {}

    # convert logvar to variance for calculations
    class_var = class_logvar.exp_()

    # calculate var inverse for each group using group vars
    for i in range(len(labels_batch)):
        group_label = labels_batch[i].item()

        # remove 0 values from variances
        class_var[i][class_var[i] == float(0)] = 1e-6

        if group_label in var_dict.keys():
            var_dict[group_label] += 1 / class_var[i]
        else:
            var_dict[group_label] = 1 / class_var[i]

    # invert var inverses to calculate mu and return value
    for group_label in var_dict.keys():
        var_dict[group_label] = 1 / var_dict[group_label]

    # calculate mu for each group
    for i in range(len(labels_batch)):
        group_label = labels_batch[i].item()

        if group_label in mu_dict.keys():
            mu_dict[group_label] += class_mu[i] * (1 / class_var[i])
        else:
            mu_dict[group_label] = class_mu[i] * (1 / class_var[i])

    # multiply group var with sums calculated above to get mu for the group
    for group_label in mu_dict.keys():
        mu_dict[group_label] *= var_dict[group_label]

    # replace individual mu and logvar values for each sample with group mu and logvar
    group_mu = torch.FloatTensor(class_mu.size(0), class_mu.size(1))
    group_var = torch.FloatTensor(class_var.size(0), class_var.size(1))

    if is_cuda:
        group_mu = group_mu.cuda(device)
        group_var = group_var.cuda(device)

    for i in range(len(labels_batch)):
        group_label = labels_batch[i].item()

        group_mu[i] = mu_dict[group_label]
        group_var[i] = var_dict[group_label]

        # remove 0 from var before taking log
        group_var[i][group_var[i] == float(0)] = 1e-6

    # convert group vars into logvars before returning
    return Variable(group_mu, requires_grad=True), Variable(torch.log(group_var), requires_grad=True)

def reparameterize(training, mu, logvar):
    if training:
        std = logvar.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        return eps.mul(std).add_(mu)
    else:
        return mu


def group_wise_reparameterize(training, mu, logvar, labels_batch, is_cuda,device):
    eps_dict = {}

    # generate only 1 eps value per group label
    for label in torch.unique(labels_batch):
        if is_cuda:
            # eps_dict[label.item()] = torch.cuda.FloatTensor(1, logvar.size(1)).normal_(0., 0.1)
            eps_dict[label.item()] = torch.FloatTensor(1, logvar.size(1)).normal_(0., 0.1).to(device)    
        else:
            eps_dict[label.item()] = torch.FloatTensor(1, logvar.size(1)).normal_(0., 0.1)

    if training:
        std = logvar.mul(0.5).exp_()
        reparameterized_var = Variable(std.data.new(std.size()))

        # multiply std by correct eps and add mu
        for i in range(logvar.size(0)):
            reparameterized_var[i] = std[i].mul(Variable(eps_dict[labels_batch[i].item()]))
            reparameterized_var[i].add_(mu[i])

        return reparameterized_var
    else:
        return mu
