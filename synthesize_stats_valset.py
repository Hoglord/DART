import re
import os
import json
import argparse
from string import punctuation

import torch
import yaml
import numpy as np
from torch.utils.data import DataLoader
from g2p_en import G2p
# from pypinyin import pinyin, Style

from utils.model import get_model, get_vocoder
from utils.tools import get_configs_of, to_device, synth_samples, synth_samples_recoset
from dataset import TextDataset
from text import text_to_sequence


def read_lexicon(lex_path):
    lexicon = {}
    with open(lex_path) as f:
        for line in f:
            temp = re.split(r"\s+", line.strip("\n"))
            word = temp[0]
            phones = temp[1:]
            if word.lower() not in lexicon:
                lexicon[word.lower()] = phones
    return lexicon


def preprocess_english(text, preprocess_config):
    text = text.rstrip(punctuation)
    lexicon = read_lexicon(preprocess_config["path"]["lexicon_path"])

    g2p = G2p()
    phones = []
    words = re.split(r"([,;.\-\?\!\s+])", text)
    for w in words:
        if w.lower() in lexicon:
            phones += lexicon[w.lower()]
        else:
            phones += list(filter(lambda p: p != " ", g2p(w)))
    phones = "{" + "}{".join(phones) + "}"
    phones = re.sub(r"\{[^\w\s]?\}", "{sp}", phones)
    phones = phones.replace("}{", " ")

    print("Raw Text Sequence: {}".format(text))
    print("Phoneme Sequence: {}".format(phones))
    sequence = np.array(
        text_to_sequence(
            phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
        )
    )

    return np.array(sequence)


# def preprocess_mandarin(text, preprocess_config):
#     lexicon = read_lexicon(preprocess_config["path"]["lexicon_path"])

#     phones = []
#     pinyins = [
#         p[0]
#         for p in pinyin(
#             text, style=Style.TONE3, strict=False, neutral_tone_with_five=True
#         )
#     ]
#     for p in pinyins:
#         if p in lexicon:
#             phones += lexicon[p]
#         else:
#             phones.append("sp")

#     phones = "{" + " ".join(phones) + "}"
#     print("Raw Text Sequence: {}".format(text))
#     print("Phoneme Sequence: {}".format(phones))
#     sequence = np.array(
#         text_to_sequence(
#             phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
#         )
#     )

#     return np.array(sequence)


def synthesize(device, model, args, configs, vocoder, batchs, control_values):
    preprocess_config, model_config, train_config = configs
    pitch_control, energy_control, duration_control = control_values

    for batch in batchs:
        batch = to_device(batch, device)
        with torch.no_grad():
            # Forward
            output = model(
                *(batch[2:-2]),
                spker_embeds=batch[-2],
                accents=batch[-1],
                p_control=pitch_control,
                e_control=energy_control,
                d_control=duration_control
            )
            synth_samples(
                batch,
                output,
                vocoder,
                model_config,
                preprocess_config,
                train_config["path"]["result_path"],
                args,
            )

def synthesize_stats(device, model, args, configs, vocoder, batchs, control_values):
    preprocess_config, model_config, train_config = configs
    pitch_control, energy_control, duration_control = control_values
    arraypath = train_config["path"]["array_path"]

    acc_mu=np.load(os.path.join(arraypath,'inf_acc_mu.npy'))
    acc_var=np.load(os.path.join(arraypath,'inf_acc_var.npy'))

    spk_mu=np.load(os.path.join(arraypath,'inf_spk_mu.npy'))
    spk_var=np.load(os.path.join(arraypath,'inf_spk_var.npy'))

    acc_id=np.load(os.path.join(arraypath,'inf_acc_id.npy'))
    spk_id=np.load(os.path.join(arraypath,'inf_spk_id.npy'))


    # speakers = np.array([speaker_map[args.speaker_id]]) if model_config["multi_speaker"] else np.array([0]) # single speaker is allocated 0
    spk_lab = ["RRBI", "ABA", "SKA", "EBVS", "TNI", "NCC", "BWC", "HQTV", "TXHC", "ERMS", "PNV", "LXC", "HKK", "ASI", "THV", "MBMPS", "SVBI", "ZHAA", "HJK", "TLV", "NJS", "YBAA", "YDCK", "YKWK"]
    i=0
    batch_size=args.batch_size
    for batch in batchs:
        batch = to_device(batch, device)
        with torch.no_grad():
            # Forward
            # output = model(
            #     *(batch[2:-2]),
            #     spker_embeds=batch[-2],
            #     accents=batch[-1],
            #     p_control=pitch_control,
            #     e_control=energy_control,
            #     d_control=duration_control
            # )

            # z_acc=np.mean(acc_mu[acc_id==batch[8][0].cpu().item()],axis=0)
            # z_spk=np.mean(spk_mu[spk_id==batch[2][0].cpu().item()],axis=0)

            if batch_size>1:
                z_acc=torch.zeros((batch_size,128))
                z_spk=torch.zeros((batch_size,128))
                for ss in range(batch_size):
                    z_acc[ss,:]=torch.from_numpy(np.mean(acc_mu[acc_id==batch[-1][ss].cpu().item()],axis=0))
                    z_spk[ss,:]=torch.from_numpy(np.mean(spk_mu[spk_id==batch[2][ss].cpu().item()],axis=0))

                # z_acc=np.mean(acc_mu[acc_id==batch[-1].cpu().item()],axis=0)
                # z_spk=np.mean(spk_mu[spk_id==batch[2].cpu().item()],axis=0)

                z_acc=z_acc.to(device)
                z_spk=z_spk.to(device)
            else:
                z_acc=np.mean(acc_mu[acc_id==batch[-1].cpu().item()],axis=0)
                z_spk=np.mean(spk_mu[spk_id==batch[2].cpu().item()],axis=0)

                z_acc=torch.from_numpy(z_acc).unsqueeze(0).to(device)
                z_spk=torch.from_numpy(z_spk).unsqueeze(0).to(device)

            output = model.inference_stats(
                *(batch[2:-2]),
                spker_embeds=batch[-2],
                accents=batch[-1],
                p_control=pitch_control,
                e_control=energy_control,
                d_control=duration_control,
                z_acc=z_acc,
                z_spk=z_spk,
            )
            synth_samples_recoset(
                batch,
                output,
                vocoder,
                model_config,
                preprocess_config,
                train_config["path"]["result_path"],
                args,
            )
            print(i)
            i+=1



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--mode",
    #     type=str,
    #     choices=["batch", "single"],
    #     required=True,
    #     help="Synthesize a whole dataset or a single sentence",
    # )
    # parser.add_argument(
    #     "--pitch_control",
    #     type=float,
    #     default=1.0,
    #     help="control the pitch of the whole utterance, larger value for higher pitch",
    # )
    # parser.add_argument(
    #     "--energy_control",
    #     type=float,
    #     default=1.0,
    #     help="control the energy of the whole utterance, larger value for larger volume",
    # )
    # parser.add_argument(
    #     "--duration_control",
    #     type=float,
    #     default=1.0,
    #     help="control the speed of the whole utterance, larger value for slower speaking rate",
    # )
    args = parser.parse_args()


    args.dataset='L2ARCTIC'
    args.pitch_control=1.0
    args.energy_control=1.0
    args.duration_control=1.0
    args.mode='batch'

    args.source='val_unsup.txt'
    # args.source=None
    # args.speaker_id='NCC'
    # args.basename='SVBI_a0009'
    args.speaker_id='SVBI'
    args.accent='Hindi'
    # args.accent2='Arabic'
    # args.accw=1
    # args.accw2=0
    # args.basename='HKK_a0019'
    args.restore_step=720000
    # args.text='He turned sharply and faced Gregson across the table'
    args.text=None
    # args.siga=0.001
    # args.sigs=-0.001
    # args.flata=True
    # args.flats=True
    args.batch_size=16

    # Check source texts
    if args.mode == "batch":
        assert args.source is not None and args.text is None
    if args.mode == "single":
        assert args.source is None and args.text is not None

    # Read Config
    preprocess_config, model_config, train_config = get_configs_of(args.dataset)
    configs = (preprocess_config, model_config, train_config)
    if preprocess_config["preprocessing"]["pitch"]["pitch_type"] == "cwt":
        from utils.pitch_tools import get_lf0_cwt
        preprocess_config["preprocessing"]["pitch"]["cwt_scales"] = get_lf0_cwt(np.ones(10))[1]
    os.makedirs(
        os.path.join(train_config["path"]["result_path"], str(args.restore_step)), exist_ok=True)

    # Set Device
    torch.manual_seed(train_config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(train_config["seed"])
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print("Device of CompTransTTS:", device)

    # Get model
    model = get_model(args, configs, device, train=False)

    # Load vocoder
    vocoder = get_vocoder(model_config, device)

    # Preprocess texts
    if args.mode == "batch":
        # Get dataset
        dataset = TextDataset(args.source, preprocess_config, model_config)
        batchs = DataLoader(
            dataset,
            batch_size=args.batch_size,
            collate_fn=dataset.collate_fn,
        )
    if args.mode == "single":
        ids = raw_texts = [args.text[:100]]

        # Speaker Info
        load_spker_embed = model_config["multi_speaker"] \
            and preprocess_config["preprocessing"]["speaker_embedder"] != 'none'
        with open(os.path.join(preprocess_config["path"]["preprocessed_path"], "speakers.json")) as f:
            speaker_map = json.load(f)
        speakers = np.array([speaker_map[args.speaker_id]]) if model_config["multi_speaker"] else np.array([0]) # single speaker is allocated 0
        spker_embed = np.load(os.path.join(
            preprocess_config["path"]["preprocessed_path"],
            "spker_embed",
            "{}-spker_embed.npy".format(args.speaker_id),
        )) if load_spker_embed else None

        if preprocess_config["preprocessing"]["text"]["language"] == "en":
            texts = np.array([preprocess_english(args.text, preprocess_config)])
        # elif preprocess_config["preprocessing"]["text"]["language"] == "zh":
        #     texts = np.array([preprocess_mandarin(args.text, preprocess_config)])
        text_lens = np.array([len(texts[0])])

        # SINGLE ONE
        # with open(os.path.join(preprocess_config["path"]["preprocessed_path"], "accents.json")) as f:
        #     accent_map = json.load(f)
            
        # accents_to_indices = dict()
        
        # for _idx, acc in enumerate(preprocess_config['accents']):
        #     accents_to_indices[acc] = _idx
      
        # accents = np.array([accents_to_indices[accent_map[ref_spk]]])

        # SAMPLING one
        acc_name=args.accent
        accents_to_indices = dict()
        for _idx, acc in enumerate(preprocess_config['accents']):
            accents_to_indices[acc] = _idx
        accents = np.array([accents_to_indices[acc_name]])

        batchs = [(ids, raw_texts, speakers, texts, text_lens, max(text_lens), spker_embed, accents)]

    control_values = args.pitch_control, args.energy_control, args.duration_control

    synthesize_stats(device, model, args, configs, vocoder, batchs, control_values)
