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
from utils.tools import get_configs_of, to_device, synth_samples_conversion
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

def synthesize_stats(device, model, args, configs, vocoder, batchs, control_values, listening_test_path):
    preprocess_config, model_config, train_config = configs
    pitch_control, energy_control, duration_control = control_values
    arraypath = train_config["path"]["array_path"]

    acc_mu=np.load(os.path.join(arraypath,'inf_acc_mu.npy'))
    acc_var=np.load(os.path.join(arraypath,'inf_acc_var.npy'))

    spk_mu=np.load(os.path.join(arraypath,'inf_spk_mu.npy'))
    spk_var=np.load(os.path.join(arraypath,'inf_spk_var.npy'))

    acc_id=np.load(os.path.join(arraypath,'inf_acc_id.npy'))
    spk_id=np.load(os.path.join(arraypath,'inf_spk_id.npy'))



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
            z_acc=np.mean(acc_mu[acc_id==batch[-1].cpu().item()],axis=0)
            z_spk=np.mean(spk_mu[spk_id==batch[2].cpu().item()],axis=0)
            accents2=None

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
            synth_samples_conversion(
                batch,
                output,
                vocoder,
                model_config,
                preprocess_config,
                listening_test_path,
                args,
            )



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--mode",
    #     type=str,
    #     choices=["batch", "single"],
    #     required=True,
    #     help="Synthesize a whole dataset or a single sentence",
    # )
    parser.add_argument(
        "--pitch_control",
        type=float,
        default=1.0,
        help="control the pitch of the whole utterance, larger value for higher pitch",
    )
    parser.add_argument(
        "--energy_control",
        type=float,
        default=1.0,
        help="control the energy of the whole utterance, larger value for larger volume",
    )
    parser.add_argument(
        "--duration_control",
        type=float,
        default=1.0,
        help="control the speed of the whole utterance, larger value for slower speaking rate",
    )
    parser.add_argument("--restore_step", type=int, default=704000)
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        default=L2ARCTIC,
        help="name of dataset",
    )
    args = parser.parse_args()

    listening_test_path="./output/result/L2ARCTIC/listening_test"
    # args.dataset='L2ARCTIC'
    # args.pitch_control=1.0
    # args.energy_control=1.0
    # args.duration_control=1.0
    args.mode='single'

    # args.source='val.txt'
    args.source=None
    # args.speaker_id='NCC'
    # args.basename='SVBI_a0009'
    # args.speaker_id='SVBI'
    # args.accent='Hindi'
    # args.restore_step=704000
    args.text='He turned sharply and faced Gregson across the table'



    spklist=["SVBI","HKK","NCC","THV","ABA","EBVS"]
    acclist=["Arabic", "Chinese", "Hindi", "Korean", "Spanish", "Vietnamese"]

    fulltxtlist=["And you always come to that shop to order the same meal",
    "This piece of cake is so yummy, I can't wait to bake another one",
    "I will go inside and tell the truth",
    "My dog is a silly companion",
    "Without you, I would not be able to do it",
    "He will knock you down with a single hit"]

    indexlist=[0,1,2,3,4,5]
    txtlist=[fulltxtlist[i] for i in indexlist]

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
    # os.makedirs(
    #     os.path.join(train_config["path"]["result_path"], str(args.restore_step)), exist_ok=True)
    os.makedirs(
        os.path.join(listening_test_path, str(args.restore_step)), exist_ok=True)


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
    i=0

    for spk in spklist:
        for txt in txtlist:
            for acc in acclist:

                print(i)
                i+=1

                args.speaker_id=spk
                args.accent=acc
                args.text=txt


                # Preprocess texts
                if args.mode == "batch":
                    # Get dataset
                    dataset = TextDataset(args.source, preprocess_config, model_config)
                    batchs = DataLoader(
                        dataset,
                        batch_size=16,
                        collate_fn=dataset.collate_fn,
                    )
                if args.mode == "single":
                    ids = raw_texts = [args.text[:100]]

                    # Speaker Info
                    # load_spker_embed = model_config["multi_speaker"] \
                    #     and preprocess_config["preprocessing"]["speaker_embedder"] != 'none'
                    with open(os.path.join(preprocess_config["path"]["preprocessed_path"], "speakers.json")) as f:
                        speaker_map = json.load(f)
                    speakers = np.array([speaker_map[args.speaker_id]]) if model_config["multi_speaker"] else np.array([0]) # single speaker is allocated 0
                    # spker_embed = np.load(os.path.join(
                    #     preprocess_config["path"]["preprocessed_path"],
                    #     "spker_embed",
                    #     "{}-spker_embed.npy".format(args.speaker_id),
                    # )) if load_spker_embed else None
                    spker_embed = None
                    if preprocess_config["preprocessing"]["text"]["language"] == "en":
                        texts = np.array([preprocess_english(args.text, preprocess_config)])
                    # elif preprocess_config["preprocessing"]["text"]["language"] == "zh":
                    #     texts = np.array([preprocess_mandarin(args.text, preprocess_config)])
                    text_lens = np.array([len(texts[0])])

                    #
                    acc_name=args.accent
                    accents_to_indices = dict()
                    for _idx, acc in enumerate(preprocess_config['accents']):
                        accents_to_indices[acc] = _idx
                    accents = np.array([accents_to_indices[acc_name]])

                    batchs = [(ids, raw_texts, speakers, texts, text_lens, max(text_lens), spker_embed, accents)]

                control_values = args.pitch_control, args.energy_control, args.duration_control

                synthesize_stats(device, model, args, configs, vocoder, batchs, control_values, listening_test_path)
