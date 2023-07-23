# from model import PreDefinedEmbedder
# from encoder.model import SpeakerEncoder
from encoder import inference as encoder
from encoder.params_model import model_embedding_size as speaker_embedding_size
import librosa
import numpy as np
import yaml
import torch
import os
from tqdm import tqdm
import argparse
import pathlib
import scipy
from sklearn.metrics.pairwise import cosine_similarity
from Metrics.f0_frame_error import FFE
from utils.tools import pad_1D


import math
import glob
import pyworld
import json
import pysptk
import matplotlib.pyplot as plot
import torch
import zipfile
import torchaudio
import jiwer
import argparse

# from binary_io import BinaryIOCollection
def compute_mcd(args):
    
    ref_wav_path = args.ref_wav_dir
    
    synth_wav_path = args.synth_wav_dir
    
    # SAMPLING_RATE = 22050
    FRAME_PERIOD = 5.0
    alpha = 0.65  # commonly used at 22050 Hz
    fft_size = 512
    mcep_size = 25
    
    # Compute mcep for reference audio
    ref_mcep_dir = "mcdout/mcep_numpy/ref"
    # compute mcep for synthesied audio
    synth_mcep_dir = "mcdout/mcep_numpy/synth"
    
    if os.path.exists(f"{ref_wav_path}/{ref_mcep_dir}"):
        pass
    else:
        os.makedirs(f"{ref_wav_path}/{ref_mcep_dir}")
        
    if os.path.exists(f"{synth_wav_path}/{synth_mcep_dir}"):
        pass
    else:
        os.makedirs(f"{synth_wav_path}/{synth_mcep_dir}")
                      
    
    def load_wav(wav_file, sr):
        """
        Load a wav file with librosa.
        :param wav_file: path to wav file
        :param sr: sampling rate
        :return: audio time series numpy array
        """
        wav, _ = librosa.load(wav_file, sr=sr, mono=True)

        return wav


    def log_spec_dB_dist(x, y):
        log_spec_dB_const = 10.0 / math.log(10.0) * math.sqrt(2.0)
        diff = x - y
        
        return log_spec_dB_const * math.sqrt(np.inner(diff, diff))

    def wav2mcep_numpy(wavfile, target_directory, alpha=0.65, fft_size=512, mcep_size=34,type=None):
        # make relevant directories
        if not os.path.exists(target_directory):
            os.makedirs(target_directory)

        loaded_wav = load_wav(wavfile, sr=args.sampling_rate)

        # Use WORLD vocoder to spectral envelope
        _, sp, _ = pyworld.wav2world(loaded_wav.astype(np.double), fs=args.sampling_rate,
                                    frame_period=FRAME_PERIOD, fft_size=fft_size)

        # Extract MCEP features
        mgc = pysptk.sptk.mcep(sp, order=mcep_size, alpha=alpha, maxiter=0,
                            etype=1, eps=1.0E-8, min_det=0.0, itype=3)

        # fname = os.path.basename(wavfile).split('.')[0]
        base_name = wavfile.split("/")[-2:]
        # if type == "ref":
        #     basename = wavfile.split("/")[-3]
        # else:
        #     basename = wavfile.split("/")[-2]
            
        # np.save(os.path.join(target_directory, basename + '-' + fname + '.npy'),
        #         mgc,
        #         allow_pickle=False)
        # np.save(os.path.join(target_directory, fname + '.npy'),
        # mgc,
        # allow_pickle=False)

        #L2ARCTIC
        os.makedirs(os.path.join(target_directory,base_name[0]),exist_ok=True)
        np.save(os.path.join(target_directory, base_name[0], base_name[1].split('.')[0] + '.npy'),
        mgc,
        allow_pickle=False)

    # computer average mcd using mcep files 
    def average_mcd(ref_mcep_files, synth_mcep_files, cost_function):
        """
        Calculate the average MCD.
        :param ref_mcep_files: list of strings, paths to MCEP target reference files
        :param synth_mcep_files: list of strings, paths to MCEP converted synthesised files
        :param cost_function: distance metric used
        :returns: average MCD, total frames processed
        """
        min_cost_tot = 0.0
        frames_tot = 0
        i=0
        for ref,synth in zip(ref_mcep_files,synth_mcep_files):
            ref_vec = np.load(ref)
            ref_frame_no = len(ref_vec)
            synth_vec = np.load(synth)

            # dynamic time warping using librosa
            min_cost, _ = librosa.sequence.dtw(ref_vec[:, 1:].T, synth_vec[:, 1:].T, 
                                            metric=cost_function)
            min_cost_tot += np.mean(min_cost)
            frames_tot += ref_frame_no
            print('mcd',i)
            i+=1
                    
        mean_mcd = min_cost_tot / frames_tot
        
        return mean_mcd, frames_tot

    ref_wav_files = glob.glob(f"{ref_wav_path}/*/*.wav",recursive=True)
    synth_wav_files = glob.glob(f"{synth_wav_path}/*/*.wav",recursive=True)

    
    for wav in tqdm(ref_wav_files):
        wav2mcep_numpy(wav, f"{ref_wav_path}/{ref_mcep_dir}", fft_size=fft_size, mcep_size=mcep_size)
    for wav in tqdm(synth_wav_files):
        wav2mcep_numpy(wav, f"{synth_wav_path}/{synth_mcep_dir}", fft_size=fft_size, mcep_size=mcep_size)
    trg_refs = glob.glob(f"{ref_wav_path}/{ref_mcep_dir}/*/*.npy")
    conv_synths = glob.glob(f"{synth_wav_path}/{synth_mcep_dir}/*/*.npy")
    cost_function = log_spec_dB_dist

    mcd, tot_frames_used = average_mcd(trg_refs, conv_synths, cost_function)
    return mcd, tot_frames_used

def computer_wer(args):
    ref_wav_path = args.ref_wav_dir
    synth_wav_path = args.synth_wav_dir
    device = torch.device('cpu')  # gpu also works, but our models are fast enough for CPU
    
    model, decoder, utils = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                       model='silero_stt',
                                       jit_model='jit_xlarge',
                                       language='en', # also available 'de', 'es'
                                       device=device)
    
    (read_batch, split_into_batches,read_audio, prepare_model_input) = utils  # see function signature for details
    
    ref_txt_files = glob.glob(f"{ref_wav_path}/*/*.lab",recursive=True)
    # print(len(ref_wav_files))
    batches = split_into_batches(ref_txt_files, batch_size=10)
    # print(len(batches))
    ground_truth = []
    hypothesis = []
    
    for batch in tqdm(batches):
        # print(batch)
        # input = prepare_model_input(read_batch(batch),
        #                         device=device)
        # output = model(input)
        for example in batch:
            # print(decoder(example.cpu()))
            with open(example) as f:
                lines = f.readlines()
            ground_truth.append(lines[0])
            # file_list = str(example).split("/")
            # file_name = file_list[-1].split(".")[0]
            # basename = file_list[-3]
            # synth_wav_file = synth_wav_path + "/" + basename + "/" +file_name + ".wav"
            # synth_wav_file = synth_wav_path + "/" + file_name + ".wav"
            synth_wav_file = os.path.join(synth_wav_path, example.split('/')[-2], os.path.basename(example.split('.')[0]) + '.wav')
            input = prepare_model_input(read_batch([synth_wav_file]), device=device)
            output = model(input)
            hypothesis.append(decoder(output[0].cpu()))
        
    # synth_wav_files = glob.glob(f"{synth_wav_path}/*/*.wav",recursive=True)    
    # batches = split_into_batches(synth_wav_files, batch_size=10)
   
    
    # for batch in tqdm(batches):
    #     input = prepare_model_input(read_batch(batch),
    # #                             device=device)
    #     output = model(input)
    #     for example in output:
    #         # print(decoder(example.cpu()))
    #         hypothesis.append(decoder(example.cpu()))

    transformation = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemoveWhiteSpace(replace_by_space=True),
    jiwer.RemoveMultipleSpaces(),
    jiwer.ReduceToListOfListOfWords(word_delimiter=" ")
    ])
    
    wer = jiwer.wer(
    ground_truth, 
    hypothesis, 
    truth_transform=transformation, 
    hypothesis_transform=transformation)
    
    
    cer = jiwer.cer(
    ground_truth, 
    hypothesis, 
    truth_transform=transformation, 
    hypothesis_transform=transformation)
    
    return wer,cer


def average_frame_error_rate(args):
    ref_wav_dir = args.ref_wav_dir
    ref_wav_dir = pathlib.Path(ref_wav_dir)
    synth_wav_dir = args.synth_wav_dir
    
    def load_audio(wav_path):
        wav_raw, _ = librosa.load(wav_path, sr=args.sampling_rate)
        _, index = librosa.effects.trim(wav_raw, top_db= args.trim_top_db, frame_length= args.filter_length, hop_length= args.hop_length)
        wav = wav_raw[index[0]:index[1]]
        duration = (index[1] - index[0]) / args.hop_length
        return wav_raw.astype(np.float32), wav.astype(np.float32), int(duration)

    preprocess_config = yaml.load(open(
        os.path.join(args.config_dir, "preprocess.yaml"), "r"), Loader=yaml.FullLoader)
    
    ffe = FFE(args.sampling_rate)
    
    frame_error_rate = []
    for ref_wav_path in tqdm(ref_wav_dir.rglob("*.wav")):
        try:
            ref_wav_raw, ref_wav, ref_duration = load_audio(ref_wav_path)
            # ref_spker_embed = speaker_emb(ref_wav)
            #added
            #'''
            # ref_spker_embed = encoder.embed_utterance(ref_wav).reshape(1, -1)
            #'''
            if args.dataset=="L2ARCTIC":
                wav_name = str(ref_wav_path).split("/")[-2:]
                synth_wav_path = os.path.join(synth_wav_dir, wav_name[0], wav_name[1])
            else:
                wav_name = str(ref_wav_path).split("/")[-1]
                synth_wav_path = synth_wav_dir + "/" + wav_name
            # print("----->>>>", wav_name)
            synth_wav_raw, synth_wav, synth_duration = load_audio(synth_wav_path)
            data = [ref_wav,synth_wav]
            data = pad_1D(data)
            ref_wav,synth_wav = data
            
            # synth_spker_embed = speaker_emb(synth_wav)
            #added
            #'''
            # synth_spker_embed = encoder.embed_utterance(synth_wav).reshape(1,-1)
            #'''
            # breakpoint()
            # score = cosine_similarity(ref_spker_embed, synth_spker_embed)
            score = ffe.calculate_ffe(torch.tensor(ref_wav),torch.tensor(synth_wav))
            # print(score)
            frame_error_rate.append(score)
        except Exception as e:
            print(e)
            
    
    return np.mean(frame_error_rate),np.var(frame_error_rate), frame_error_rate
    

def average_cosine_similarity(args):
    ref_wav_dir = args.ref_wav_dir
    ref_wav_dir = pathlib.Path(ref_wav_dir)
    synth_wav_dir = args.synth_wav_dir
    # synth_wav_dir = pathlib.Path(synth_wav_dir)
 
    def load_audio(wav_path):
        wav_raw, _ = librosa.load(wav_path, sr=args.sampling_rate)
        _, index = librosa.effects.trim(y=wav_raw, top_db= args.trim_top_db, frame_length= args.filter_length, hop_length= args.hop_length)
        wav = wav_raw[index[0]:index[1]]
        duration = (index[1] - index[0]) / args.hop_length
        return wav_raw.astype(np.float32), wav.astype(np.float32), int(duration)

    preprocess_config = yaml.load(open(
            os.path.join(args.config_dir, "preprocess.yaml"), "r"), Loader=yaml.FullLoader)

    # if preprocess_config["preprocessing"]["speaker_embedder"] == "DeepSpeaker":
    #     speaker_emb = PreDefinedEmbedder(preprocess_config)
    # elif preprocess_config["preprocessing"]["speaker_embedder"] == "GE2E":
    #     encoder.load_model(Path(preprocess_config["preprocessing"]["speaker_embedder_path"]))
        
    encoder.load_model(Path("./encoder/pretrained_models/encoder.pt"))

    cosine_score = []
    for ref_wav_path in tqdm(ref_wav_dir.rglob("*.wav")):
        try:
            ref_wav_raw, ref_wav, ref_duration = load_audio(ref_wav_path)
            # ref_spker_embed = speaker_emb(ref_wav)
            #added
            #'''
            ref_spker_embed = encoder.embed_utterance(ref_wav).reshape(1, -1)
            #'''
            if args.dataset=="L2ARCTIC":
                wav_name = str(ref_wav_path).split("/")[-2:]
                synth_wav_path = os.path.join(synth_wav_dir, wav_name[0], wav_name[1])
            else:
                wav_name = str(ref_wav_path).split("/")[-1]
                synth_wav_path = synth_wav_dir + "/" + wav_name
            # print("----->>>>", wav_name)
            synth_wav_raw, synth_wav, synth_duration = load_audio(synth_wav_path)
            # synth_spker_embed = speaker_emb(synth_wav)
            #added
            #'''
            synth_spker_embed = encoder.embed_utterance(synth_wav).reshape(1,-1)
            #'''
            # breakpoint()
            score = cosine_similarity(ref_spker_embed, synth_spker_embed)
            # print(score)
            cosine_score.append(score)
        except:
            print("Error: Processing the file")
            
    
    return np.mean(cosine_score),np.var(cosine_score), cosine_score



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="L2ARCTIC",
        help="Dataset",
    )
    parser.add_argument(
        "--ref_wav_dir",
        type=str,
        required=False,
        # default='/666/dsets/L2ARCTIC-16k/raw_data/',
        help="Path to orignal wav files",
    )
    parser.add_argument(
        "--synth_wav_dir",
        type=str,
        required=False,
        # default='/666/ASRU-MLVAE/output/result/L2ARCTIC/100000/',
        help="Path to denoised  wav files",
    )
    parser.add_argument(
        "--config_dir",
        type=str,
        default="./config/L2ARCTIC",
        help="config file",
    )
    parser.add_argument(
        "--trim_top_db",
        type=int,
        default=23,
        help="trim_top_db",
    )
    parser.add_argument(
        "--filter_length",
        type=int,
        default=1024,
        help="filter_length",
    )
    parser.add_argument(
        "--hop_length",
        type=int,
        default=256,
        help="hop_length",
    )
    parser.add_argument(
        "--sampling_rate",
        type=int,
        default=16000,
        help="sampling_rate",
    )
    args = parser.parse_args()
    return args


from pathlib import Path
def fetch_ref_from_synth(args):
    synth_wav_dir = Path(args.synth_wav_dir)
    dataset = args.dataset
    ref_wav_dir = os.path.join("/".join(args.synth_wav_dir.split("/")[:-2]), f"{dataset}_GT")
    if not os.path.exists(ref_wav_dir):
        os.makedirs(ref_wav_dir)

    # if dataset=="L2ARCTIC":
    spk_lab = ["RRBI", "ABA", "SKA", "EBVS", "TNI", "NCC", "BWC", "HQTV", "TXHC", "ERMS", "PNV", "LXC", "HKK", "ASI", "THV", "MBMPS", "SVBI", "ZHAA", "HJK", "TLV", "NJS", "YBAA", "YDCK", "YKWK"]
    for spk in spk_lab:
        os.makedirs(os.path.join(ref_wav_dir,spk),exist_ok=True)    

    for synth_wav_path in tqdm(synth_wav_dir.rglob("*.wav")):
        synth_wav_path = str(synth_wav_path)

        if dataset == "L2ARCTIC":
            base_name = synth_wav_path.split("/")[-2:]
            ref_wav_path = os.path.join(args.ref_wav_dir, base_name[0], base_name[1])
            ref_target_dir = os.path.join(ref_wav_dir, base_name[0], base_name[1])
            os.system(f"cp {ref_wav_path} {ref_target_dir}") # wavs
            os.system(f"cp {ref_wav_path.split('.')[0]+'.lab'} {ref_target_dir.split('.')[0]+'.lab'}") # texts
        else:
            base_name = synth_wav_path.split("/")[-1]
            if dataset == "VCTK":
                spk = base_name.split("-")[0]
            else:
                spk = base_name.split("_")[0]
            ref_wav_path = os.path.join(args.ref_wav_dir, spk, base_name)
            os.system(f"cp {ref_wav_path} {ref_wav_dir}")
    return ref_wav_dir
#EDIT FOR L2ARCTIC

if __name__ == "__main__":

    args = get_args()
    # args.ref_wav_dir='/666/dsets/L2ARCTIC-16k/raw_data/'
    # args.synth_wav_dir='/666/ASRU-MLVAE/output/result/L2ARCTIC/100000/obj_eval'
    if True:
        # fetch_ref_from_synth(args)
        new_path=fetch_ref_from_synth(args)
        args.ref_wav_dir=new_path
    
    mean_cos,var_cos, cosine_scores = average_cosine_similarity(args)
    print(f'COS mean={mean_cos}, var={var_cos}')
    
    mean_ffe,var_ffe, frame_error_rate = average_frame_error_rate(args)
    print(f'FFE mean={mean_ffe}, var={var_ffe}')

    with open("cos_sim_hyperx.txt", "w") as f:
        for cos in cosine_scores:
            # breakpoint()
            f.write(str(cos[0][0]))
            f.write("\n")
            # breakpoint()

    mcd,tot_frames_used = compute_mcd(args)
    print(f'MCD = {mcd} dB, calculated over a total of {tot_frames_used} frames')

    wer,cer = computer_wer(args)
    print(f"Word Error Rate = {wer}, Character Error Rate = {cer}")


    print('COS',mean_cos,var_cos)
    print('FFE',mean_ffe,var_ffe)
    print('MCD',mcd)
    print('WER CER',wer,cer)
    
    with open("obj_eval_results.txt", "w") as f:
        f.write('cosine sim '+ str(mean_cos) + '+-var '+ str(var_cos)+'\n')
        f.write('FFE '+ str(mean_ffe) + '+-var '+ str(var_ffe)+'\n')
        f.write('MCD '+ str(mcd) + '\n')
        f.write('WER '+ str(wer) + ' CER '+ str(cer))