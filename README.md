This is a demo repo for STAR-TTS, submitted to SLT2024

Code used in the paper is here

Audio samples are on the associated webpage: https://hoglord.github.io/STAR-TTS/

This code is based on https://github.com/keonlee9420/Comprehensive-Transformer-TTS

### Training

To train on L2-ARCTIC, you can call:
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --dataset L2ARCTIC
```
### Inference

For inference from a checkpoint, you can utilize the two enclosed functions `synthesize_converted.py` or `synthesize_stats_valset.py`. The first one would synthesize sentences in the script (a loop for speakers, accents, and sentences), the latter would synthesize from a metadata .txt file. Note that before you run any synthesis script, you should first run  `extract_stats.py` script on your current checkpoint to extract and save the MLVAE embeddings for speakers and accents first.

An example use of the synthesis scripts is:
```bash
CUDA_VISIBLE_DEVICES=0 python synthesize_converted.py --dataset L2ARCTIC --restore_step 704000
```
