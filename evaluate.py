import argparse
import os

import torch
import yaml
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.model import get_model, get_vocoder
from utils.tools import to_device, log, synth_one_sample, plot_embedding, evaluation_synth_fc
from model import CompTransTTSLoss
from dataset import Dataset
import numpy as np

def evaluate(device, model, step, configs, logger=None, vocoder=None, losses=None):
    preprocess_config, model_config, train_config = configs
    out_dir = train_config["path"]["plot_path"]
    array_path = train_config["path"]["array_path"]


    # colors = 'k','r','b','g','y','c','m'
    colors = 'r','b','g','y','c','m'
    labels = preprocess_config["accents"]
    colors2 = 'g','r','r','c','g','b','b','m','b','c','m','b','y','g','m','c','g','r','y','m','c','r','y','y'
    # colors2 = 'r','b','g','y','k','c','r','b','g','y','k','c','r','b','g','y','k','c','r','b','g','y','k','c','r','b','g','y'
    labels2 = ["RRBI", "ABA", "SKA", "EBVS", "TNI", "NCC", "BWC", "HQTV", "TXHC", "ERMS", "PNV", "LXC", "HKK", "ASI", "THV", "MBMPS", "SVBI", "ZHAA", "HJK", "TLV", "NJS", "YBAA", "YDCK", "YKWK"]
    # labels2 = ["RRBI", "ABA", "SKA", "EBVS", "TNI", "NCC", "BWC", "HQTV", "TXHC", "ERMS", "CLB", "PNV", "BDL", "LXC", "HKK", "ASI", "THV", "MBMPS", "SLT", "SVBI", "ZHAA", "HJK", "RMS", "TLV", "NJS", "YBAA", "YDCK", "YKWK"]



    # Get dataset
    learn_alignment = model_config["duration_modeling"]["learn_alignment"]
    dataset_tag = "unsup" if learn_alignment else "sup"
    dataset = Dataset(
        "val_{}.txt".format(dataset_tag), preprocess_config, model_config, train_config, sort=False, drop_last=False
    )
    batch_size = train_config["optimizer"]["batch_size"]
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn,
    )

    # Get loss function
    Loss = CompTransTTSLoss(preprocess_config, model_config, train_config).to(device).eval()

    # Evaluation
    loss_sums = [{k:0 for k in loss.keys()} if isinstance(loss, dict) else 0 for loss in losses]
    accent_id_list=[]
    spk_id_list=[]
    acc_emb_list=[]
    acc_emb_sg_list=[]
    spk_emb_list=[]
    evalind=0
    randomind=np.random.randint(batch_size)
    # stable_batch=[]
    # inf_batch=[]
    for batchs in loader:
        for batch in batchs:
            print(evalind)
            evalind+=1
            batch = to_device(batch, device)
            # if evalind==1:
            #     for kk,row in enumerate(batch):
            #         print(kk)
            #         # print(row)
            #         # if type(row)=='int' or type(row)=='numpy.int64':
            #         if kk in [5,8]:
            #             stable_batch.append(row)
            #             inf_batch.append(row)
            #         else:
            #             stable_batch.append(row[0])
            #             inf_batch.append(row[randomind])
            #     print(stable_batch)
            with torch.no_grad():
                # Forward
                output = model(*(batch[2:]), step=step)
                mlvae_stats=output[1]
                batch[9:11], output = output[-2:], output[:-2] # Update pitch and energy by VarianceAdaptor
                # Cal Loss
                losses = Loss(batch, output, step=step)

                for i in range(len(losses)):
                    if isinstance(losses[i], dict):
                        for k in loss_sums[i].keys():
                            loss_sums[i][k] += losses[i][k].item() * len(batch[0])
                    else:
                        loss_sums[i] += losses[i].item() * len(batch[0])

                acc_emb = mlvae_stats[1][0]
                spk_emb = mlvae_stats[1][2]
                acc_emb_sg = mlvae_stats[1][4]
                acc_emb_list.append(acc_emb.cpu().detach())
                acc_emb_sg_list.append(acc_emb_sg.cpu().detach())
                spk_emb_list.append(spk_emb.cpu().detach())

                accent_id_list.append(batch[-1].cpu().detach())#check
                spk_id_list.append(batch[2].cpu().detach())


    embedding_acc=np.zeros((512,acc_emb_list[0].size()[1]))
    embedding_acc_sg=np.zeros((512,acc_emb_sg_list[0].size()[1]))
    embedding_spk=np.zeros((512,spk_emb_list[0].size()[1]))

    embedding_accent_id=np.zeros((512))
    embedding_spk_id=np.zeros((512))

    xi=0
    for bat in acc_emb_list:
        for ii in bat:
            embedding_acc[xi,:] = np.array(ii)
            xi+=1

    xi=0
    for bat in acc_emb_sg_list:
        for ii in bat:
            embedding_acc_sg[xi,:] = np.array(ii)
            xi+=1

    xi=0
    for bat in spk_emb_list:
        for ii in bat:
            embedding_spk[xi,:] = np.array(ii)
            xi+=1

    xi=0
    for bat in accent_id_list:
        for ii in bat:
            embedding_accent_id[xi] = np.array(ii)
            xi+=1

    xi=0
    for bat in spk_id_list:
        for ii in bat:
            embedding_spk_id[xi] = np.array(ii)
            xi+=1    

    embedding_accent_id=np.int32(embedding_accent_id) 
    np.save(os.path.join(array_path,'acc_mu.npy'),embedding_acc)
    np.save(os.path.join(array_path,'acc_mu_sg.npy'),embedding_acc_sg)
    np.save(os.path.join(array_path,'spk_mu.npy'),embedding_spk)
    np.save(os.path.join(array_path,'acc_id.npy'),embedding_accent_id)
    np.save(os.path.join(array_path,'spk_id.npy'),embedding_spk_id)


    plot_embedding(out_dir, embedding_acc, embedding_accent_id,colors,labels,filename=str(step)+'embeddingacc.png')

    plot_embedding(out_dir, embedding_acc_sg, embedding_accent_id,colors,labels,filename=str(step)+'embeddingaccsg.png')

    plot_embedding(out_dir, embedding_spk, embedding_spk_id,colors2,labels2,filename=str(step)+'embeddingspk.png')

    plot_embedding(out_dir, embedding_acc, embedding_spk_id,colors2,labels2,filename=str(step)+'embeddingacc_spklabels.png')
    plot_embedding(out_dir, embedding_acc_sg, embedding_spk_id,colors2,labels2,filename=str(step)+'embeddingacc_sg_spklabels.png')
    plot_embedding(out_dir, embedding_spk, embedding_accent_id,colors,labels,filename=str(step)+'embeddingspk_acclabels.png')

    plot_embedding(out_dir, np.concatenate((embedding_acc,embedding_spk),1), embedding_spk_id,colors2,labels2,filename=str(step)+'embeddingcombined_spklabels.png')
    plot_embedding(out_dir, np.concatenate((embedding_acc,embedding_spk),1), embedding_accent_id,colors,labels,filename=str(step)+'embeddingcombined_acclabels.png')


    loss_means = []
    loss_means_ = []
    for loss_sum in loss_sums:
        if isinstance(loss_sum, dict):
            loss_mean = {k:v / len(dataset) for k, v in loss_sum.items()}
            loss_means.append(loss_mean)
            loss_means_.append(sum(loss_mean.values()))
        else:
            loss_means.append(loss_sum / len(dataset))
            loss_means_.append(loss_sum / len(dataset))

    message = "Total Loss: {:.4f}, Mel Loss: {:.4f}, Mel PostNet Loss: {:.4f}, Pitch Loss: {:.4f}, Energy Loss: {:.4f}, Duration Loss: {:.4f}, CTC Loss: {:.4f}, Binarization Loss: {:.4f},\
        Prosody Loss: {:.4f},Encoder Loss: {:.4f} \
        KL Loss Accent: {:.4f},KL Loss Speaker: {:.4f} \
        Accent VQ Loss: {:.4f},Speaker VQ Loss: {:.4f}".format(
        *([step] + [l for l in loss_means_])
        )
    # message = "Validation Step {}, Total Loss: {:.4f}, Mel Loss: {:.4f}, Mel PostNet Loss: {:.4f}, Pitch Loss: {:.4f}, Energy Loss: {:.4f}, Duration Loss: {:.4f}, CTC Loss: {:.4f}, Binarization Loss: {:.4f}, Prosody Loss: {:.4f},Speaker Loss: {:.4f}".format(
    #     *([step] + [l for l in loss_means_])
    # )

    loader2 = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=dataset.collate_fn,
    )

    stableind=5
    pickind=int(np.random.randint(50))
    while pickind==stableind:
        pickind=int(np.random.randint(50))
    curind=0
    for batchs in loader2:
        for batchx in batchs:
            batchx = to_device(batchx, device)
            if pickind==curind:
                inf_batch=batchx
            elif stableind==curind:
                stable_batch=batchx
            curind+=1


        # return (
        #     ids,
        #     raw_texts,
        #     speakers,
        #     texts,
        #     text_lens,
        #     max(text_lens),
        #     mels,
        #     mel_lens,
        #     max(mel_lens),
        #     pitches,
        #     f0s,
        #     uvs,
        #     cwt_specs,
        #     f0_means,
        #     f0_stds,
        #     energies,
        #     durations,
        #     mel2phs,
        #     attn_priors,
        #     spker_embeds,
        #     accents,
        # )

        # (ids, raw_texts, speakers, texts, src_lens, max_src_len, spker_embeds, accents) = data
        # loader = [(ids, raw_texts, speakers, texts, mel, text_lens, max(text_lens), spker_embed, accents)]

    for k in [12,11,10,9,8,7,6]:#[6,7,8,9,10,11,12,13,14,15,16,17,18]
        inf_batch.pop(k)
        stable_batch.pop(k)
        # inf_batch.append(batchik[k,pickind])
        # stable_batch.append(batchik[k,stableind])

    acc_pick=np.random.randint(6)
    spk_pick=np.random.randint(24)

    z_acc=torch.from_numpy(np.mean(embedding_acc[embedding_accent_id==acc_pick],axis=0)).unsqueeze(0).to(device).type('torch.cuda.FloatTensor') #pick random accent
    z_spk=torch.from_numpy(np.mean(embedding_spk[embedding_spk_id==spk_pick],axis=0)).unsqueeze(0).to(device).type('torch.cuda.FloatTensor') #pick random speaker

    output_conv1 = model.inference_stats(
        *(inf_batch[2:-2]),
        spker_embeds=inf_batch[-2],
        accents=inf_batch[-1],
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
        z_acc=z_acc,
        z_spk=z_spk,
    )

    z_acc2=torch.from_numpy(np.mean(embedding_acc[embedding_accent_id==1],axis=0)).unsqueeze(0).to(device).type('torch.cuda.FloatTensor') #Chinese
    z_spk2=torch.from_numpy(np.mean(embedding_spk[embedding_spk_id==16],axis=0)).unsqueeze(0).to(device).type('torch.cuda.FloatTensor') #SVBI
    
    output_conv2 = model.inference_stats(
        *(stable_batch[2:-2]),
        spker_embeds=stable_batch[-2],
        accents=stable_batch[-1],
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
        z_acc=z_acc2,
        z_spk=z_spk2,
    )

    if logger is not None:
        figs, fig_attn, wav_reconstruction, wav_prediction, tag = synth_one_sample(
            batch,
            output,
            vocoder,
            model_config,
            preprocess_config,
        )

        figs_inf, wav_inf1, wav_inf2 = evaluation_synth_fc(
            inf_batch,
            stable_batch,
            output_conv1,
            output_conv2,
            vocoder,
            model_config,
            preprocess_config,
        )

        log(logger, step, losses=loss_means)
        if fig_attn is not None:
            log(
                logger,
                step,
                img=fig_attn,
                tag="Validation/attn",
            )
        log(
            logger,
            step,
            figs=figs,
            tag="Validation",
        )
        sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
        log(
            logger,
            step,
            audio=wav_reconstruction,
            sampling_rate=sampling_rate,
            tag="Validation/reconstructed",
        )
        log(
            logger,
            step,
            audio=wav_prediction,
            sampling_rate=sampling_rate,
            tag="Validation/synthesized",
        )
        log(
            logger,
            step,
            audio=wav_inf1,
            sampling_rate=sampling_rate,
            tag="Validation/converted1_{}_{}".format(int(acc_pick),int(spk_pick)),
        )
        log(
            logger,
            step,
            audio=wav_inf2,
            sampling_rate=sampling_rate,
            tag="Validation/converted2",
        )
    return message
