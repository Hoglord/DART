from transformers import Wav2Vec2FeatureExtractor, HubertModel
from .pooling import AttentivePooling, MeanStdPooling, GruPooling
from .loss import AngularProximityMagnet, Am_softmax, ArcFace, ArcLinear, ArcMarginModel, ArcMarginProduct, SoftmaxAngularProto
from .loss import l2_norm
import os
import numpy as np
import torch.nn as nn
import torch
from collections import OrderedDict


class Xtractor(nn.Module):
    
    def __init__(self,preprocess_config,model_config,train_config) -> None:
        super(Xtractor,self).__init__()
        
        self.preprocess_config = preprocess_config
        self.model_config = model_config
        self.train_config = train_config
        self.speaker_number = preprocess_config["preprocessing"]["speaker_number"]
        self.pretrained_model_name = preprocess_config["HuBERT"]["pretrained_model_name"]
        self.load_saved_states = preprocess_config["HuBERT"]["load_saved_states"]
        if self.load_saved_states:
            pass
        else:
            self.processor = Wav2Vec2FeatureExtractor.from_pretrained(self.pretrained_model_name)
            self.model = HubertModel.from_pretrained(self.pretrained_model_name)
            self.model.feature_extractor._freeze_parameters()
        
        self.embedding_size = model_config["external_speaker_dim"]
        
        self.before_speaker_embedding = torch.nn.Sequential(OrderedDict([
            ("lin_be",torch.nn.Linear(in_features = 2048, out_features = self.embedding_size,bias=False)),
            ("bn_be", torch.nn.BatchNorm1d(self.embedding_size)) 
            ]))
        
        # self.stat_pooling = AttentivePooling(1024,
        #                                      num_freqs=1,
        #                                      attention_channels=128,
        #                                      global_context=False)
        if model_config["stat_pooling"]["type"] == "mean":
            self.stat_pooling = MeanStdPooling()
        elif model_config["stat_pooling"]["type"] == "GRU":
            self.stat_pooling = GruPooling(input_size = model_config["stat_pooling"]["input_size"],
                                           gru_node = model_config["stat_pooling"]["gru_node"],
                                           nb_gru_layer = model_config["stat_pooling"]["nb_gru_layer"])
        elif model_config["stat_pooling"]["type"] == "attentive":
            self.stat_pooling = AttentivePooling(256,80,global_context=True)
            
        
        
        self.loss = train_config["loss"]["xtractor_loss"]
        
        if self.loss == "aam":
            self.after_speaker_embedding  = ArcMarginProduct(self.embedding_size,
                                                             int(self.speaker_number),
                                                             s=30,
                                                             m=0.2,
                                                             easy_margin = False)
        elif self.loss == 'aps':
            self.after_speaker_embedding = SoftmaxAngularProto(int(self.speaker_number))
        elif self.loss == 'smn':
            self.after_speaker_embedding = AngularProximityMagnet(int(self.speaker_number))
            
            
    def forward(self,x,is_eval=False,target=None,norm_embedding=True):
        
        if not self.load_saved_states:
            inputs = self.processor(x,return_tensors="pt",sampling_rate=self.preprocess_config["preprocessing"]["audio"]["sampling_rate"])
            input_values = inputs["input_values"].cuda()
            attention_mask = inputs["attention_mask"].cuda()
            hidden_states = self.model(input_values=input_values.squeeze(0),attention_mask=attention_mask.squeeze(0)).last_hidden_state
        else:
            hidden_states = x
        
        output = self.stat_pooling(torch.transpose(hidden_states,1,2))
        
        output = self.before_speaker_embedding(output)
        
        if norm_embedding:
            output = l2_norm(output)
        
        
        if self.loss in ["aam",'aps','circle']:
            output = self.after_speaker_embedding(output,target=target), torch.nn.functional.normalize(output,dim=1)
        elif self.loss == 'smn':
            if not is_eval:
                output = self.after_speaker_embedding(output,target=target), output
        
        return output
        
        
            