import torch
import torch.nn as nn
from torch.nn import init
from collections import OrderedDict
import torch.nn.functional as F
from utils import reparameterize, group_wise_reparameterize, accumulate_group_evidence
from .transformers.blocks import LinearNorm
from torch.autograd import Variable

# MLVAE network
class MLVAENet(nn.Module):
    def __init__(self,model_config):
        super(MLVAENet,self).__init__()
        self.linear_model = nn.Sequential(OrderedDict([
            ('linear_1', nn.Linear(in_features=256, out_features=256, bias=True)),
            ('tan_h_1', nn.Tanh())
            ]))
        # style
        self.style_mu = LinearNorm(
                256, 
                model_config["speaker_encoder"]["z_dim"])
        self.style_logvar = LinearNorm(
                256,
                model_config["speaker_encoder"]["z_dim"])
        # class
        self.class_mu = LinearNorm(
                256,
                model_config["accent_encoder"]["z_dim"])
        self.class_logvar = LinearNorm(
                256, 
                model_config["accent_encoder"]["z_dim"])
        # self.accent_class = nn.Sequential(
        #         LinearNorm(
        #         model_config["speaker_encoder"]["z_dim"], 
        #         32),
        #         nn.ReLU(),
        #         LinearNorm(32,
        #         model_config["accent_encoder"]["n_classes"])
        # )
        # self.speaker_class = nn.Sequential(
        #         LinearNorm(
        #         model_config["accent_encoder"]["z_dim"], 
        #         32),
        #         nn.ReLU(),
        #         LinearNorm(32,
        #         model_config["speaker_encoder"]["n_classes"])
        # )
	# weight initialization
        for m in self.modules():
            if type(m) == nn.Linear or type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
                torch.nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)



    def forward(self, x, labels_batch):
        x = x.view(x.size(0), -1)
        
        x = self.linear_model(x)
        
        style_latent_space_mu = self.style_mu(x)
        style_latent_space_logvar = self.style_logvar(x)
        
        class_latent_space_mu = self.class_mu(x)
        class_latent_space_logvar = self.class_logvar(x)

        accent_latent_space_mu = class_latent_space_mu.clone()
        accent_latent_space_logvar = class_latent_space_logvar.clone()
        
        grouped_mu, grouped_logvar = accumulate_group_evidence(class_latent_space_mu.data, class_latent_space_logvar.data, labels_batch,is_cuda=True,device = accent_latent_space_mu.get_device())
        
        style_latent_embeddings = reparameterize(training=True, mu=style_latent_space_mu, logvar=style_latent_space_logvar)
        
        accent_latent_embeddings = reparameterize(training=True, mu=accent_latent_space_mu, logvar=accent_latent_space_logvar)

        class_latent_embeddings = group_wise_reparameterize(training=True, mu=grouped_mu, logvar=grouped_logvar, labels_batch=labels_batch,is_cuda=True,device = accent_latent_space_mu.get_device())
        
        # acc_prob = self.accent_class(style_latent_embeddings) #acc classifier on spk embs, for adv loss
        # spk_prob = self.speaker_class(class_latent_embeddings) #spk classifier on acc embs,

        # z = torch.cat((style_latent_embeddings, class_latent_embeddings), dim=1)



        return (class_latent_embeddings, style_latent_embeddings, accent_latent_embeddings, (grouped_mu, grouped_logvar, style_latent_space_mu, style_latent_space_logvar, accent_latent_space_mu, accent_latent_space_logvar))

    def inference(self,x,labels_batch):
        x = x.view(x.size(0),-1)

        x = self.linear_model(x)


        style_latent_space_mu = self.style_mu(x)
        style_latent_space_logvar = self.style_logvar(x)
        
        class_latent_space_mu = self.class_mu(x)
        class_latent_space_logvar = self.class_logvar(x)

        accent_latent_space_mu = class_latent_space_mu.clone()
        accent_latent_space_logvar = class_latent_space_logvar.clone()


        grouped_mu, grouped_logvar = accumulate_group_evidence(class_latent_space_mu.data, class_latent_space_logvar.data, labels_batch,is_cuda=True,device = accent_latent_space_mu.get_device())

        
        style_latent_embeddings = reparameterize(
                training=False, mu=style_latent_space_mu, logvar=style_latent_space_logvar)
        
        accent_latent_embeddings = reparameterize(training=False, mu=accent_latent_space_mu, logvar=accent_latent_space_logvar)
        class_latent_embeddings = group_wise_reparameterize(training=False, mu=grouped_mu, logvar=grouped_logvar, labels_batch=labels_batch, is_cuda=True, device = accent_latent_space_mu.get_device())

        # cat_prob = self.categorical_layer(class_latent_embeddings)

        # acc_prob = self.accent_class(style_latent_embeddings) #acc classifier on spk embs, for adv loss
        # spk_prob = self.speaker_class(class_latent_embeddings) #spk classifier on acc embs,

        # z = torch.cat((style_latent_embeddings, class_latent_embeddings), dim=1)

        return (class_latent_embeddings, style_latent_embeddings, accent_latent_embeddings, (grouped_mu, grouped_logvar, style_latent_space_mu, style_latent_space_logvar, accent_latent_space_mu, accent_latent_space_logvar))

