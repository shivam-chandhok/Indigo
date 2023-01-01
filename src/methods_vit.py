import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
import torch.optim as optim
from data.domain_dataset import DistributedBalancedSampler,BalancedSampler
import src.deit
import timm.models.mycrossvit as crossvit
import src.cait
import src.losses
from src.networks import resnet18,resnet50
from src.sin_models import resnet50_trained_on_SIN, resnet50_trained_on_SIN_and_IN, resnet50_trained_on_SIN_and_IN_then_finetuned_on_IN
from src.dino import dino_small, dino_small_dist
from timm.models.vision_transformer import vit_small_patch16_224, vit_tiny_patch16_224, deit_small_distilled_patch16_224, deit_tiny_distilled_patch16_224, vit_small_patch16_224_in21k
from src.snl import GTransformerEncoder  
from timm.models.vision_transformer_hybrid import vit_tiny_r_s16_p8_224, vit_small_r26_s32_224, MyHybridEmbed, HybridEmbed
from CLIP.clip import clip
from functools import partial
from fastprogress.fastprogress import master_bar, progress_bar



model_dict = {
    'resnet50': (resnet50, 2048),
    'resnet18': (resnet18, 512),
    'resnet50_trained_on_SIN': (lambda pretrained: resnet50_trained_on_SIN(), 2048),
    'vit_small': (vit_small_patch16_224, 384),
    'deit_small': (deit_small_distilled_patch16_224, 384),
    'deit_tiny': (deit_tiny_distilled_patch16_224, 192),
    'vit_small_hybrid': (vit_small_r26_s32_224, 384),
    'dino_small_sin': (dino_small, 384),
    'clip_vit_b' : (lambda pretrained: ((clip.load("ViT-B/16", device='cuda')[0]).visual).float(), 512),
    'virtex_r50': (lambda pretrained: torch.hub.load("kdexd/virtex", "resnet50", pretrained=pretrained), 512)
}

def imgnet_unormalize(x):
    mean = torch.tensor([-0.485/0.229, -0.456/0.224, -0.406/0.225]).cuda()
    std = torch.tensor([1/0.229, 1/0.224, 1/0.225]).cuda()
    mean = mean[None, :, None, None]
    std = std[None, :, None, None]
    
    return (x-mean)/std

def clip_normalize(x):
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).cuda()
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).cuda()
    mean = mean[None, :, None, None]
    std = std[None, :, None, None]
    
    return (x-mean)/std

# Init losses and utility functions for mixing samples
CE = nn.CrossEntropyLoss()
RG = np.random.default_rng()

def swap(xs, a, b):
    xs[a], xs[b] = xs[b], xs[a]

def derange(xs):
    x_new = [] + xs
    for a in range(1, len(x_new)):
        b = RG.choice(range(0, a))
        swap(x_new, a, b)
    return x_new

# CE on mixed labels, represented as vectors
def manual_CE(predictions, labels):
    loss = -torch.mean(torch.sum(labels * torch.log_softmax(predictions,dim=1),dim=1))
    return loss

# Standard mix
def std_mix(x,indeces,ratio):
    return ratio*x + (1.-ratio)*x[indeces]

# Init ZSL classifier with normalized embeddings
class UnitClassifier(nn.Module):
    def __init__(self, attributes, classes,device='cuda'):
        super(UnitClassifier, self).__init__()
        print(classes)
        self.fc = nn.Linear(attributes[0].size(0), classes.size(0), bias=False).to(device)

        for i,c in enumerate(classes):
            norm_attributes = attributes[c.item()].to(device)
            norm_attributes/=torch.norm(norm_attributes,2)
            self.fc.weight[i].data[:] = norm_attributes

    def forward(self, x):
        o = self.fc(x)
        return o

class UnitClassifier2(nn.Module):
    def __init__(self, attributes, device='cuda'):
        super(UnitClassifier2, self).__init__()
        self.attributes = attributes
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, x):
        x = x / x.norm(dim=-1, keepdim=True)
        att = self.attributes / self.attributes.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * x @ att.t()
        return logits_per_image

# class UnitClassifier2(nn.Module):
#     def __init__(self, attributes, device='cuda'):
#         super(UnitClassifier2, self).__init__()
#         self.attributes = attributes
#         self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
#         self.fc = nn.Linear(384, 512)

#     def forward(self, x):
#         x = self.fc(x)
#         x = x / x.norm(dim=-1, keepdim=True)
#         att = self.attributes / self.attributes.norm(dim=-1, keepdim=True)

#         # cosine similarity as logits
#         logit_scale = self.logit_scale.exp()
#         logits_per_image = logit_scale * x @ att.t()
#         return logits_per_image

    
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        try:
            m.bias.data.fill_(0)
        except:
            print('bias not present')
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


# Actual method
class StandardTraining:

    # Init following the given config
    def __init__(self, seen_classes, unseen_classes, attributes, configs, zsl_only=False, dg_only=False, model=None):
        self.end_to_end = True
        self.domain_mix = True

        if configs['backbone'] == 'none':
            self.end_to_end = False
            self.backbone = nn.Identity()
            self.lr_net = None
        else:
            self.backbone,_ = model_dict[model]
            self.backbone = self.backbone(pretrained=True)
#             self.backbone = resnet50(pretrained=True)
            # self.backbone = vit_small_r26_s32_224(pretrained=True)
            # self.backbone, self.teacher_preprocess = clip.load("ViT-B/16", device='cuda')
            # self.backbone = (self.backbone.visual).float()
            self.lr_net=configs['lr_net']
            self.backbone.eval()
        
        # Uncomment for freezing training backbone
        # for param in self.backbone.parameters():
        #     param.requires_grad = False

        self.zsl_only = zsl_only
        if self.zsl_only:
            self.domain_mix = False

        self.seen_classes = seen_classes if not dg_only else torch.Tensor([0 for _ in range(seen_classes)])
        self.unseen_classes = unseen_classes
        self.attributes = attributes


        self.device = 'cuda'
        
        
        attSize = 0 if dg_only else self.attributes[0].size(0)
        self.current_epoch = -1
        self.mixup_w = configs['mixup_img_w']
        self.mixup_feat_w = configs['mixup_feat_w']

        self.max_beta = configs['mixup_beta']
        self.mixup_beta = 0.0
        self.mixup_step = configs['mixup_step']

        self.step = configs['step']
        self.batch_size = configs['batch_size']
        self.lr = configs['lr']
        self.nesterov = configs['nesterov']
        self.decay = configs['weight_decay']
        self.freeze_bn = configs['freeze_bn']

        input_dim = configs['input_dim']
        self.semantic_w = configs['semantic_w']

        if dg_only:
            self.semantic_projector = nn.Identity()
            self.train_classifier = nn.Linear(input_dim, unseen_classes)
            self.train_classifier.apply(weights_init)
            self.train_classifier.eval()
            self.final_classifier = self.train_classifier
        else:
            self.semantic_projector = nn.Linear(input_dim, attSize)
            self.semantic_projector.apply(weights_init)
            self.semantic_projector.eval()
            self.train_classifier = UnitClassifier(self.attributes, seen_classes, self.device)
            self.train_classifier.eval()
            self.final_classifier = UnitClassifier(self.attributes, unseen_classes, self.device)
            self.final_classifier.eval()

        if not configs['multi_domain']:
            self.dpb = 1
            self.iters = None
        else:
            self.dpb = configs['domains_per_batch']
            self.iters = configs['iters_per_epoch']


        self.criterion = CE
        self.mixup_criterion = manual_CE
        self.current_epoch = -1
        self.dg_only = dg_only

        self.to(self.device)

    # Create one hot labels
    def create_one_hot(self, y):
        y_onehot = torch.LongTensor(y.size(0), self.seen_classes.size(0)).to(self.device)
        y_onehot.zero_()
        y_onehot.scatter_(1, y.view(-1, 1), 1)
        return y_onehot

    # Utilities for saving/loading/retrieving parameters
    def get_classifier_params(self):
        if self.dg_only:
            return self.train_classifier.parameters()
        return self.semantic_projector.parameters()

    def save(self, dict):
        dict['backbone'] = self.backbone.state_dict()
        dict['semantic_projector'] = self.semantic_projector.state_dict()
        dict['train_classifier'] = self.train_classifier.state_dict()
        dict['final_classifier'] = self.final_classifier.state_dict()
        dict['epoch'] = self.current_epoch

    def load(self, dict):
        self.backbone.load_state_dict(dict['backbone'])
        self.semantic_projector.load_state_dict(dict['semantic_projector'])
        self.train_classifier.load_state_dict(dict['train_classifier'])
        if self.dg_only:
            self.final_classifier = self.train_classifier
        else:
            self.final_classifier.load_state_dict(dict['final_classifier'])
        try:
            self.current_epoch = dict['epoch']
        except:
            self.current_epoch = 0

    def to(self, device):
        self.backbone = self.backbone.to(device)
        self.semantic_projector = self.semantic_projector.to(device)
        self.train_classifier = self.train_classifier.to(device)
        if self.dg_only:
            self.final_classifier = self.train_classifier
        else:
            self.final_classifier = self.final_classifier.to(device)

        self.backbone = nn.DataParallel(self.backbone)
        self.semantic_projector = nn.DataParallel(self.semantic_projector)
        self.train_classifier = nn.DataParallel(self.train_classifier)
        if self.dg_only:
            self.final_classifier = self.train_classifier
        else:
            self.final_classifier = nn.DataParallel(self.final_classifier)

        self.device = device

    # Utilities for going from train to eval mode
    def train(self):
        self.backbone.train()
        self.semantic_projector.train()
        self.train_classifier.train()
        self.final_classifier.train()

    def eval(self):
        self.backbone.eval()
        self.semantic_projector.eval()
        self.final_classifier.eval()
        self.train_classifier.eval()

    def zero_grad(self):
        self.backbone.zero_grad()
        self.semantic_projector.zero_grad()
        self.final_classifier.zero_grad()
        self.train_classifier.zero_grad()


    # Utilities for forward passes
    def predict(self, input, label):
        features = self.backbone(input)
        return self.train_classifier(self.semantic_projector(features.float()))

    def forward(self, input, return_features=False):
        features = self.backbone(input)
        prediction = self.train_classifier(self.semantic_projector(features))
        if return_features:
            return prediction, features
        return prediction

    def forward_features(self,features):
        return self.train_classifier(self.semantic_projector(features))

    # Get indeces of samples to mix, following the sampling distribution of the current epoch, as for the curriculum
    def get_sample_mixup(self, domains):
        # Check how many domains are in each batch (if 1 skip, but we have 1 just in ZSL only exps)
        if self.dpb>1:
            doms = list(range(len(torch.unique(domains))))
            bs = domains.size(0) // len(doms)
            selected = derange(doms)
            permuted_across_dom = torch.cat([(torch.randperm(bs) + selected[i] * bs) for i in range(len(doms))])
            permuted_within_dom = torch.cat([(torch.randperm(bs) + i * bs) for i in range(len(doms))])
            ratio_within_dom = torch.from_numpy(RG.binomial(1, self.mixup_domain, size=domains.size(0)))
            indeces = ratio_within_dom * permuted_within_dom + (1. - ratio_within_dom) * permuted_across_dom
        else:
            indeces = torch.randperm(domains.size(0))
        return indeces.long()

    # Get ratio to perform mixup
    def get_ratio_mixup(self,domains):
        return torch.from_numpy(RG.beta(self.mixup_beta, self.mixup_beta, size=domains.size(0))).float()

    # Get both
    def get_mixup_sample_and_ratio(self,domains):
        return self.get_sample_mixup(domains), self.get_ratio_mixup(domains)

    # Get mixed inputs/labels
    def get_mixed_input_labels(self,input,labels,indeces, ratios,dims=2):
        if dims==4:
            return std_mix(input, indeces, ratios.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)), std_mix(labels, indeces, ratios.unsqueeze(-1))
        else:
            return std_mix(input, indeces, ratios.unsqueeze(-1)), std_mix(labels, indeces, ratios.unsqueeze(-1))

    # Actual training procedure
    def fit(self, mb, data):
        # Update mix related variables, as for the curriculum strategy
        self.current_epoch+=1
        self.mixup_beta = min(self.max_beta,max(self.max_beta*(self.current_epoch)/self.mixup_step,0.1))
        self.mixup_domain = min(1.0, max((self.mixup_step * 2.-self.current_epoch) / self.mixup_step, 0.0))

        # Init dataloaders
        if self.dpb>1:
            dataloader = DataLoader(data, batch_size=self.batch_size, num_workers=8,
                                sampler= BalancedSampler(data,self.batch_size//self.dpb,
                                                                   iters=self.iters,domains_per_batch=self.dpb),
                                drop_last=True)
        else:
            dataloader = DataLoader(data, batch_size=self.batch_size, num_workers=2, drop_last=True, shuffle=True)


        # Init plus update optimizers
        scale_lr = 0.1 ** (self.current_epoch // self.step)

        optimizer_net = None

        if self.end_to_end:
            optimizer_net = optim.SGD(self.backbone.parameters(), lr=self.lr_net * scale_lr, momentum=0.9,
                                      weight_decay=self.decay, nesterov=self.nesterov)
#             optimizer_net = optim.Adam(self.backbone.parameters(), 1e-3)

        optimizer_zsl = optim.SGD(self.get_classifier_params(), lr=self.lr * scale_lr, momentum=0.9,
                                      weight_decay=self.decay, nesterov=self.nesterov)
#         optimizer_zsl = optim.Adam(self.get_classifier_params(), 1e-3)

        # Eventually freeze BN, done it for DG only
        if self.freeze_bn:
            self.eval()
        else:
            self.train()

        self.zero_grad()

        # Init logger values
        sem_loss = 0.
        mimg_loss = 0.
        mfeat_loss = 0.
        
        for i, (inputs, _, domains, labels) in enumerate(progress_bar(dataloader,parent=mb)):

            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            #one_hot_labels = self.create_one_hot(labels)

            # Forward + compute AGG loss
#             inputs = clip_normalize(imgnet_unormalize(inputs))
            features = self.backbone(inputs)
            preds = self.train_classifier(self.semantic_projector(features.float()))
            semantic_loss = self.criterion(preds,labels)
            sem_loss += semantic_loss.item()

            total_loss = semantic_loss

            self.zero_grad()
            total_loss.backward()
            del total_loss


            if optimizer_net is not None:
                optimizer_net.step()
            optimizer_zsl.step()
            mb.child.comment = 'Batch {}/{} | Sem Loss {:.3f}, Mimg_Loss {:.3f}, Mfeat_Loss {:.3f}'.format(i, len(dataloader),sem_loss/(i+1), mimg_loss/(i+1), mfeat_loss/(i+1))
            if i%50==0:
                graphs = [[i, sem_loss/(i+1)]]
                mb.update_graph(graphs, [0,750], [0,10])
                #print('Batch {}/{} | Sem Loss {:.3f}, Mimg_Loss {:.3f}, Mfeat_Loss {:.3f}'.format(i, len(dataloader),sem_loss/(i+1), mimg_loss/(i+1), mfeat_loss/(i+1)))



        self.eval()

        return sem_loss/(i+1), mimg_loss/(i+1), mfeat_loss/(i+1)



# Actual method
class DistillTraining:

    # Init following the given config
    def __init__(self, seen_classes, unseen_classes, attributes, configs, zsl_only=False, dg_only=False, model=None, teacher=None):
        self.end_to_end = True
        self.domain_mix = True
        self.teacher = teacher

        if configs['backbone'] == 'none':
            self.end_to_end = False
            self.backbone = nn.Identity()
            self.lr_net = None
        else:
            self.backbone,_ = model_dict[model]
            self.backbone = self.backbone(pretrained=True)
#             self.backbone = resnet50(pretrained=True)
            # self.backbone = vit_small_r26_s32_224(pretrained=True)
            # self.backbone = deit_small_distilled_patch16_224(pretrained=True)
            # self.backbone, self.preprocess = clip.load("ViT-B/16", device='cuda')
            # self.backbone = (self.backbone.visual).float()
            self.lr_net=configs['lr_net']
            self.backbone.eval()
        

        self.zsl_only = zsl_only
        if self.zsl_only:
            self.domain_mix = False

        self.seen_classes = seen_classes if not dg_only else torch.Tensor([0 for _ in range(seen_classes)])
        self.unseen_classes = unseen_classes
        self.attributes = attributes


        self.device = 'cuda'
        
        attSize = 0 if dg_only else self.attributes[0].size(0)
        self.current_epoch = -1
        self.mixup_w = configs['mixup_img_w']
        self.mixup_feat_w = configs['mixup_feat_w']

        self.max_beta = configs['mixup_beta']
        self.mixup_beta = 0.0
        self.mixup_step = configs['mixup_step']

        self.step = configs['step']
        self.batch_size = configs['batch_size']
        self.lr = configs['lr']
        self.nesterov = configs['nesterov']
        self.decay = configs['weight_decay']
        self.freeze_bn = configs['freeze_bn']

        input_dim = configs['input_dim']
        self.semantic_w = configs['semantic_w']

        if dg_only:
            self.semantic_projector = nn.Linear(input_dim, unseen_classes)
            self.train_classifier = nn.Linear(input_dim, unseen_classes)
            self.train_classifier.apply(weights_init)
            self.semantic_projector.apply(weights_init)
            self.train_classifier.eval()
            self.final_classifier = self.train_classifier
        else:
            self.semantic_projector = nn.Linear(input_dim, attSize)
            self.semantic_projector.apply(weights_init)
            self.semantic_projector.eval()
            self.train_classifier = UnitClassifier(self.attributes, seen_classes, self.device)
            self.train_classifier.eval()
            self.final_classifier = UnitClassifier(self.attributes, unseen_classes, self.device)
            self.final_classifier.eval()

        if not configs['multi_domain']:
            self.dpb = 1
            self.iters = None
        else:
            self.dpb = configs['domains_per_batch']
            self.iters = configs['iters_per_epoch']

        self.criterion = losses.DistillationLoss(CE, teacher, 'soft' , 0.1, 3)
        self.mixup_criterion = manual_CE
        self.current_epoch = -1
        self.dg_only = dg_only

        self.to(self.device)

    # Create one hot labels
    def create_one_hot(self, y):
        y_onehot = torch.LongTensor(y.size(0), self.seen_classes.size(0)).to(self.device)
        y_onehot.zero_()
        y_onehot.scatter_(1, y.view(-1, 1), 1)
        return y_onehot

    # Utilities for saving/loading/retrieving parameters
    def get_classifier_params(self):
        if self.dg_only:
            return list(self.train_classifier.parameters()) + list(self.semantic_projector.parameters())
        return self.semantic_projector.parameters()

    def save(self, dict):
        dict['backbone'] = self.backbone.state_dict()
        dict['semantic_projector'] = self.semantic_projector.state_dict()
        dict['train_classifier'] = self.train_classifier.state_dict()
        dict['final_classifier'] = self.final_classifier.state_dict()
        dict['epoch'] = self.current_epoch

    def load(self, dict):
        self.backbone.load_state_dict(dict['backbone'])
        self.semantic_projector.load_state_dict(dict['semantic_projector'])
        self.train_classifier.load_state_dict(dict['train_classifier'])
        if self.dg_only:
            self.final_classifier = self.train_classifier
        else:
            self.final_classifier.load_state_dict(dict['final_classifier'])
        try:
            self.current_epoch = dict['epoch']
        except:
            self.current_epoch = 0

    def to(self, device):
        self.backbone = self.backbone.to(device)
        self.semantic_projector = self.semantic_projector.to(device)
        self.train_classifier = self.train_classifier.to(device)
        self.distill_classifier = self.distill_classifier.to(device)
        if self.dg_only:
            self.final_classifier = self.train_classifier
        else:
            self.final_classifier = self.final_classifier.to(device)

        self.backbone = nn.DataParallel(self.backbone)
        self.semantic_projector = nn.DataParallel(self.semantic_projector)
        self.train_classifier = nn.DataParallel(self.train_classifier)
        if self.dg_only:
            self.final_classifier = self.train_classifier
        else:
            self.final_classifier = nn.DataParallel(self.final_classifier)

        self.device = device

    # Utilities for going from train to eval mode
    def train(self):
        self.backbone.train()
        self.semantic_projector.train()
        self.train_classifier.train()
        self.final_classifier.train()

    def eval(self):
        self.backbone.eval()
        self.semantic_projector.eval()
        self.final_classifier.eval()
        self.train_classifier.eval()

    def zero_grad(self):
        self.backbone.zero_grad()
        self.semantic_projector.zero_grad()
        self.final_classifier.zero_grad()
        self.train_classifier.zero_grad()


    # Utilities for forward passes
    def predict(self, input, label):
        features,features_dist = self.backbone(input)
        s1 = nn.functional.softmax(self.train_classifier(features.float()),1)
        s2 = nn.functional.softmax(self.semantic_projector(features_dist),1)
        return (s1+s2)/2


    def forward(self, input, return_features=False):
        features = self.backbone(input)
        prediction = self.train_classifier(self.semantic_projector(features))
        if return_features:
            return prediction, features
        return prediction

    def forward_features(self,features):
        return self.train_classifier(self.semantic_projector(features))

    # Get indeces of samples to mix, following the sampling distribution of the current epoch, as for the curriculum
    def get_sample_mixup(self, domains):
        # Check how many domains are in each batch (if 1 skip, but we have 1 just in ZSL only exps)
        if self.dpb>1:
            doms = list(range(len(torch.unique(domains))))
            bs = domains.size(0) // len(doms)
            selected = derange(doms)
            permuted_across_dom = torch.cat([(torch.randperm(bs) + selected[i] * bs) for i in range(len(doms))])
            permuted_within_dom = torch.cat([(torch.randperm(bs) + i * bs) for i in range(len(doms))])
            ratio_within_dom = torch.from_numpy(RG.binomial(1, self.mixup_domain, size=domains.size(0)))
            indeces = ratio_within_dom * permuted_within_dom + (1. - ratio_within_dom) * permuted_across_dom
        else:
            indeces = torch.randperm(domains.size(0))
        return indeces.long()

    # Get ratio to perform mixup
    def get_ratio_mixup(self,domains):
        return torch.from_numpy(RG.beta(self.mixup_beta, self.mixup_beta, size=domains.size(0))).float()

    # Get both
    def get_mixup_sample_and_ratio(self,domains):
        return self.get_sample_mixup(domains), self.get_ratio_mixup(domains)

    # Get mixed inputs/labels
    def get_mixed_input_labels(self,input,labels,indeces, ratios,dims=2):
        if dims==4:
            return std_mix(input, indeces, ratios.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)), std_mix(labels, indeces, ratios.unsqueeze(-1))
        else:
            return std_mix(input, indeces, ratios.unsqueeze(-1)), std_mix(labels, indeces, ratios.unsqueeze(-1))

    # Actual training procedure
    def fit(self, data):
        # Update mix related variables, as for the curriculum strategy
        self.current_epoch+=1
        self.mixup_beta = min(self.max_beta,max(self.max_beta*(self.current_epoch)/self.mixup_step,0.1))
        self.mixup_domain = min(1.0, max((self.mixup_step * 2.-self.current_epoch) / self.mixup_step, 0.0))

        # Init dataloaders
        if self.dpb>1:
            dataloader = DataLoader(data, batch_size=self.batch_size, num_workers=8,
                                sampler= BalancedSampler(data,self.batch_size//self.dpb,
                                                                   iters=self.iters,domains_per_batch=self.dpb),
                                drop_last=True)
        else:
            dataloader = DataLoader(data, batch_size=self.batch_size, num_workers=0,
                                    sampler=Sampler(data,num_replicas=self.world_size,
                                                               rank=self.rank,shuffle=True), drop_last=True)

        # Init plus update optimizers
        scale_lr = 0.1 ** (self.current_epoch // self.step)

        optimizer_net = None

        if self.end_to_end:
            optimizer_net = optim.SGD(self.backbone.parameters(), lr=self.lr_net * scale_lr, momentum=0.9,
                                      weight_decay=self.decay, nesterov=self.nesterov)
#             optimizer_net = optim.Adam(self.backbone.parameters(), 1e-3)

        optimizer_zsl = optim.SGD(self.get_classifier_params(), lr=self.lr * scale_lr, momentum=0.9,
                                      weight_decay=self.decay, nesterov=self.nesterov)
#         optimizer_zsl = optim.Adam(self.get_classifier_params(), 1e-3)

        # Eventually freeze BN, done it for DG only
        if self.freeze_bn:
            self.eval()
        else:
            self.train()

        self.zero_grad()

        # Init logger values
        sem_loss = 0.
        mimg_loss = 0.
        mfeat_loss = 0.

        for i, (inputs, _, domains, labels) in enumerate(dataloader):

            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            #one_hot_labels = self.create_one_hot(labels)

            # Forward + compute AGG loss
#             inputs = clip_normalize(imgnet_unormalize(inputs))
            features, features_dist = self.backbone(inputs)
            preds = self.train_classifier(features)
            preds_dist = self.semantic_projector(features_dist)
            outputs = [preds, preds_dist]
            semantic_loss = self.criterion(inputs,outputs,labels)
            sem_loss += semantic_loss.item()

            total_loss = semantic_loss

            self.zero_grad()
            total_loss.backward()
            del total_loss


            if optimizer_net is not None:
                optimizer_net.step()
            optimizer_zsl.step()
            if i%50==0:
                print('Batch {}/{} | Sem Loss {:.3f}, Mimg_Loss {:.3f}, Mfeat_Loss {:.3f}'.format(i, len(dataloader),sem_loss/(i+1), mimg_loss/(i+1), mfeat_loss/(i+1)))



        self.eval()

        return sem_loss/(i+1), mimg_loss/(i+1), mfeat_loss/(i+1)




    

class CrossClassDistillTraining:

    # Init following the given config
    def __init__(self, seen_classes, unseen_classes, attributes, configs, zsl_only=False, dg_only=False, model=None, teacher=None, text_features=None):
        self.end_to_end = True
        self.domain_mix = True
        self.text_features = text_features

        if configs['backbone'] == 'none':
            self.end_to_end = False
            self.backbone = nn.Identity()
            self.lr_net = None
        else:
            self.backbone = model_dict[model][0]
            self.backbone = self.backbone(pretrained=True)
            self.lr_net=configs['lr_net']
            self.backbone.eval()

        self.teacher_backbone, self.teacher_dim = model_dict[teacher]
        self.teacher_backbone = self.teacher_backbone(pretrained=True)
        self.teacher_backbone.eval()

        self.zsl_only = zsl_only
        if self.zsl_only:
            self.domain_mix = False

        self.seen_classes = torch.Tensor([i for i in range(seen_classes)])
        self.unseen_classes = torch.Tensor([i for i in range(unseen_classes)])
        self.attributes = attributes
        self.att = []
        for i in self.attributes:
            self.att.append(self.attributes[i])
        self.att = torch.stack(self.att, 0)


        self.device = 'cuda'
        
        attSize = self.attributes[0].size(0)
        self.current_epoch = -1
        self.mixup_w = configs['mixup_img_w']
        self.mixup_feat_w = configs['mixup_feat_w']

        self.max_beta = configs['mixup_beta']
        self.mixup_beta = 0.0
        self.mixup_step = configs['mixup_step']

        self.step = configs['step']
        self.batch_size = configs['batch_size']
        self.lr = configs['lr']
        self.nesterov = configs['nesterov']
        self.decay = configs['weight_decay']
        self.freeze_bn = configs['freeze_bn']

        input_dim = configs['input_dim']
        self.semantic_w = configs['semantic_w']

        if dg_only:
            self.semantic_projector = nn.Sequential(nn.Linear(self.teacher_dim, input_dim), nn.LayerNorm(input_dim))
            # self.semantic_projector = nn.Linear(input_dim, attSize)
            self.semantic_projector.apply(weights_init)
            self.semantic_projector.eval()
            
            self.train_classifier = nn.Linear(input_dim, unseen_classes)
            self.train_classifier.apply(weights_init)
            self.train_classifier.eval()
            self.final_classifier = self.train_classifier
            
            self.distill_classifier = UnitClassifier2(self.text_features)
            self.distill_classifier.eval()
        else:
            # self.semantic_projector = nn.Linear(input_dim, attSize)
            self.semantic_projector = nn.Sequential(nn.Linear(2048, input_dim), nn.LayerNorm(input_dim))
            self.semantic_projector.apply(weights_init)
            self.semantic_projector.eval()
            self.train_classifier = UnitClassifier(self.attributes, self.seen_classes, self.device)
            self.train_classifier.eval()
            self.final_classifier = UnitClassifier(self.attributes, self.unseen_classes, self.device)
            self.final_classifier.eval()

        if not configs['multi_domain']:
            self.dpb = 1
            self.iters = None
        else:
            self.dpb = configs['domains_per_batch']
            self.iters = configs['iters_per_epoch']

        self.criterion = CE
        self.mixup_criterion = manual_CE
        self.current_epoch = -1
        self.dg_only = dg_only

        self.to(self.device)

    # Create one hot labels
    def create_one_hot(self, y):
        y_onehot = torch.LongTensor(y.size(0), self.seen_classes.size(0)).to(self.device)
        y_onehot.zero_()
        y_onehot.scatter_(1, y.view(-1, 1), 1)
        return y_onehot

    # Utilities for saving/loading/retrieving parameters
    def get_classifier_params(self):
        if self.dg_only:
            return list(self.train_classifier.parameters()) + list(self.semantic_projector.parameters()) + list(self.distill_classifier.parameters())
        return self.semantic_projector.parameters()

    def save(self, dict):
        dict['backbone'] = self.backbone.state_dict()
        dict['semantic_projector'] = self.semantic_projector.state_dict()
        dict['train_classifier'] = self.train_classifier.state_dict()
        dict['final_classifier'] = self.final_classifier.state_dict()
        dict['epoch'] = self.current_epoch

    def load(self, dict):
        self.backbone.load_state_dict(dict['backbone'])
        self.semantic_projector.load_state_dict(dict['semantic_projector'])
        self.train_classifier.load_state_dict(dict['train_classifier'])
        if self.dg_only:
            self.final_classifier = self.train_classifier
        else:
            self.final_classifier.load_state_dict(dict['final_classifier'])
        try:
            self.current_epoch = dict['epoch']
        except:
            self.current_epoch = 0

    def to(self, device):
        self.backbone = self.backbone.to(device)
        self.teacher_backbone  =self.teacher_backbone.to(device)
        self.semantic_projector = self.semantic_projector.to(device)
        self.train_classifier = self.train_classifier.to(device)
        self.distill_classifier = self.distill_classifier.to(device)
        if self.dg_only:
            self.final_classifier = self.train_classifier
        else:
            self.final_classifier = self.final_classifier.to(device)

        self.backbone = nn.DataParallel(self.backbone)
        self.teacher_backbone = nn.DataParallel(self.teacher_backbone)
        self.semantic_projector = nn.DataParallel(self.semantic_projector)
        self.train_classifier = nn.DataParallel(self.train_classifier)
        if self.dg_only:
            self.final_classifier = self.train_classifier
        else:
            self.final_classifier = nn.DataParallel(self.final_classifier)

        self.device = device

    # Utilities for going from train to eval mode
    def train(self):
        self.backbone.train()
        self.semantic_projector.train()
        self.train_classifier.train()
        self.final_classifier.train()

    def eval(self):
        self.backbone.eval()
        self.semantic_projector.eval()
        self.final_classifier.eval()
        self.train_classifier.eval()

    def zero_grad(self):
        self.backbone.zero_grad()
        self.semantic_projector.zero_grad()
        self.final_classifier.zero_grad()
        self.train_classifier.zero_grad()


    # Utilities for forward passes
    def predict(self, input, label, mask=None):
        with torch.no_grad():
            teacher_feats  = self.teacher_backbone(clip_normalize(imgnet_unormalize(input)))
        # teacher_feats = teacher_feats / teacher_feats.norm(dim=-1, keepdim=True)
        teacher_feats = self.semantic_projector(teacher_feats)
        _,features = self.backbone(input, teacher_feats)
        preds = self.train_classifier(features)
        
        return preds

    def forward(self, input, return_features=False):
        features = self.backbone(input)
        prediction = self.train_classifier(self.semantic_projector(features))
        if return_features:
            return prediction, features
        return prediction

    def forward_features(self,features):
        return self.train_classifier(self.semantic_projector(features))

    # Get indeces of samples to mix, following the sampling distribution of the current epoch, as for the curriculum
    def get_sample_mixup(self, domains):
        # Check how many domains are in each batch (if 1 skip, but we have 1 just in ZSL only exps)
        if self.dpb>1:
            doms = list(range(len(torch.unique(domains))))
            bs = domains.size(0) // len(doms)
            selected = derange(doms)
            permuted_across_dom = torch.cat([(torch.randperm(bs) + selected[i] * bs) for i in range(len(doms))])
            permuted_within_dom = torch.cat([(torch.randperm(bs) + i * bs) for i in range(len(doms))])
            ratio_within_dom = torch.from_numpy(RG.binomial(1, self.mixup_domain, size=domains.size(0)))
            indeces = ratio_within_dom * permuted_within_dom + (1. - ratio_within_dom) * permuted_across_dom
        else:
            indeces = torch.randperm(domains.size(0))
        return indeces.long()

    # Get ratio to perform mixup
    def get_ratio_mixup(self,domains):
        return torch.from_numpy(RG.beta(self.mixup_beta, self.mixup_beta, size=domains.size(0))).float()

    # Get both
    def get_mixup_sample_and_ratio(self,domains):
        return self.get_sample_mixup(domains), self.get_ratio_mixup(domains)

    # Get mixed inputs/labels
    def get_mixed_input_labels(self,input,labels,indeces, ratios,dims=2):
        if dims==4:
            return std_mix(input, indeces, ratios.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)), std_mix(labels, indeces, ratios.unsqueeze(-1))
        else:
            return std_mix(input, indeces, ratios.unsqueeze(-1)), std_mix(labels, indeces, ratios.unsqueeze(-1))

    # Actual training procedure
    def fit(self, mb, data):
        # Update mix related variables, as for the curriculum strategy
        self.current_epoch+=1
        self.mixup_beta = min(self.max_beta,max(self.max_beta*(self.current_epoch)/self.mixup_step,0.1))
        self.mixup_domain = min(1.0, max((self.mixup_step * 2.-self.current_epoch) / self.mixup_step, 0.0))

        # Init dataloaders
        if self.dpb>1:
            dataloader = DataLoader(data, batch_size=self.batch_size, num_workers=8,
                                sampler= BalancedSampler(data,self.batch_size//self.dpb,
                                                                   iters=self.iters,domains_per_batch=self.dpb),
                                drop_last=True)
        else:
            dataloader = DataLoader(data, batch_size=self.batch_size, num_workers=0, shuffle=True)

        # Init plus update optimizers
        scale_lr = 0.1 ** (self.current_epoch // self.step)

        optimizer_net = None

        if self.end_to_end:
            optimizer_net = optim.SGD(self.backbone.parameters(), lr=self.lr_net * scale_lr, momentum=0.9,
                                      weight_decay=self.decay, nesterov=self.nesterov)
#             optimizer_net = optim.Adam(self.backbone.parameters(), 1e-4)

        optimizer_zsl = optim.SGD(self.get_classifier_params(), lr=self.lr * scale_lr, momentum=0.9,
                                      weight_decay=self.decay, nesterov=self.nesterov)
#         optimizer_zsl = optim.Adam(self.get_classifier_params(), 1e-4)

        # Eventually freeze BN, done it for DG only
        if self.freeze_bn:
            self.eval()
        else:
            self.train()

        self.zero_grad()
#         self.teacher_backbone.eval()

        # Init logger values
        sem_loss = 0.
        mimg_loss = 0.
        mfeat_loss = 0.

        for i, (inputs, _, domains, labels) in enumerate(progress_bar(dataloader,parent=mb)):
    

            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            domains = domains.to(self.device)
            one_hot_labels = self.create_one_hot(labels)

            with torch.no_grad():
                teacher_feats  = self.teacher_backbone(clip_normalize(imgnet_unormalize(inputs)))
            # print(teacher_feats.shape)
            teacher_feats = self.semantic_projector(teacher_feats)

            features_dist, features = self.backbone(inputs, teacher_feats)
            preds = self.train_classifier(features)
            
            semantic_loss = self.criterion(preds,labels)
            sem_loss += semantic_loss.item()

            preds2 = self.distill_classifier(features_dist)
            domain_loss = self.criterion(preds2,labels*5 + domains)
            mimg_loss += domain_loss.item()

            total_loss = 0.5*semantic_loss + 0.5*domain_loss

            
            self.zero_grad()
            total_loss.backward()
            del total_loss


            if optimizer_net is not None:
                optimizer_net.step()
            optimizer_zsl.step()
            mb.child.comment = 'Batch {}/{} | Sem Loss {:.3f}, Mimg_Loss {:.3f}, Mfeat_Loss {:.3f}'.format(i, len(dataloader),sem_loss/(i+1), mimg_loss/(i+1), mfeat_loss/(i+1))



        self.eval()

        return sem_loss/(i+1), mimg_loss/(i+1), mfeat_loss/(i+1)

class DeiTAttention(nn.Module):
    def __init__(self, name = 'deit_tiny', teacher_dim=512, input_dim=384, num_classes = 345, text_features=None):
        super().__init__()
        self.model, self.dim = model_dict[name]
        self.model = self.model(pretrained=True)
        if text_features is not None:
            self.text_classifier = UnitClassifier2(text_features)
        self.teacher_fc = nn.Sequential(nn.Linear(teacher_dim, self.dim), nn.LayerNorm(self.dim))
        self.backbone_fc = nn.Sequential(nn.Linear(input_dim, self.dim), nn.LayerNorm(self.dim))
        self.fc = nn.Linear(self.dim, num_classes)
        self.fc2 = nn.Linear(self.dim, 512)
    
    def forward(self, f_b_, f_t):
        # print(f_b_.shape, f_t.shape)
        f_b = self.backbone_fc(f_b_)
        f_t = self.teacher_fc(f_t)
        x = torch.stack([f_b, f_t], 1)
        cls_token = self.model.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_token, x), dim=1)
        x = self.model.pos_drop(x + self.model.pos_embed[:,:3,:])
        x = self.model.blocks(x)
        x = self.model.norm(x)
        # print(x.shape)
        return self.fc(x[:,2]) , self.text_classifier(self.fc2(x[:,1]))

# class DeiTAttention(nn.Module):
#     def __init__(self, name = 'deit_tiny', teacher_dim=512, input_dim=384, num_classes = 345, text_features=None):
#         super().__init__()
#         self.model, self.dim = model_dict[name]
#         self.model = self.model(pretrained=True)
#         if text_features is not None:
#             self.text_classifier = UnitClassifier2(text_features)
#         self.teacher_fc = nn.Sequential(nn.Linear(768, self.dim), nn.LayerNorm(self.dim))
#         self.backbone_fc = nn.Sequential(nn.Linear(input_dim, self.dim), nn.LayerNorm(self.dim))
#         self.fc = nn.Linear(self.dim, num_classes)
#         self.fc2 = nn.Linear(self.dim, 512)
    
#     def forward(self, f_b_, f_t):
#         f_b = self.backbone_fc(f_b_)
#         f_t = self.teacher_fc(f_t)
#         # print(f_b.shape, f_t.shape)
#         x = torch.cat([f_t, f_b], 1)
#         # cls_token = self.model.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
#         # x = torch.cat((cls_token, x), dim=1)
#         # x = self.model.pos_drop(x + self.model.pos_embed[:,:3,:])
#         x = self.model.blocks(x)
#         x = self.model.norm(x)
#         return self.fc(x[:,0]) , self.text_classifier(self.fc2(0.5*x[:,197] + 0.5*x[:, 198]))


class LateFusionTraining:

    # Init following the given config
    def __init__(self, seen_classes, unseen_classes, attributes, configs, zsl_only=False, dg_only=False, model=None, teacher=None, text_features=None):
        self.end_to_end = True
        self.domain_mix = True

        if configs['backbone'] == 'none':
            self.end_to_end = False
            self.backbone = nn.Identity()
            self.lr_net = None
        else:
            self.backbone = model_dict[model][0]
            self.backbone = self.backbone(pretrained=True)
            self.lr_net=configs['lr_net']
            self.backbone.eval()

        self.teacher_backbone, self.teacher_dim = model_dict[teacher]
        self.teacher_backbone = self.teacher_backbone(pretrained=True)
        self.teacher_backbone.eval()

        self.zsl_only = zsl_only
        if self.zsl_only:
            self.domain_mix = False

        self.seen_classes = torch.Tensor([i for i in range(seen_classes)])
        self.unseen_classes = torch.Tensor([i for i in range(unseen_classes)])
        self.attributes = attributes
        self.att = []
        for i in self.attributes:
            self.att.append(self.attributes[i])
        self.att = torch.stack(self.att, 0)


        self.device = 'cuda'
        
        attSize = self.attributes[0].size(0)
        self.current_epoch = -1
        self.mixup_w = configs['mixup_img_w']
        self.mixup_feat_w = configs['mixup_feat_w']

        self.max_beta = configs['mixup_beta']
        self.mixup_beta = 0.0
        self.mixup_step = configs['mixup_step']

        self.step = configs['step']
        self.batch_size = configs['batch_size']
        self.lr = configs['lr']
        self.nesterov = configs['nesterov']
        self.decay = configs['weight_decay']
        self.freeze_bn = configs['freeze_bn']

        input_dim = configs['input_dim']
        self.semantic_w = configs['semantic_w']

        if dg_only:
            self.semantic_projector = nn.Sequential(nn.Linear(self.teacher_dim, input_dim), nn.LayerNorm(input_dim))
            # self.semantic_projector = nn.Linear(input_dim, attSize)
            self.semantic_projector.apply(weights_init)
            self.semantic_projector.eval()
            
            self.train_classifier = DeiTAttention(input_dim=input_dim, teacher_dim=self.teacher_dim, num_classes=unseen_classes, text_features=text_features)
            # self.train_classifier = nn.Linear(2*input_dim, unseen_classes)
            self.train_classifier.apply(weights_init)
            self.train_classifier.eval()
            self.final_classifier = self.train_classifier
            
            self.distill_classifier = UnitClassifier(self.attributes, self.seen_classes, self.device)
            self.distill_classifier.eval()
        else:
            # self.semantic_projector = nn.Linear(input_dim, attSize)
            self.semantic_projector = nn.Sequential(nn.Linear(2048, input_dim), nn.LayerNorm(input_dim))
            self.semantic_projector.apply(weights_init)
            self.semantic_projector.eval()
            self.train_classifier = UnitClassifier(self.attributes, self.seen_classes, self.device)
            self.train_classifier.eval()
            self.final_classifier = UnitClassifier(self.attributes, self.unseen_classes, self.device)
            self.final_classifier.eval()

        if not configs['multi_domain']:
            self.dpb = 1
            self.iters = None
        else:
            self.dpb = configs['domains_per_batch']
            self.iters = configs['iters_per_epoch']

        self.criterion = CE
        self.mixup_criterion = manual_CE
        self.current_epoch = -1
        self.dg_only = dg_only

        self.to(self.device)

    # Create one hot labels
    def create_one_hot(self, y):
        y_onehot = torch.LongTensor(y.size(0), self.seen_classes.size(0)).to(self.device)
        y_onehot.zero_()
        y_onehot.scatter_(1, y.view(-1, 1), 1)
        return y_onehot

    # Utilities for saving/loading/retrieving parameters
    def get_classifier_params(self):
        if self.dg_only:
            return list(self.train_classifier.parameters()) + list(self.semantic_projector.parameters())
        return self.semantic_projector.parameters()

    def save(self, dict):
        dict['backbone'] = self.backbone.state_dict()
        dict['semantic_projector'] = self.semantic_projector.state_dict()
        dict['train_classifier'] = self.train_classifier.state_dict()
        dict['final_classifier'] = self.final_classifier.state_dict()
        dict['epoch'] = self.current_epoch

    def load(self, dict):
        self.backbone.load_state_dict(dict['backbone'])
        self.semantic_projector.load_state_dict(dict['semantic_projector'])
        self.train_classifier.load_state_dict(dict['train_classifier'])
        if self.dg_only:
            self.final_classifier = self.train_classifier
        else:
            self.final_classifier.load_state_dict(dict['final_classifier'])
        try:
            self.current_epoch = dict['epoch']
        except:
            self.current_epoch = 0

    def to(self, device):
        self.backbone = self.backbone.to(device)
        self.teacher_backbone  =self.teacher_backbone.to(device)
        self.semantic_projector = self.semantic_projector.to(device)
        self.train_classifier = self.train_classifier.to(device)
        if self.dg_only:
            self.final_classifier = self.train_classifier
        else:
            self.final_classifier = self.final_classifier.to(device)

        self.backbone = nn.DataParallel(self.backbone)
        self.teacher_backbone = nn.DataParallel(self.teacher_backbone)
        self.semantic_projector = nn.DataParallel(self.semantic_projector)
        # self.train_classifier = nn.DataParallel(self.train_classifier)
        if self.dg_only:
            self.final_classifier = self.train_classifier
        else:
            self.final_classifier = nn.DataParallel(self.final_classifier)

        self.device = device

    # Utilities for going from train to eval mode
    def train(self):
        self.backbone.train()
        self.semantic_projector.train()
        self.train_classifier.train()
        self.final_classifier.train()

    def eval(self):
        self.backbone.eval()
        self.semantic_projector.eval()
        self.final_classifier.eval()
        self.train_classifier.eval()

    def zero_grad(self):
        self.backbone.zero_grad()
        self.semantic_projector.zero_grad()
        self.final_classifier.zero_grad()
        self.train_classifier.zero_grad()


    # Utilities for forward passes
    def predict(self, input, label, mask=None):
        with torch.no_grad(): 
            teacher_feats  = self.teacher_backbone(clip_normalize(imgnet_unormalize(input)))
        # teacher_feats_ = self.semantic_projector(teacher_feats_)
        features = self.backbone(input)
        preds,_ = self.train_classifier(features, teacher_feats)
        if mask is not None:
            preds = preds[:, mask]
        return preds

    def forward(self, input, return_features=False):
        features = self.backbone(input)
        prediction = self.train_classifier(self.semantic_projector(features))
        if return_features:
            return prediction, features
        return prediction

    def forward_features(self,features):
        return self.train_classifier(self.semantic_projector(features))

    # Get indeces of samples to mix, following the sampling distribution of the current epoch, as for the curriculum
    def get_sample_mixup(self, domains):
        # Check how many domains are in each batch (if 1 skip, but we have 1 just in ZSL only exps)
        if self.dpb>1:
            doms = list(range(len(torch.unique(domains))))
            bs = domains.size(0) // len(doms)
            selected = derange(doms)
            permuted_across_dom = torch.cat([(torch.randperm(bs) + selected[i] * bs) for i in range(len(doms))])
            permuted_within_dom = torch.cat([(torch.randperm(bs) + i * bs) for i in range(len(doms))])
            ratio_within_dom = torch.from_numpy(RG.binomial(1, self.mixup_domain, size=domains.size(0)))
            indeces = ratio_within_dom * permuted_within_dom + (1. - ratio_within_dom) * permuted_across_dom
        else:
            indeces = torch.randperm(domains.size(0))
        return indeces.long()

    # Get ratio to perform mixup
    def get_ratio_mixup(self,domains):
        return torch.from_numpy(RG.beta(self.mixup_beta, self.mixup_beta, size=domains.size(0))).float()

    # Get both
    def get_mixup_sample_and_ratio(self,domains):
        return self.get_sample_mixup(domains), self.get_ratio_mixup(domains)

    # Get mixed inputs/labels
    def get_mixed_input_labels(self,input,labels,indeces, ratios,dims=2):
        if dims==4:
            return std_mix(input, indeces, ratios.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)), std_mix(labels, indeces, ratios.unsqueeze(-1))
        else:
            return std_mix(input, indeces, ratios.unsqueeze(-1)), std_mix(labels, indeces, ratios.unsqueeze(-1))

    # Actual training procedure
    def fit(self, mb, data):
        # Update mix related variables, as for the curriculum strategy
        self.current_epoch+=1
        self.mixup_beta = min(self.max_beta,max(self.max_beta*(self.current_epoch)/self.mixup_step,0.1))
        self.mixup_domain = min(1.0, max((self.mixup_step * 2.-self.current_epoch) / self.mixup_step, 0.0))

        # Init dataloaders
        if self.dpb>1:
            dataloader = DataLoader(data, batch_size=self.batch_size, num_workers=8,
                                sampler= BalancedSampler(data,self.batch_size//self.dpb,
                                                                   iters=self.iters,domains_per_batch=self.dpb),
                                drop_last=True)
        else:
            dataloader = DataLoader(data, batch_size=self.batch_size, num_workers=0, shuffle=True)

        # Init plus update optimizers
        scale_lr = 0.1 ** (self.current_epoch // self.step)

        optimizer_net = None

        if self.end_to_end:
            optimizer_net = optim.SGD(self.backbone.parameters(), lr=self.lr_net * scale_lr, momentum=0.9,
                                      weight_decay=self.decay, nesterov=self.nesterov)

        optimizer_zsl = optim.SGD(self.get_classifier_params(), lr=self.lr * scale_lr, momentum=0.9,
                                      weight_decay=self.decay, nesterov=self.nesterov)

        # Eventually freeze BN, done it for DG only
        if self.freeze_bn:
            self.eval()
        else:
            self.train()

        self.zero_grad()
#         self.teacher_backbone.eval()

        # Init logger values
        sem_loss = 0.
        mimg_loss = 0.
        mfeat_loss = 0.

        
        for i, (inputs, _, domains, labels) in enumerate(progress_bar(dataloader,parent=mb)):

            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            domains = domains.to(self.device)
            targets = labels*5 + domains
            # print(clabels.shape)

            one_hot_labels = self.create_one_hot(labels)
            
            with torch.no_grad():   
                teacher_feats  = self.teacher_backbone(clip_normalize(imgnet_unormalize(inputs)))
            
            # teacher_feats_ = self.semantic_projector(teacher_feats_)
            features = self.backbone(inputs)
            preds, preds2 = self.train_classifier(features, teacher_feats)
            
            semantic_loss =  self.criterion(preds, labels) 
            semantic_loss2 = self.criterion(preds2, targets)
            sem_loss += semantic_loss.item()
            mimg_loss += semantic_loss2.item()

            total_loss = 0.5*semantic_loss + 0.5*semantic_loss2

            
            self.zero_grad()
            total_loss.backward()
            del total_loss


            if optimizer_net is not None:
                optimizer_net.step()
            optimizer_zsl.step()
            mb.child.comment = 'Batch {}/{} | Sem Loss {:.3f}, Mimg_Loss {:.3f}, Mfeat_Loss {:.3f}'.format(i, len(dataloader),sem_loss/(i+1), mimg_loss/(i+1), mfeat_loss/(i+1))
            # if i%50==0:
            #     print('Batch {}/{} | Sem Loss {:.3f}, Mimg_Loss {:.3f}, Mfeat_Loss {:.3f}'.format(i, len(dataloader),sem_loss/(i+1), mimg_loss/(i+1), mfeat_loss/(i+1)))



        self.eval()

        return sem_loss/(i+1), mimg_loss/(i+1), mfeat_loss/(i+1)

class DeiTAttention2(nn.Module):
    def __init__(self, name = 'deit_tiny', teacher_dim=512, input_dim=384, num_classes = 345, text_features=None):
        super().__init__()
        self.model, self.dim = model_dict[name]
        self.model = self.model(pretrained=True)
        self.text_features = text_features
        self.teacher_fc = nn.Sequential(nn.Linear(teacher_dim, self.dim), nn.LayerNorm(self.dim))
        self.backbone_fc = nn.Sequential(nn.Linear(input_dim, self.dim), nn.LayerNorm(self.dim))
        self.fc = nn.Linear(self.dim, num_classes)
        self.fc2 = nn.Linear(input_dim, num_classes)
    
    def forward(self, f_b_, f_t):
        f_b = self.backbone_fc(f_b_)
        f_t = self.teacher_fc(f_t)
        x = torch.stack([f_b, f_t], 1)
        cls_token = self.model.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_token, x), dim=1)
        x = self.model.pos_drop(x + self.model.pos_embed[:,:3,:])
        x = self.model.blocks(x)
        x = self.model.norm(x)
        return self.fc(x[:,1]) 

class MidFusionTraining:

    # Init following the given config
    def __init__(self, seen_classes, unseen_classes, attributes, configs, zsl_only=False, dg_only=False, model=None, teacher=None):
        self.end_to_end = True
        self.domain_mix = True
        

        if configs['backbone'] == 'none':
            self.end_to_end = False
            self.backbone = nn.Identity()
            self.lr_net = None
        else:
            self.backbone = model_dict[model][0]
            self.backbone = self.backbone(pretrained=True)
            self.lr_net=configs['lr_net']
            self.backbone.eval()

        self.teacher_backbone, self.teacher_dim = model_dict[teacher]
        self.teacher_backbone = self.teacher_backbone(pretrained=True)
        self.teacher_backbone.eval()

        self.zsl_only = zsl_only
        if self.zsl_only:
            self.domain_mix = False

        self.seen_classes = torch.Tensor([i for i in range(seen_classes)])
        self.unseen_classes = torch.Tensor([i for i in range(unseen_classes)])
        self.attributes = attributes
        self.att = []
        for i in self.attributes:
            self.att.append(self.attributes[i])
        self.att = torch.stack(self.att, 0)


        self.device = 'cuda'
        
        attSize = self.attributes[0].size(0)
        self.current_epoch = -1
        self.mixup_w = configs['mixup_img_w']
        self.mixup_feat_w = configs['mixup_feat_w']

        self.max_beta = configs['mixup_beta']
        self.mixup_beta = 0.0
        self.mixup_step = configs['mixup_step']

        self.step = configs['step']
        self.batch_size = configs['batch_size']
        self.lr = configs['lr']
        self.nesterov = configs['nesterov']
        self.decay = configs['weight_decay']
        self.freeze_bn = configs['freeze_bn']

        input_dim = configs['input_dim']
        self.semantic_w = configs['semantic_w']

        if dg_only:
            self.semantic_projector = nn.Sequential(nn.Linear(768, input_dim), nn.LayerNorm(input_dim))
            # self.semantic_projector = nn.Linear(input_dim, attSize)
            self.semantic_projector.apply(weights_init)
            self.semantic_projector.eval()
            
            self.train_classifier = nn.Linear(input_dim, unseen_classes)
            # self.train_classifier = DeiTAttention2(input_dim=input_dim, teacher_dim=input_dim, num_classes=unseen_classes)
            # self.train_classifier.apply(weights_init)
            self.train_classifier.eval()
            self.final_classifier = self.train_classifier
            
            self.distill_classifier = UnitClassifier(self.attributes, self.seen_classes, self.device)
            self.distill_classifier.eval()
        else:
            # self.semantic_projector = nn.Linear(input_dim, attSize)
            self.semantic_projector = nn.Sequential(nn.Linear(2048, input_dim), nn.LayerNorm(input_dim))
            self.semantic_projector.apply(weights_init)
            self.semantic_projector.eval()
            self.train_classifier = UnitClassifier(self.attributes, self.seen_classes, self.device)
            self.train_classifier.eval()
            self.final_classifier = UnitClassifier(self.attributes, self.unseen_classes, self.device)
            self.final_classifier.eval()

        if not configs['multi_domain']:
            self.dpb = 1
            self.iters = None
        else:
            self.dpb = configs['domains_per_batch']
            self.iters = configs['iters_per_epoch']

        self.criterion = CE
        self.mixup_criterion = manual_CE
        self.current_epoch = -1
        self.dg_only = dg_only

        self.to(self.device)

    # Create one hot labels
    def create_one_hot(self, y):
        y_onehot = torch.LongTensor(y.size(0), self.seen_classes.size(0)).to(self.device)
        y_onehot.zero_()
        y_onehot.scatter_(1, y.view(-1, 1), 1)
        return y_onehot

    # Utilities for saving/loading/retrieving parameters
    def get_classifier_params(self):
        if self.dg_only:
            return list(self.train_classifier.parameters()) + list(self.semantic_projector.parameters())
        return self.semantic_projector.parameters()

    def save(self, dict):
        dict['backbone'] = self.backbone.state_dict()
        dict['semantic_projector'] = self.semantic_projector.state_dict()
        dict['train_classifier'] = self.train_classifier.state_dict()
        dict['final_classifier'] = self.final_classifier.state_dict()
        dict['epoch'] = self.current_epoch

    def load(self, dict):
        self.backbone.load_state_dict(dict['backbone'])
        self.semantic_projector.load_state_dict(dict['semantic_projector'])
        self.train_classifier.load_state_dict(dict['train_classifier'])
        if self.dg_only:
            self.final_classifier = self.train_classifier
        else:
            self.final_classifier.load_state_dict(dict['final_classifier'])
        try:
            self.current_epoch = dict['epoch']
        except:
            self.current_epoch = 0

    def to(self, device):
        self.backbone = self.backbone.to(device)
        self.teacher_backbone  =self.teacher_backbone.to(device)
        self.semantic_projector = self.semantic_projector.to(device)
        self.train_classifier = self.train_classifier.to(device)
        if self.dg_only:
            self.final_classifier = self.train_classifier
        else:
            self.final_classifier = self.final_classifier.to(device)

        self.backbone = nn.DataParallel(self.backbone)
        self.teacher_backbone = nn.DataParallel(self.teacher_backbone)
        self.semantic_projector = nn.DataParallel(self.semantic_projector)
        self.train_classifier = nn.DataParallel(self.train_classifier)
        if self.dg_only:
            self.final_classifier = self.train_classifier
        else:
            self.final_classifier = nn.DataParallel(self.final_classifier)

        self.device = device

    # Utilities for going from train to eval mode
    def train(self):
        self.backbone.train()
        self.semantic_projector.train()
        self.train_classifier.train()
        self.final_classifier.train()

    def eval(self):
        self.backbone.eval()
        self.semantic_projector.eval()
        self.final_classifier.eval()
        self.train_classifier.eval()

    def zero_grad(self):
        self.backbone.zero_grad()
        self.semantic_projector.zero_grad()
        self.final_classifier.zero_grad()
        self.train_classifier.zero_grad()


    # Utilities for forward passes
    def predict(self, input, label, mask=None):
        with torch.no_grad():
            teacher_feats  = self.teacher_backbone(clip_normalize(imgnet_unormalize(input)), False)
        # teacher_feats = teacher_feats / teacher_feats.norm(dim=-1, keepdim=True)
        teacher_feats = self.semantic_projector(teacher_feats)
        features = self.backbone(input, teacher_feats, 2)
        preds = self.train_classifier(features)
        
        return preds

    def forward(self, input, return_features=False):
        features = self.backbone(input)
        prediction = self.train_classifier(self.semantic_projector(features))
        if return_features:
            return prediction, features
        return prediction

    def forward_features(self,features):
        return self.train_classifier(self.semantic_projector(features))

    # Get indeces of samples to mix, following the sampling distribution of the current epoch, as for the curriculum
    def get_sample_mixup(self, domains):
        # Check how many domains are in each batch (if 1 skip, but we have 1 just in ZSL only exps)
        if self.dpb>1:
            doms = list(range(len(torch.unique(domains))))
            bs = domains.size(0) // len(doms)
            selected = derange(doms)
            permuted_across_dom = torch.cat([(torch.randperm(bs) + selected[i] * bs) for i in range(len(doms))])
            permuted_within_dom = torch.cat([(torch.randperm(bs) + i * bs) for i in range(len(doms))])
            ratio_within_dom = torch.from_numpy(RG.binomial(1, self.mixup_domain, size=domains.size(0)))
            indeces = ratio_within_dom * permuted_within_dom + (1. - ratio_within_dom) * permuted_across_dom
        else:
            indeces = torch.randperm(domains.size(0))
        return indeces.long()

    # Get ratio to perform mixup
    def get_ratio_mixup(self,domains):
        return torch.from_numpy(RG.beta(self.mixup_beta, self.mixup_beta, size=domains.size(0))).float()

    # Get both
    def get_mixup_sample_and_ratio(self,domains):
        return self.get_sample_mixup(domains), self.get_ratio_mixup(domains)

    # Get mixed inputs/labels
    def get_mixed_input_labels(self,input,labels,indeces, ratios,dims=2):
        if dims==4:
            return std_mix(input, indeces, ratios.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)), std_mix(labels, indeces, ratios.unsqueeze(-1))
        else:
            return std_mix(input, indeces, ratios.unsqueeze(-1)), std_mix(labels, indeces, ratios.unsqueeze(-1))

    # Actual training procedure
    def fit(self, mb, data):
        # Update mix related variables, as for the curriculum strategy
        self.current_epoch+=1
        self.mixup_beta = min(self.max_beta,max(self.max_beta*(self.current_epoch)/self.mixup_step,0.1))
        self.mixup_domain = min(1.0, max((self.mixup_step * 2.-self.current_epoch) / self.mixup_step, 0.0))

        # Init dataloaders
        if self.dpb>1:
            dataloader = DataLoader(data, batch_size=self.batch_size, num_workers=8,
                                sampler= BalancedSampler(data,self.batch_size//self.dpb,
                                                                   iters=self.iters,domains_per_batch=self.dpb),
                                drop_last=True)
        else:
            dataloader = DataLoader(data, batch_size=self.batch_size, num_workers=0, shuffle=True)

        # Init plus update optimizers
        scale_lr = 0.1 ** (self.current_epoch // self.step)

        optimizer_net = None

        if self.end_to_end:
            optimizer_net = optim.SGD(self.backbone.parameters(), lr=self.lr_net * scale_lr, momentum=0.9,
                                      weight_decay=self.decay, nesterov=self.nesterov)
#             optimizer_net = optim.Adam(self.backbone.parameters(), 1e-4)

        optimizer_zsl = optim.SGD(self.get_classifier_params(), lr=self.lr * scale_lr, momentum=0.9,
                                      weight_decay=self.decay, nesterov=self.nesterov)
#         optimizer_zsl = optim.Adam(self.get_classifier_params(), 1e-4)

        # Eventually freeze BN, done it for DG only
        if self.freeze_bn:
            self.eval()
        else:
            self.train()

        self.zero_grad()
#         self.teacher_backbone.eval()

        # Init logger values
        sem_loss = 0.
        mimg_loss = 0.
        mfeat_loss = 0.

        for i, (inputs, _, domains, labels) in enumerate(progress_bar(dataloader,parent=mb)):
    

            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            one_hot_labels = self.create_one_hot(labels)

            with torch.no_grad():
                teacher_feats  = self.teacher_backbone(clip_normalize(imgnet_unormalize(inputs)), False)
            # print(teacher_feats.shape)
            teacher_feats = self.semantic_projector(teacher_feats)
            features = self.backbone(inputs, teacher_feats, 2)
            preds = self.train_classifier(features)
            
            semantic_loss = self.criterion(preds,labels)
            sem_loss += semantic_loss.item()

            total_loss = semantic_loss

            
            self.zero_grad()
            total_loss.backward()
            del total_loss


            if optimizer_net is not None:
                optimizer_net.step()
            optimizer_zsl.step()
            mb.child.comment = 'Batch {}/{} | Sem Loss {:.3f}, Mimg_Loss {:.3f}, Mfeat_Loss {:.3f}'.format(i, len(dataloader),sem_loss/(i+1), mimg_loss/(i+1), mfeat_loss/(i+1))

            #if i%50==0:
                #print('Batch {}/{} | Sem Loss {:.3f}, Mimg_Loss {:.3f}, Mfeat_Loss {:.3f}'.format(i, len(dataloader),sem_loss/(i+1), mimg_loss/(i+1), mfeat_loss/(i+1)))



        self.eval()

        return sem_loss/(i+1), mimg_loss/(i+1), mfeat_loss/(i+1)