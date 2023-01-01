import argparse
import os
import torch
import json
import torch.backends.cudnn as cudnn
from torch import distributed as distributed
from data.mat_dataset import MatDataset
from data.domain_dataset import PACSDataset,DomainNetDataset, DomainDataset
from src.methods import CuMix
from src.methods_vit import model_dict, StandardTraining, DistillTraining, CrossClassDistillTraining, LateFusionTraining, MidFusionTraining
from src.utils import test
from CLIP.clip import clip
import numpy as np
import pickle
from tqdm import tqdm
import random
from src.configs import config as cfg
from sklearn.cluster import KMeans
from fastprogress.fastprogress import master_bar, progress_bar
from sys import exit
from torch.utils.data import DataLoader
import wandb
# os.environ['WANDB_SILENT']="true"
# wandb.init(project="MMDG", entity="chandhokshivam2")
# Code!

 # Successful exit

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
set_seed(0)

ZSL_DATASETS = ['CUB', 'FLO', 'SUN', 'AWA1']
PACS_DOMAINS = ['photo', 'art_painting', 'cartoon', 'sketch']
DNET_DOMAINS = ['real', 'quickdraw', 'sketch', 'painting', 'infograph', 'clipart']

parser = argparse.ArgumentParser(description='Zero-shot Learning meets Domain Generalization -- ZSL experiments')
parser.add_argument('--target', default='cub', help='Which experiment to run (e.g. [cub, awa1, flo, sun, all])')
parser.add_argument('--zsl', action='store_true', help='ZSL setting?')
parser.add_argument('--dg', action='store_true', help='DG setting?')
parser.add_argument('--method', default='class_token_distill', help='DG setting?')
parser.add_argument('--dataset', default='pacs', type=str, help='Data root directory')
parser.add_argument('--model', default='vit_small_hybrid', type=str, help='Data root directory', choices=list(model_dict.keys()))
parser.add_argument('--teacher', default='clip', type=str, help='Data root directory', choices=list(model_dict.keys()))
parser.add_argument('--name', default='test', type=str, help='Name of the experiment (used to store '
                                                           'the logger and the checkpoints)')
parser.add_argument('--runs', default=1, type=int, help='Number of runs per experiment')
parser.add_argument('--log_dir', default= cfg.LOG_PATH, type=str, help='Log directory')
parser.add_argument('--ckpt_dir', default= cfg.CKPT_PATH, type=str, help='Checkpoint directory')
parser.add_argument('--config_file', default=None, help='Config file for the method.')

parser.add_argument("--local_rank", type=int, default=0)

args = parser.parse_args()

# distributed.init_process_group(backend='nccl', init_method='env://')
# device_id, device = args.local_rank, torch.device(args.local_rank)
# rank, world_size = distributed.get_rank(), distributed.get_world_size()
# torch.cuda.set_device(device_id)
# world_info = {'world_size': world_size, 'rank': rank, 'device_id': device_id}


# Check association dataset--setting are correct + init remaining stuffs from configs
assert args.dg or args.zsl, "Please specify if you want to benchmark ZSL and/or DG performances"

config_file = args.config_file
with open(config_file) as json_file:
    configs = json.load(json_file)
print(args.config_file, configs)
print(vars(args))
multi_domain = False
input_dim = 2048
configs['freeze_bn'] = False

# Semantic W is used to rescale the principal semantic loss.
# Needed to have the same baseline results as https://github.com/HAHA-DL/Episodic-DG/tree/master/PACS for DG only exps
semantic_w = 1.0


if args.dg:
    target = args.target
    multi_domain = True
    if args.zsl:
        assert args.target in DNET_DOMAINS, \
            args.target + " is not a valid target domain in  DomainNet. Please specify a valid DomainNet target in " + DNET_DOMAINS.__str__()
        DOMAINS = DNET_DOMAINS
        dataset = DomainDataset
    else:
        if args.dataset == 'pacs':
            args.data_root = '/home/code-base/user_space/pacs'
            assert args.target in PACS_DOMAINS, \
                args.target + " is not a valid target domain in PACS. Please specify a valid PACS target in " + PACS_DOMAINS.__str__()
            DOMAINS = PACS_DOMAINS
            dataset = PACSDataset
            input_dim = 192
            semantic_w = 3.0
            configs['freeze_bn']=True
        elif args.dataset == 'domainnet':
            args.data_root = cfg.BASE_DATA
            assert args.target in DNET_DOMAINS, \
                args.target + " is not a valid target domain in PACS. Please specify a valid PACS target in " + PACS_DOMAINS.__str__()
            DOMAINS = DNET_DOMAINS
            dataset = DomainNetDataset
            input_dim = model_dict[args.model][1]
            semantic_w = 3.0
            configs['freeze_bn']=True

    sources = DOMAINS + []
    ns = len(sources)
    sources.remove(target)
    assert len(sources) < ns, 'Something is wrong, no source domains reduction after remove with target.'
else:
    target = args.target.upper()
    assert target in ZSL_DATASETS, \
        args.target + " is not a valid ZSL dataset. Please specify a valid dataset " + ZSL_DATASETS.__str__()
    sources = target
    dataset = MatDataset
    configs['mixup_img_w'] = 0.0
    configs['iters_per_epoch'] = 'max'

configs['input_dim'] = input_dim
configs['semantic_w'] = semantic_w
configs['multi_domain'] = multi_domain

# Init loggers and checkpoints path
log_dir = args.log_dir
checkpoint_dir = os.path.join(args.ckpt_dir, args.dataset, args.name)
cudnn.benchmark = False

exp_name=args.name+'.pkl'


try:
    os.makedirs(log_dir)
except OSError:
    pass

try:
    os.makedirs(checkpoint_dir)
except OSError:
    pass

with open(os.path.join(checkpoint_dir,'config.json'), 'w') as f:
    json.dump(configs, f)

with open(os.path.join(checkpoint_dir,'args.json'), 'w') as f:
    json.dump(vars(args), f)
log_file = os.path.join(checkpoint_dir, exp_name)
if os.path.exists(log_file):
    print("WARNING: Your experiment logger seems to exist. Change the name to avoid unwanted overwriting.")

checkpoint_file = os.path.join(checkpoint_dir, 'runN.pth')
if os.path.exists(checkpoint_file):
    print("WARNING: Your experiment checkpoint seems to exist. Change the name to avoid unwanted overwriting.")

logger = {'results':[], 'config': configs, 'target': target, 'checkpoints':[], 'sem_loss':[[] for _ in range(args.runs)],
          'mimg_loss':[[] for _ in range(args.runs)], 'mfeat_loss':[[] for _ in range(args.runs)]}
results = []
results_top = []
val_datasets = None

# Start experiments loop
for r in range(args.runs):
    print('\nTarget: ' + target + '    run ' +str(r+1) +'/'+str(args.runs))

    # Create datasets
    train_dataset = dataset(args.data_root, sources,train=True)
    test_dataset = dataset(args.data_root, target, train=False)
    if args.dg and not args.zsl and args.dataset == 'pacs':
        val_datasets = []
        for s in sources:
            val_datasets.append(dataset(args.data_root, s, train=False, validation=True))

    attributes = train_dataset.full_attributes
    seen = train_dataset.seen
    unseen = train_dataset.unseen

    # Init method
    if args.method == 'standard':
        teacher =  None
        method = StandardTraining(seen_classes=seen,unseen_classes=unseen,attributes=attributes,configs=configs,zsl_only = not args.dg,
                   dg_only = not args.zsl, model = args.model)
    elif args.method == 'distill':
        assert 'deit' in args.model, "Only DeiT model supported for distillation"
        teacher = StandardTraining(seen_classes=seen,unseen_classes=unseen,attributes=attributes,configs=configs,zsl_only = not args.dg,
                   dg_only = not args.zsl, model = args.teacher)
        ckpt = torch.load('/home/pmangla/user_space/cumix/checkpoints2/domainnet/infograph_clip_vit_b_16/run1.pth', map_location='cpu')
        teacher.load(ckpt)
        teacher1 = lambda x: teacher.train_classifier(teacher.backbone(x))
        # teacher.train_classifier = torch.nn.DataParallel(teacher.train_classifier)
        print('Teacher Loaded')
        method = DistillTraining(seen_classes=seen,unseen_classes=unseen,attributes=attributes,configs=configs,zsl_only = not args.dg,
                   dg_only = not args.zsl, model = args.model, teacher=teacher1)
    elif args.method == 'class_token_distill':
        text_inputs = []
        for c in train_dataset.classes:
    	    for d in train_dataset.domains:
        	    text_inputs += [ f"a photo of {c} in {d} domain"]
        text_inputs = torch.cat([clip.tokenize(c) for c in text_inputs])
        text_features = (clip.load("ViT-B/16", device='cpu')[0]).encode_text(text_inputs)
        method = CrossClassDistillTraining(seen_classes=seen,unseen_classes=unseen,attributes=attributes,configs=configs,zsl_only = not args.dg,
                   dg_only = not args.zsl, model = args.model, teacher = args.teacher, text_features=text_features.detach().cuda())
    elif args.method == 'late_fusion_training':
        text_inputs = []
        for c in train_dataset.classes:
    	    for d in train_dataset.domains:
                text_inputs += [ f"a {d} of {c}"]
        	    # text_inputs += [ f"a photo of {c} in {d} domain"]
        text_inputs = torch.cat([clip.tokenize(c) for c in text_inputs])
        text_features = (clip.load("ViT-B/16", device='cpu')[0]).encode_text(text_inputs)
        method = LateFusionTraining(seen_classes=seen,unseen_classes=unseen,attributes=attributes,configs=configs,zsl_only = not args.dg,
                   dg_only = not args.zsl, model = args.model, teacher = args.teacher, text_features=text_features.detach().cuda())
    elif args.method == 'mid_fusion_training':
        method = MidFusionTraining(seen_classes=seen,unseen_classes=unseen,attributes=attributes,configs=configs,zsl_only = not args.dg,
                   dg_only = not args.zsl, model = args.model, teacher = args.teacher)
    elif args.method == 'cumix':
        method = CuMix(seen_classes=seen,unseen_classes=unseen,attributes=attributes,configs=configs,zsl_only = not args.dg,
                   dg_only = not args.zsl)
    elif args.method == 'clip_zsl_inference':
        print('Starting CLIP Inference')
        model, preprocess = clip.load('ViT-B/16', 'cuda')
        model.eval()
        
        test_dataset_clip = dataset(args.data_root, target, train=False, transformer=preprocess)
        
        print(test_dataset_clip.classes)
        text_inputs = [ f"a photo of {c}" for c in test_dataset_clip.classes]
        # print(text_inputs)
        text_inputs = torch.cat([clip.tokenize(c) for c in text_inputs]).cuda()

        mb_clip = master_bar(range(1))
        total, correct = 0, 0
        features = []
        lab = []
        for j in mb:
            for i, (data, _,_, labels) in enumerate(progress_bar(DataLoader(test_dataset_clip, batch_size=240),parent=mb_clip)):
                
                data = data.cuda()
                labels = labels.cuda()
                with torch.no_grad():
                    features.append(model.encode_image(data))
                    
                    lab.append(labels.cpu().numpy())
                    logits_per_image, logits_per_text = model(data, text_inputs)
                preds = torch.argmax(logits_per_image, -1)
                total += data.size(0)
                correct += (preds == labels).sum().item()
                
        print('Accuracy : ', float(correct)/total)
        tsne_features = torch.cat(features)
        flat_list = [item for sublist in lab for item in sublist]
        tsne_labels = [str(i) for i in flat_list]
        tsne_table = wandb.Table(
        columns = [str(i) for i in range(512)], 
        data    = tsne_features.cpu().numpy()
        
    )
        tsne_table.add_column(name = "target", data = tsne_labels)
        wandb.log({
        "embeddings": tsne_table 
            })
            
        exit(0)

    temp_results = []
    top_sources = 0.
    top_target = 0.
    top_idx=-1

    # Strat training loop
    mb = master_bar(range(0, configs['epochs']))

    for e in mb:
        better = False
        semantic_loss, mimg_loss, mfeat_loss = method.fit(mb, train_dataset)
        accuracy = test(method, test_dataset, zsl=args.zsl)
        print(accuracy)

        # In case of DG only, perform validation on source domains, as in https://arxiv.org/pdf/1710.03077.pdf
        if val_datasets is not None:
            acc_sources = 0.
            for v in val_datasets:
                acc_sources += test(method, v, 'cuda' , zsl=False)
            acc_sources /= len(sources)
            if acc_sources >= top_sources:
                if accuracy > top_target:
                    better = True
                    temp_results = accuracy
                    top_target = accuracy
                top_sources = acc_sources
            print('Validation Accuracy : ',acc_sources, ' Test Accuracy : ',accuracy)
        else:
            if accuracy > top_target:
                better = True
                temp_results = accuracy
                top_target = accuracy
            print(' Test Accuracy : ',accuracy)

        if better:
            checkpoint_dict = {}
            method.save(checkpoint_dict)
            current_checkpoint_name = checkpoint_file.replace('runN.pth','run'+str(r+1)+'.pth')
            print('Saving checkpoint at ', current_checkpoint_name)
            torch.save(checkpoint_dict, current_checkpoint_name)
            better = False
        
        # Store losses
        logger['sem_loss'][r].append(semantic_loss)
        logger['mimg_loss'][r].append(mimg_loss)
        logger['mfeat_loss'][r].append(mfeat_loss)

    # Store checkpoints and update logger
#     checkpoint_dict = {}
#     method.save(checkpoint_dict)
#     current_checkpoint_name = checkpoint_file.replace('runN.pth','run'+str(r+1)+'.pth')
#     torch.save(checkpoint_dict, current_checkpoint_name)

    logger['results'].append(temp_results)
    logger['checkpoints'].append(current_checkpoint_name)
    print(target,logger['results'][top_idx])


print('\nResults for ' + target, np.mean(logger['results']),np.std(logger['results']))

with open(log_file, 'wb') as handle:
    pickle.dump(logger, handle, protocol=pickle.HIGHEST_PROTOCOL)

# %%
