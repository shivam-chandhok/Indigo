import argparse
import os
import torch
import json
import torch.backends.cudnn as cudnn
from torch import distributed as distributed
from data.mat_dataset import MatDataset
from data.domain_dataset import PACSDataset, DomainDataset
from methods_emebd import CuMix
from utils import test
import numpy as np
import pickle
from tqdm import tqdm
import random
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
parser.add_argument('--distill', action='store_true', help='DG setting?')
parser.add_argument('--class_distill', action='store_true', help='DG setting?')
parser.add_argument('--data_root', default='/home/code-base/user_space/pacs', type=str, help='Data root directory')
parser.add_argument('--name', default='test', type=str, help='Name of the experiment (used to store '
                                                           'the logger and the checkpoints)')
parser.add_argument('--runs', default=1, type=int, help='Number of runs per experiment')
parser.add_argument('--log_dir', default='/home/code-base/user_space/cumix/logs', type=str, help='Log directory')
parser.add_argument('--ckpt_dir', default='/home/code-base/user_space/cumix/checkpoints', type=str, help='Checkpoint directory')
parser.add_argument('--config_file', default=None, help='Config file for the method.')
parser.add_argument('--resume', default=None, help='Config file for the method.')
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
        assert args.target in PACS_DOMAINS, \
            args.target + " is not a valid target domain in PACS. Please specify a valid PACS target in " + PACS_DOMAINS.__str__()
        DOMAINS = PACS_DOMAINS
        CLASSES = ['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person']
        dataset = PACSDataset
        input_dim = 512
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
checkpoint_dir = os.path.join(args.ckpt_dir, args.name)
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
    if args.dg and not args.zsl:
        val_datasets = []
        for s in sources:
            val_datasets.append(dataset(args.data_root, s, train=False, validation=True))

    attributes = train_dataset.full_attributes
    seen = train_dataset.seen
    unseen = train_dataset.unseen

    method = CuMix(seen_classes=seen,unseen_classes=unseen,attributes=attributes,configs=configs,zsl_only = not args.dg,
                   dg_only = not args.zsl)
    
    if args.resume is not None:
        file = torch.load(args.resume)
        method.load(file)
        print('checkpoint loaded ', args.resume)

    temp_results = []
    top_sources = 0.
    top_idx=-1
    
#     class_weights = method.train_classifier.module.weight.data.cpu().numpy()
#     print(class_weights.shape)
#     class_weights = method.fit(train_dataset)
#     print(class_weights.shape)

#     class_dict = {}
#     for i,k in enumerate(CLASSES):
#         class_dict[k] = class_weights[i]
#     path = os.path.join(args.data_root, args.name + ".npy")
#     print('Saving at ', path)
#     np.save(path, class_dict, allow_pickle=True)
    
    kmeans = method.cluster(train_dataset)
    print(kmeans.cluster_centers_.shape)
    path = os.path.join(args.data_root, args.name + ".pkl")
    with open(path, "wb") as f:
        pickle.dump(kmeans, f)
