# import the necessary packages
import torch
import os


# define the number of images to generate and interpolate
BASE_DATA = '/home/msarkar/pmangla/user_space/domainnet'
LOG_PATH = '/home/msarkar/pmangla/user_space/cumix/logs'
CKPT_PATH = '/home/msarkar/pmangla/user_space/cumix/checkpoints2'


# define the path to the base output directory
BASE_OUTPUT = "results/plots"

# define the path to the output model output and latent
# space interpolation
SAVE_IMG_PATH = os.path.join(BASE_OUTPUT, "image_samples.png")
