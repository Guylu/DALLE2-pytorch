from torch.optim.lr_scheduler import StepLR
from sklearn.decomposition import PCA
import umap
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
import torch
import warnings
import sys
from scipy.stats import pearsonr
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
import pickle
import wandb
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import time
from sklearn.model_selection import train_test_split
from dalle2_pytorch import DiffusionPriorNetwork, DiffusionPrior

hyperparameter_defaults = dict(
    epochs=200,
    seed=96,
    lr=0.0005,
    batch_size=500,
    optimizer="adam",
    schedualer='cosine',
    first_cycle_steps=1000,
    momentum=0,
    weight_decay=0.1,
    step_size=945,
    gamma=0.99,
    layers=0,
    dropout=0.2,
    nhead=5,
    d_hid=30,
    nlayers=13,
    num=11,  # 12
    mixup=True,
    trial=23,  # [(x, x + 40) for x in range(21, 101, 20)] # 5 - l = 1
    activation='gelu',
    reduce_ag_linear=False,
    pred="lasso",
    regularize=True,
    debug_mode=True,
    # True
)


class dot_dict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def init():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"running on {device}")
    torch.manual_seed(wandb.config.seed)
    torch.cuda.manual_seed(wandb.config.seed)
    torch.cuda.manual_seed_all(wandb.config.seed)
    np.random.seed(wandb.config.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(wandb.config.seed)
    os.environ['PYTHONHASHSEED'] = str(wandb.config.seed)
    return device


wandb.init(config=hyperparameter_defaults, project="GCLIP_Prior", allow_val_change=True)  # , mode="disabled")
# resume=True)  # , mode="disabled")
config = wandb.config
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(wandb.config.seed)
torch.cuda.manual_seed(wandb.config.seed)
torch.cuda.manual_seed_all(wandb.config.seed)
np.random.seed(wandb.config.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(wandb.config.seed)
os.environ['PYTHONHASHSEED'] = str(wandb.config.seed)

plt.switch_backend('agg')
n = 7
wandb.run.name += f'__{n}M_T4-way'

# False True
LOGGING = config.debug_mode
print(f'LOGGING: {LOGGING}')


def get_scheduler(opt):
    if config.schedualer == 'step':
        schedualer = StepLR(opt, step_size=config.step_size, gamma=config.gamma)
    elif config.schedualer == 'cosine':
        schedualer = CosineAnnealingWarmupRestarts(
            opt,
            first_cycle_steps=config.first_cycle_steps,  # config.epochs / 250,
            cycle_mult=1.0,
            max_lr=config.lr,
            min_lr=0,
            warmup_steps=245
        )
    else:
        schedualer = None
    return schedualer


def get_optimizer(g_clip):
    # define optimizer
    optimizer = None
    if config.optimizer == "sgd":
        optimizer = torch.optim.SGD(g_clip.parameters(), lr=config.lr, momentum=config.momentum,
                                    weight_decay=config.weight_decay)
    elif config.optimizer == "adam":
        optimizer = torch.optim.Adam(g_clip.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    elif config.optimizer == "adamw":
        optimizer = torch.optim.AdamW(g_clip.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    return optimizer


#######################################################################################################################
#######################################################################################################################
############################################  START OF CODE ###########################################################
#######################################################################################################################
#######################################################################################################################


if __name__ == '__main__':
    # config.debug_mode = True
    device = init()
    # raise Exception("hi")

    print("Let's use", torch.cuda.device_count(), "GPUs!")
    gpu_count = torch.cuda.device_count()

    prior_network = DiffusionPriorNetwork(
        dim=50,
        depth=6,
        dim_head=64,
        heads=8
    ).to("cuda:0")

    # diffusion prior network, which contains the CLIP and network (with transformer) above

    diffusion_prior = DiffusionPrior(
        net=prior_network,
        clip=None,
        image_embed_dim=50,
        timesteps=100,
        cond_drop_prob=0.2,
        condition_on_text_encodings=False,
    ).to("cuda:0")

    # mock data
    a1 = torch.load("../Best_attn_embds_eval2.pt", map_location=torch.device('cpu'))

    inter = a1[0].index.intersection(a1[1].index)
    a1[0] = a1[0].loc[inter]
    a1[1] = a1[1].loc[inter]
    print(f'number of samples is {len(inter)}')

    text_train, text_test, images_train, images_test = train_test_split(a1[0].values,
                                                                        a1[1].values,
                                                                        test_size=0.309, random_state=42)

    # feed text and images into diffusion prior network
    batch_size = config.batch_size

    trainset = torch.utils.data.TensorDataset(torch.tensor(text_train).to("cuda:0"),
                                              torch.tensor(images_train).to("cuda:0"))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, drop_last=True)

    diffusion_prior.to(device)

    opt = get_optimizer(diffusion_prior)

    print("starting to train Genomic CLIP")
    diffusion_prior.train()
    wandb.watch(diffusion_prior)
    schedualer = get_scheduler(opt)

    Epoch = 0
    print(f'Starting at epoch:{Epoch}')

    PRINT_INTERVAL = 5
    print(f'PRINT_INTERVAL: {PRINT_INTERVAL}')
    print(time.strftime('%X %x %Z'))

    diffusion_prior.train()
    diffusion_prior = diffusion_prior.to("cuda:0")
    for epoch in range(Epoch + 1, config.epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            text, images = data

            # zero the parameter gradients
            opt.zero_grad()

            # forward + backward + optimize
            loss = diffusion_prior(text_embed=text, image_embed=images)
            loss.backward()
            opt.step()

            # print statistics
            running_loss += loss.item()
        if epoch % PRINT_INTERVAL == 0:  # print every 2000 mini-batches
            print(f'[{epoch + 1}] loss: {running_loss / PRINT_INTERVAL:.8f}')
            with torch.no_grad():
                loss = diffusion_prior(text_embed=torch.tensor(text_test).cuda(),
                                       image_embed=torch.tensor(images_test).cuda())
                print(f'        [{epoch + 1}] Test loss: {loss:.8f}')
            wandb.log({"train loss": running_loss / PRINT_INTERVAL})
            wandb.log({"test loss": loss})
            running_loss = 0.0
        wandb.log({"epoch": epoch})
        schedualer.step()
    print('Finished Training')
    diffusion_prior.eval()
    with torch.no_grad():
        loss = diffusion_prior(text_embed=torch.tensor(text_test).cuda(),
                               image_embed=torch.tensor(images_test).cuda())
    print(loss)

    text_embed = torch.tensor(text_test).to("cuda:1")
    image_embed = torch.tensor(images_test).cuda()

    pred = diffusion_prior.to("cuda:1").sample(text_embed, num_samples_per_batch=3, cond_scale=1).detach().cpu().numpy()

    print("finita la cola!")
