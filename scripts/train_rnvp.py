import os
import sys
import glob
import copy
import argparse
import itertools
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import wandb

import cv2
import torch

from wild_visual_navigation.learning.model.rnvp import LinearRNVP

np.set_printoptions(threshold=sys.maxsize)

# Settings
WANDB_LOGGING = False
PLOT_LOSS_LABEL = False
PLOT_ONLY = False
PLOT_MEAN = True

# Training params
BATCH_SIZE = 500
EPOCHS = 10
NUM_WORKERS = 0
NUM_SAMPLES = 700  # Number of samples generated during prediction

# Model params
INPUT_DIM = 384  # Dimension of input (DINO = 90 / 384, geom features = 9 / 4)
FLOW_N = 10  # Number of affine coupling layers
RNVP_TOPOLOGY = 200  # Size of the hidden layers in each coupling layer
CONDITIONING_SIZE = 0

DATASET = "all"

# Name of the dataset
TRAIN_DATASET = f"{DATASET}_train"
TRAIN_SEG_SIZE = "2000"
TEST_DATASET = f"{DATASET}_train"
TEST_SEG_SIZE = "2000"
EVAL_DATASET = f"{DATASET}_eval"


DATASET = TRAIN_DATASET
SEG_SIZE = TRAIN_SEG_SIZE


# Dataset paths
POS_DATASET_PATH = f"/home/rschmid/RosBags/{DATASET}/features/pos_feat/pos_feat.pt"
ALL_DATASET_PATH = f"/home/rschmid/RosBags/{DATASET}/features/all_feat"
SEG_PATH = f"/home/rschmid/RosBags/{DATASET}/features/seg"
CENTER_PATH = f"/home/rschmid/RosBags/{DATASET}/features/center"
IMG_PATH = f"/home/rschmid/RosBags/{DATASET}/image"

MODEL_NAME = f"{TRAIN_DATASET}_{TRAIN_SEG_SIZE}_bs={BATCH_SIZE}_eps={EPOCHS}_nf={FLOW_N}_top={RNVP_TOPOLOGY}"

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class DinoFeatureDataset(torch.utils.data.Dataset):
    def __init__(self, feature_file):
        self.feature_file = feature_file

        feat = np.array(torch.load(feature_file))

        self.features = torch.Tensor(feat)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx, :]


def train(train_dataset=TRAIN_DATASET, train_seg_size=TRAIN_SEG_SIZE, device=DEVICE, batch_size=BATCH_SIZE, epochs=EPOCHS, flow_n=FLOW_N,
          rnvp_topology=RNVP_TOPOLOGY, input_dim=INPUT_DIM, num_workers=NUM_WORKERS, model_name=MODEL_NAME,
          conditioning_size=CONDITIONING_SIZE,
          pos_dataset_path=POS_DATASET_PATH):
    print(f"Train dataset: {train_dataset}, Train seg size: {train_seg_size}, Device: {device}, Batchsize: {batch_size}, "
          f"Epochs: {epochs}, FLOW Number: {flow_n}, RNVP_TOPOLOGY: {rnvp_topology}")

    # Create model
    nf_model = LinearRNVP(input_dim=input_dim, coupling_topology=[rnvp_topology], flow_n=flow_n, batch_norm=True,
                               mask_type='odds', conditioning_size=conditioning_size,
                          use_permutation=True, single_function=True)

    optimizer = torch.optim.Adam(itertools.chain(nf_model.parameters()), lr=1e-4, weight_decay=1e-5)
    # optimizer = torch.optim.Adam(itertools.chain(nf_model.parameters()), lr=1e-4)

    # Set data loader
    pos_dataset = DinoFeatureDataset(pos_dataset_path)
    pos_loader = torch.utils.data.DataLoader(pos_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)

    # Set model to train mode
    nf_model = nf_model.to(device)
    nf_model.train()

    for _ in tqdm(range(EPOCHS)):

        for x in tqdm(pos_loader):
            x = x.to(DEVICE).squeeze()

            # Run model
            z, log_det = nf_model.forward(x)

            # Train via maximum likelihood
            prior_logprob = nf_model.logprob(z)
            # Compute the log probability, compute loss over mean of a batch
            loss = -torch.mean(prior_logprob.sum(1) + log_det)

            if WANDB_LOGGING:
                wandb.log({"train_loss": loss.item()})

            nf_model.zero_grad()
            loss.backward()
            optimizer.step()

        print("Loss:", loss.item())

    # Save model every epoch
    torch.save(nf_model.state_dict(), f"/home/rschmid/{model_name}.pth")
    print("Finished training")


if __name__ == "__main__":
    train()
