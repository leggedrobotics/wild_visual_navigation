import itertools
from tqdm import tqdm

import torch
import numpy as np

from wild_visual_navigation.learning.model.rnvp import LinearRNVP

# Training params
BATCH_SIZE = 500
EPOCHS = 1
NUM_WORKERS = 0
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

# Model params
INPUT_DIM = 384
FLOW_N = 10  # Number of affine coupling layers
RNVP_TOPOLOGY = 200  # Size of the hidden layers in each coupling layer

# Dataset paths
POS_DATASET_PATH = f"/home/rschmid/RosBags/all_train/features/pos_feat/pos_feat.pt"

MODEL_NAME = f"all_train_bs={BATCH_SIZE}_eps={EPOCHS}_nf={FLOW_N}_top={RNVP_TOPOLOGY}"

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


def train():
    # Create model
    nf_model = LinearRNVP(input_dim=INPUT_DIM, coupling_topology=[RNVP_TOPOLOGY], flow_n=FLOW_N, batch_norm=True,
                          mask_type='odds', conditioning_size=0,
                          use_permutation=True, single_function=True)

    optimizer = torch.optim.Adam(itertools.chain(nf_model.parameters()), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Set data loader
    pos_dataset = DinoFeatureDataset(POS_DATASET_PATH)
    pos_loader = torch.utils.data.DataLoader(pos_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True)

    # Set train mode
    nf_model = nf_model.to(DEVICE)
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

            nf_model.zero_grad()
            loss.backward()
            optimizer.step()

        print("Loss:", loss.item())

    # Save model every epoch
    torch.save(nf_model.state_dict(), f"/home/rschmid/{MODEL_NAME}.pth")
    print("Finished training")


if __name__ == "__main__":
    train()
