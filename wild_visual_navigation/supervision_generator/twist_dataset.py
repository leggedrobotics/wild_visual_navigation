#
# Copyright (c) 2022-2024, ETH Zurich, Jonas Frey, Matias Mattamala.
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#
from wild_visual_navigation import WVN_ROOT_DIR
import os
import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from typing import Optional


class TwistDataset(Dataset):
    def __init__(
        self,
        root: str,
        current_filename: str,
        desired_filename: str,
        mode: str = "train",
        percentage: float = 0.8,
        seq_size: int = 8,
        velocities: list = ["vx", "vy", "vz", "wx", "wy", "wz"],
        ts_matching_thr: str = "10 ms",
    ):
        """Generates a twist dataset
        The iterator returns a batch of size `seq_size` with sequential measurements

        Args:
            root (str): Root folder to extract the files
            current_filename (str): File with the current twist measurements
            desired_filename (str): File with the desired twist
            mode (str): Mode to load the data 'train' or 'val'
            percentage (float): Percentage for the training set
            seq_size (int): Size of the sequential batch
            velocities (list): List with the velocity components to extract from the dataset
            ts_matching_thr (str): Threshold for timestamp matching between current and desired. Default: '10 ms'

        Returns:
            None
        """

        super().__init__()

        # Using the following as a reference
        # https://stackoverflow.com/questions/34880539/pandas-merging-based-on-a-timestamp-which-do-not-match-exactly/51388559#51388559

        # Read files
        current_path = Path(os.path.join(root, current_filename))
        desired_path = Path(os.path.join(root, desired_filename))

        # Read csv
        current_df = pd.read_csv(current_path, delimiter=",")
        desired_df = pd.read_csv(desired_path, delimiter=",")

        header_rename = {
            "#sec": "sec",
            "vx [m/s]": "vx",
            "vy [m/s]": "vy",
            "vz [m/s]": "vz",
            "wx [rad/s]": "wx",
            "wy [rad/s]": "wy",
            "wz [rad/s]": "wz",
        }
        current_df = current_df.rename(columns=header_rename)
        desired_df = desired_df.rename(columns=header_rename)

        # Convert timestamp to seconds
        current_df["ts"] = current_df["sec"].astype(np.float64) + (
            current_df["nsec"].astype(np.float64) * 1.0e-9
        ).astype(np.float64)
        desired_df["ts"] = desired_df["sec"].astype(np.float64) + (
            desired_df["nsec"].astype(np.float64) * 1.0e-9
        ).astype(np.float64)

        # Assign as index
        current_df.index = pd.to_datetime(current_df["ts"])
        desired_df.index = pd.to_datetime(desired_df["ts"])

        # Find closest samples
        tol = pd.Timedelta(ts_matching_thr)
        merged_df = pd.merge_asof(
            left=current_df,
            right=desired_df,
            right_index=True,
            left_index=True,
            direction="nearest",
            tolerance=tol,
        )
        # Reindex to integer indices
        self.size = merged_df.index.size
        merged_df.index = range(self.size)

        # Make auxiliary lists with the desired velocities and the corresponding suffix
        current_velocities = [f"{v}_x" for v in velocities]
        desired_velocities = [f"{v}_y" for v in velocities]

        # Extract desired velocities
        timestamp = merged_df[["ts_x"]]
        current_df = merged_df[current_velocities]
        desired_df = merged_df[desired_velocities]

        # Make indices depending on the mode (train or eval)
        if mode == "train":
            idx_ini = 0
            idx_end = int(self.size * percentage)
        elif mode == "val":
            idx_ini = int(self.size * percentage)
            idx_end = self.size
        else:
            raise ValueError("Mode unknown")

        # Save sequence size
        self.seq_size = seq_size if seq_size < self.size else self.size

        # Make torch tensors
        # Do not pass the dtype, let torch figure it out
        self.timestamps = torch.tensor(timestamp.iloc[idx_ini:idx_end, :].values)
        self.current_twist = torch.tensor(current_df.iloc[idx_ini:idx_end, :].values).to(torch.float32)
        self.desired_twist = torch.tensor(desired_df.iloc[idx_ini:idx_end, :].values).to(torch.float32)
        self.size = self.current_twist.shape[0]

    def __len__(self):
        return self.size

    def __getitem__(self, idx: int):
        # Minor fix if we exceed the size of the dataset
        if idx + self.seq_size > self.size:
            idx = self.size - self.seq_size
        return (
            self.timestamps[idx : idx + self.seq_size],
            self.current_twist[idx : idx + self.seq_size],
            self.desired_twist[idx : idx + self.seq_size],
        )


class TwistDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root: str,
        current_filename: str,
        desired_filename: str,
        seq_size: int = 16,
        batch_size: int = 32,
    ):
        super().__init__()
        self.root = root
        self.current_filename = current_filename
        self.desired_filename = desired_filename
        self.seq_size = seq_size
        self.batch_size = batch_size
        self.data_train = TwistDataset(
            self.root,
            self.current_filename,
            self.desired_filename,
            mode="train",
            seq_size=self.seq_size,
        )
        self.data_val = TwistDataset(
            self.root,
            self.current_filename,
            self.desired_filename,
            mode="val",
            seq_size=self.seq_size,
        )

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        pass
        # # Assign train/val datasets for use in dataloaders
        # if stage == "fit" or stage is None:
        #     self.data_train = TwistDataset(self.root, self.current_filename, self.desired_filename, mode="train")
        #     self.data_val = TwistDataset(self.root, self.current_filename, self.desired_filename, mode="val")

        # # Assign test dataset for use in dataloader(s)
        # if stage == "test" or stage is None:
        #     self.data_test = TwistDataset(self.root, self.current_filename, self.desired_filename, mode="train", percentage=0.8)

        # if stage == "predict" or stage is None:
        #     self.data_predict = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(
            self.data_train, batch_size=self.batch_size, shuffle=False
        )  # batch size handled by Dataset's __getitem__

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        raise NotImplementedError("Does not support test sets yet")
        # return DataLoader(self.data_test, batch_size=32)

    def predict_dataloader(self):
        raise NotImplementedError("Does not support predict sets yet")
        # return DataLoader(self.data_predict, batch_size=32)


if __name__ == "__main__":
    root = str(os.path.join(WVN_ROOT_DIR, "assets/twist_measurements"))
    current_filename = "current_robot_twist_short.csv"
    desired_filename = "desired_robot_twist_short.csv"
    dataset = TwistDataset(root, current_filename, desired_filename)
    pl_datamodule = TwistDataModule(root, current_filename, desired_filename)
    print(pl_datamodule)
