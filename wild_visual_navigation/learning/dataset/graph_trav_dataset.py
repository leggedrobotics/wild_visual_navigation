# TODO: Jonas adapt GraphTravVisuDataset when storing fixed in create_gnn_dataset

from torch_geometric.data import InMemoryDataset, DataLoader, DataListLoader
from torch_geometric.data import LightningDataset

from wild_visual_navigation import WVN_ROOT_DIR
import os
import torch
from pathlib import Path
from torch_geometric.data import Dataset
from torchvision import transforms as T
from PIL import Image
from typing import Optional, Callable

class GraphTravDataset(InMemoryDataset):
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        mode: str = "train",
        percentage: float = 0.8,
    ):
        super().__init__(root, transform)
        paths = [str(s) for s in Path(os.path.join(root, "graph")).rglob("*.pt")]
        paths.sort()
        if mode == "train":
            paths = paths[: int(len(paths) * percentage)]
        elif mode == "val":
            paths = paths[int(len(paths) * percentage) :]
        else:
            raise ValueError("Mode unknown")

        data_list = [torch.load(p) for p in paths]
        self.data, self.slices = self.collate(data_list)


class GraphTravVisuDataset(Dataset):
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        mode: str = "train",
        percentage: float = 0.8,
    ):
        super().__init__(root, transform)
        paths = [str(s) for s in Path(os.path.join(root, "graph")).rglob("*.pt")]
        paths.sort()
        if mode == "train":
            paths = paths[: int(len(paths) * percentage)]
        elif mode == "val":
            paths = paths[int(len(paths) * percentage) :]
        else:
            raise ValueError("Mode unknown")

        self.paths = paths

        self.crop = T.Compose([T.Resize(448, T.InterpolationMode.NEAREST), T.CenterCrop(448)])

    def len(self) -> int:
        return len(self.paths)

    def get(self, idx: int) -> any:
        # TODO update the dataset generation to avoid 0,0 and the cropping operation
        graph = torch.load(self.paths[idx])
        center = torch.load(self.paths[idx].replace("graph", "center"))
        img = self.crop(torch.load(self.paths[idx].replace("graph", "img")))
        seg = torch.load(self.paths[idx].replace("graph", "seg"))
        return graph, center, img, seg


class GraphTravOnlineDataset(InMemoryDataset):
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        mode="train",
        percentage=0.8,
    ):
        super().__init__(root, transform)
        self.data_list = []
        self.paths = []

    def add(self, data, index):
        # Add new data point to the list
        self.data_list.append(data)

    def clear(self):
        self.data_list.clear()

    def set_ready(self):
        # Collate the data
        self.data, self.slices = self.collate(self.data_list)


def get_pl_graph_trav_module(
    batch_size: int = 1, num_workers: int = 0, visu: bool = False, **kwargs
) -> LightningDataset:
    root = str(os.path.join(WVN_ROOT_DIR, "results/default_mission"))

    if visu:
        train_dataset = GraphTravVisuDataset(root=root, mode="train")
        val_dataset = GraphTravVisuDataset(root=root, mode="val")
    else:
        train_dataset = GraphTravDataset(root=root, mode="train")
        val_dataset = GraphTravDataset(root=root, mode="val")

    return LightningDataset(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=False,
        *kwargs
    )


if __name__ == "__main__":
    from torch_geometric.data import DataLoader

    root = str(os.path.join(WVN_ROOT_DIR, "results/perugia_forest"))
    dataset = GraphTravDataset(root=root)
    dl = DataListLoader(dataset, batch_size=8)
    datamodule = get_pl_graph_trav_module()
