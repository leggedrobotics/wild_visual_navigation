# TODO: Jonas adapt GraphTravVisuDataset when storing fixed in create_gnn_dataset

from torch_geometric.data import InMemoryDataset, DataListLoader
from torch_geometric.data import LightningDataset

from wild_visual_navigation import WVN_ROOT_DIR
import os
import torch
from pathlib import Path
from torch_geometric.data import Dataset
from torchvision import transforms as T
from typing import Optional, Callable
from torch_geometric.data import Data


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
        graph.img = img[None]
        graph.center = center
        graph.seg = seg[None]

        graph2 = Data(x=graph.x_previous, edge_index=graph.edge_index_previous)
        return graph, graph2


class GraphTravAbblationDataset(Dataset):
    def __init__(
        self,
        perugia_root: str = "/media/Data/Datasets/2022_Perugia",
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        mode: str = "train",
        feature_key: str = "slic_dino",
        env: str = "hilly"
    ):
        super().__init__()
        
        ls = []
        with open(os.path.join(perugia_root, "wvn_output/split" , "{env}_{mode}.txt"), "r") as f:
            while True:
                line = f.readline()
                if not line:
                    break
                ls.append(line.strip())

        self.paths = ls
        self.perugia_root = perugia_root
        self.feature_key = feature_key
        self.crop = T.Compose([T.Resize(448, T.InterpolationMode.NEAREST), T.CenterCrop(448)])

    def len(self) -> int:
        return len(self.paths)

    def get(self, idx: int) -> any:
        # TODO update the dataset generation to avoid 0,0 and the cropping operation
        img_p = os.path.join(self.perugia_root, self.paths[idx] )
        graph_p = img_p.replace("image", f"features/{self.featue_key}/graph")
        seg_p = img_p.replace("image", f"features/{self.featue_key}/seg")
        center_p = img_p.replace("image", f"features/{self.featue_key}/center")
        
        graph = torch.load( graph_p )
        center = torch.load( center_p )
        img = torch.load( img_p )
        seg = torch.load( seg_p )
        graph.img = img[None]
        graph.center = center
        graph.seg = seg[None]
        
        graph2 = Data()
        return graph, None


def get_abblation_module(
    perugia_root: str,
    batch_size: int = 1,
    num_workers: int = 0,
    visu: bool = False,
    env: str = "forest",
    feature_key: str = "slic_dino",
    **kwargs
) -> LightningDataset:
    
    train_dataset = GraphTravAbblationDataset(root=perugia_root, mode="train", feature_key=feature_key, env= env)
    val_dataset = GraphTravAbblationDataset(root=perugia_root, mode="val", feature_key=feature_key, env= env)
    test_dataset = GraphTravAbblationDataset(root=perugia_root, mode="test", feature_key=feature_key, env= env)
    
    return LightningDataset(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=False,
        *kwargs
    )
    

def get_pl_graph_trav_module(
    batch_size: int = 1,
    num_workers: int = 0,
    visu: bool = False,
    dataset_folder: str = "results/default_mission",
    **kwargs
) -> LightningDataset:

    if os.path.isabs(dataset_folder):
        root = dataset_folder
    else:
        root = str(os.path.join(WVN_ROOT_DIR, dataset_folder))

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
    root = str(os.path.join(WVN_ROOT_DIR, "results/perugia_forest"))
    dataset = GraphTravDataset(root=root)
    dl = DataListLoader(dataset, batch_size=8)
    datamodule = get_pl_graph_trav_module()
