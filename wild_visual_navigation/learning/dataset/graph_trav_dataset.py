from torch_geometric.data import InMemoryDataset, DataListLoader, DataLoader
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


class GraphTravAbblationDataset(Dataset):
    def __init__(
        self,
        perugia_root: str = "/media/Data/Datasets/2022_Perugia",
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        mode: str = "train",
        feature_key: str = "slic_dino",
        env: str = "hilly",
        use_corrospondences: bool = True,
        training_data_percentage: int = 100,
    ):
        super().__init__()

        ls = []
        j = 0
        self.perugia_root = perugia_root

        with open(os.path.join(perugia_root, "wvn_output/split", f"{env}_{mode}.txt"), "r") as f:
            while True:
                line = f.readline()
                j += 1
                if not line:
                    break
                ls.append(line.strip())

                p = line.strip()
                img_p = os.path.join(self.perugia_root, p)

                if not os.path.exists(img_p):
                    print("Not found path", img_p)

        if training_data_percentage < 100:
            if int(len(ls) * training_data_percentage / 100) == 0:
                raise Exception("Defined Training Data Perentage to small !")
            ls = ls[: int(len(ls) * training_data_percentage / 100)]

        self.mode = mode
        self.env = env
        self.paths = ls
        self.feature_key = feature_key
        self.crop = T.Compose([T.Resize(448, T.InterpolationMode.NEAREST), T.CenterCrop(448)])
        self.use_corrospondences = use_corrospondences

    def len(self) -> int:
        return len(self.paths)

    def get(self, idx: int) -> any:
        # TODO update the dataset generation to avoid 0,0 and the cropping operation
        img_p = os.path.join(self.perugia_root, self.paths[idx])
        graph_p = img_p.replace("image", f"features/{self.feature_key}/graph")
        seg_p = img_p.replace("image", f"features/{self.feature_key}/seg")
        center_p = img_p.replace("image", f"features/{self.feature_key}/center")

        img = torch.load(img_p)
        graph = torch.load(graph_p)
        center = torch.load(center_p)
        seg = torch.load(seg_p)
        graph.img = img[None]
        graph.center = center
        graph.seg = seg[None]
        graph.y = (graph.y > 0).type(torch.float32)  # make now binary

        if self.mode == "test":
            key = (self.paths[idx]).split("/")[-1][:-3]
            store = os.path.join(self.perugia_root, f"wvn_output/labeling/{self.env}/labels/{key}.pt")
            label = torch.load(store)

            y_gt = []
            for i in torch.unique(seg):
                m = label[seg == i]
                pos = m.sum()
                neg = (~m).sum()
                y_gt.append(pos < neg)

            graph.y_gt = torch.stack(y_gt).type(torch.float32)
            graph.label = ~label[None]

        return graph, Data(x=graph.x_previous, edge_index=graph.edge_index_previous)


def get_abblation_module(
    perugia_root: str,
    batch_size: int = 1,
    num_workers: int = 0,
    visu: bool = False,
    env: str = "forest",
    feature_key: str = "slic_dino",
    test_equals_val: bool = False,
    val_equals_test: bool = False,
    **kwargs,
) -> LightningDataset:

    train_dataset = GraphTravAbblationDataset(
        perugia_root=perugia_root,
        mode="train",
        feature_key=feature_key,
        env=env,
        training_data_percentage=kwargs.get("training_data_percentage", 100),
    )
    val_dataset = [GraphTravAbblationDataset(perugia_root=perugia_root, mode="val", feature_key=feature_key, env=env)]

    if test_equals_val:
        test_dataset = [
            GraphTravAbblationDataset(perugia_root=perugia_root, mode="val", feature_key=feature_key, env=env)
        ]
    else:
        test_dataset = [
            GraphTravAbblationDataset(perugia_root=perugia_root, mode="test", feature_key=feature_key, env=env)
        ]

    if kwargs.get("test_all_datasets"):
        test_dataset = [
            GraphTravAbblationDataset(perugia_root=perugia_root, mode="test", feature_key=feature_key, env="forest"),
            GraphTravAbblationDataset(perugia_root=perugia_root, mode="test", feature_key=feature_key, env="grassland"),
            GraphTravAbblationDataset(perugia_root=perugia_root, mode="test", feature_key=feature_key, env="hilly"),
        ]

    if val_equals_test:
        val_dataset = test_dataset

    return (
        DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=False),
        [DataLoader(dataset=v, batch_size=batch_size, num_workers=num_workers, pin_memory=False) for v in val_dataset],
        [DataLoader(dataset=t, batch_size=batch_size, num_workers=num_workers, pin_memory=False) for t in test_dataset],
    )


if __name__ == "__main__":
    root = str(os.path.join(WVN_ROOT_DIR, "results/perugia_forest"))
    dataset = GraphTravDataset(root=root)
    dl = DataListLoader(dataset, batch_size=8)
    datamodule = get_pl_graph_trav_module()
