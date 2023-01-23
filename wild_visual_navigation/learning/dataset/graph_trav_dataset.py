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
import random


class GraphTravAblationDataset(Dataset):
    def __init__(
        self,
        feature_key: str,
        mode: str,
        env: str,
        perugia_root: str,
        training_data_percentage: int,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        minimal: bool = False,
    ):
        super().__init__()

        ls = []
        j = 0
        self.perugia_root = perugia_root
        self.minimal = minimal
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

            random.shuffle(ls)
            ls = ls[: int(len(ls) * training_data_percentage / 100)]

        self.mode = mode
        self.env = env
        self.paths = ls
        self.feature_key = feature_key
        self.crop = T.Compose([T.Resize(448, T.InterpolationMode.NEAREST), T.CenterCrop(448)])

    def len(self) -> int:
        return len(self.paths)

    def get(self, idx: int) -> any:
        # TODO update the dataset generation to avoid 0,0 and the cropping operation
        img_p = os.path.join(self.perugia_root, self.paths[idx])
        graph_p = img_p.replace("image", f"features/{self.feature_key}/graph")
        center_p = img_p.replace("image", f"features/{self.feature_key}/center")

        graph = torch.load(graph_p)
        center = torch.load(center_p)
        graph.center = center
        if not self.minimal:
            seg_p = img_p.replace("image", f"features/{self.feature_key}/seg")
            img = torch.load(img_p)
            seg = torch.load(seg_p)
            graph.img = img[None]
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

        graph.x_previous_count = graph.x_previous.shape[0]
        return graph


class GraphTravAblationDatasetInMemory(InMemoryDataset):
    def __init__(
        self,
        feature_key: str,
        mode: str,
        env: str,
        perugia_root: str,
        training_data_percentage: int,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
    ):
        super(GraphTravAblationDatasetInMemory, self).__init__()

        ds = GraphTravAblationDataset(
            feature_key=feature_key,
            mode=mode,
            env=env,
            perugia_root=perugia_root,
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
            training_data_percentage=training_data_percentage,
            minimal=True,
        )
        data_list = []
        for k in range(len(ds)):
            data_list.append(ds[k])

        self.data, self.slices = self.collate(data_list)


def get_ablation_module(
    perugia_root: str,
    env: str,
    feature_key: str,
    batch_size: int,
    num_workers: int,
    test_equals_val: bool,
    val_equals_test: bool,
    training_in_memory: bool,
    training_data_percentage: int,
    **kwargs,
) -> LightningDataset:

    if training_in_memory:
        train_dataset = GraphTravAblationDatasetInMemory(
            perugia_root=perugia_root,
            mode="train",
            feature_key=feature_key,
            env=env,
            training_data_percentage=training_data_percentage,
        )
    else:
        train_dataset = GraphTravAblationDataset(
            perugia_root=perugia_root,
            mode="train",
            feature_key=feature_key,
            env=env,
            training_data_percentage=training_data_percentage,
        )
    val_dataset = [
        GraphTravAblationDataset(
            perugia_root=perugia_root,
            mode="val",
            feature_key=feature_key,
            env=env,
            training_data_percentage=training_data_percentage,
        )
    ]

    if test_equals_val:
        test_dataset = [
            GraphTravAblationDataset(
                perugia_root=perugia_root,
                mode="val",
                feature_key=feature_key,
                env=env,
                training_data_percentage=training_data_percentage,
            )
        ]
    else:
        test_dataset = [
            GraphTravAblationDataset(
                perugia_root=perugia_root,
                mode="test",
                feature_key=feature_key,
                env=env,
                training_data_percentage=training_data_percentage,
            )
        ]

    if kwargs.get("test_all_datasets"):
        test_dataset = [
            GraphTravAblationDataset(
                perugia_root=perugia_root,
                mode="test",
                feature_key=feature_key,
                env="forest",
                training_data_percentage=training_data_percentage,
            ),
            GraphTravAblationDataset(
                perugia_root=perugia_root,
                mode="test",
                feature_key=feature_key,
                env="grassland",
                training_data_percentage=training_data_percentage,
            ),
            GraphTravAblationDataset(
                perugia_root=perugia_root,
                mode="test",
                feature_key=feature_key,
                env="hilly",
                training_data_percentage=training_data_percentage,
            ),
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
