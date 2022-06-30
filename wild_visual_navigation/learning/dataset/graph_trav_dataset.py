from torch_geometric.data import InMemoryDataset, DataLoader, DataListLoader
from wild_visual_navigation import WVN_ROOT_DIR
import os
import torch
from pathlib import Path


class GraphTravDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform)
        paths = [str(s) for s in Path(os.path.join(root, "graph")).rglob("*.pt")]
        paths.sort()
        data_list = [torch.load(p) for p in paths]
        self.data, self.slices = self.collate(data_list)


from torch_geometric.data import LightningDataset


def get_pl_graph_trav_module(batch_size=8, num_workers=1, **kwargs):
    root = str(os.path.join(WVN_ROOT_DIR, "results/perugia_forest"))
    train_dataset = GraphTravDataset(root=root)
    return LightningDataset(train_dataset, batch_size=batch_size, num_workers=num_workers, *kwargs)


if __name__ == "__main__":
    from torch_geometric.data import DataLoader

    root = str(os.path.join(WVN_ROOT_DIR, "results/perugia_forest"))
    dataset = GraphTravDataset(root=root)
    dl = DataListLoader(dataset, batch_size=8)
    datamodule = get_pl_graph_trav_module()
