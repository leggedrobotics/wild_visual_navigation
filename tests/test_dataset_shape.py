from pathlib import Path
import torch
from os.path import join
from wild_visual_navigation import WVN_ROOT_DIR


def test_dataset_shape():
    elements = ["seg", "center", "feat", "img"]
    for el in elements:
        ls = [str(s) for s in Path(join(WVN_ROOT_DIR, f"results/perugia_forest/{el}")).rglob("*.pt")]
        sa = torch.load(ls[0]).shape
        for l in ls:
            el = torch.load(l)
            print(torch.equal(torch.tensor(sa), torch.tensor(el.shape)))
            assert torch.equal(torch.tensor(sa), torch.tensor(el.shape)), f"Error {l}, {el.shape}"
