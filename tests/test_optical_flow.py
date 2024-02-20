import pytest
import sys

try:
    from pytorch_pwc.network import PwcFlowEstimator
    from pytorch_pwc import PWC_ROOT_DIR
except ImportError:
    pass


@pytest.mark.skipif("pytorch_pwc" not in sys.modules, reason="requires the pytorch_pwc library")
def pytorch_pwc_test():
    import os
    import torch
    from wild_visual_navigation import WVN_ROOT_DIR
    from wild_visual_navigation.visu import LearningVisualizer
    import numpy as np
    import PIL
    from pytictac import Timer

    tenOne = torch.FloatTensor(
        np.ascontiguousarray(
            np.array(PIL.Image.open(os.path.join(PWC_ROOT_DIR, "assets/one.png")))[:, :, ::-1]
            .transpose(2, 0, 1)
            .astype(np.float32)
            * (1.0 / 255.0)
        )
    ).cuda()
    tenTwo = torch.FloatTensor(
        np.ascontiguousarray(
            np.array(PIL.Image.open(os.path.join(PWC_ROOT_DIR, "assets/two.png")))[:, :, ::-1]
            .transpose(2, 0, 1)
            .astype(np.float32)
            * (1.0 / 255.0)
        )
    ).cuda()

    with Timer("inference"):
        fe = PwcFlowEstimator(device="cuda")

    a, b = tenOne[:, :436, :436], tenTwo[:, :436, :436]
    with Timer("inference100"):
        for i in range(100):
            res = fe.forward(a, b)

    visu = LearningVisualizer(p_visu=os.path.join(WVN_ROOT_DIR, "results/test_visu"), store=True)
    visu.plot_optical_flow(res, tenOne, tenTwo)
    print("done")


if __name__ == "__main__":
    pytorch_pwc_test()
