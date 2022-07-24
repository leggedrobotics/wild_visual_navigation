from wild_visual_navigation.feature_extractor import DinoInterface
import cv2
import os
import torch
from wild_visual_navigation.utils import Timer
from wild_visual_navigation import WVN_ROOT_DIR


def test_dino_interfacer():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    di = DinoInterface(device)

    np_img = cv2.imread(os.path.join(WVN_ROOT_DIR, "assets/images/forest_clean.png"))
    img = torch.from_numpy(cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)).to(device)
    img = img.permute(2, 0, 1)
    img = (img.type(torch.float32) / 255)[None]

    for i in range(5):
        im = img + torch.rand(img.shape, device=img.device) / 100
        di.inference(di.transform(im))

    with Timer("BS1 Dino Inference: "):
        for i in range(5):
            im = img + torch.rand(img.shape, device=img.device) / 100
            with Timer("BS1 Dino Single: "):
                res = di.inference(di.transform(im))

    # img = img.repeat(4, 1, 1, 1)
    # with Timer("BS4 Dino Inference: "):
    #     for i in range(2):
    #         im = img + torch.rand(img.shape, device=img.device) / 100
    #         with Timer("BS4 Dino Single: "):
    #             res = di.inference(di.transform(im))

    import torch_tensorrt

    spec = {
        "forward": torch_tensorrt.ts.TensorRTCompileSpec(
            {
                "inputs": [torch_tensorrt.Input([1, 3, 488, 488])],
                "enabled_precisions": {torch.float, torch.half},
                "refit": False,
                "debug": False,
                "device": {
                    "device_type": torch_tensorrt.DeviceType.GPU,
                    "gpu_id": 0,
                    "dla_core": 0,
                    "allow_gpu_fallback": True,
                },
                "capability": torch_tensorrt.EngineCapability.default,
                "num_min_timing_iters": 2,
                "num_avg_timing_iters": 1,
            }
        )
    }
    script_model = torch.jit.script(di.model)
    trt_model = torch._C._jit_to_backend("tensorrt", script_model, spec)

    with Timer("TensorRT Inference: "):
        im = img + torch.rand(img.shape, device=img.device) / 100
        trt_model(di.transform(im))


if __name__ == "__main__":
    test_dino_interfacer()
