from wild_visual_navigation.feature_extractor import FeatureExtractor
import os
from pathlib import Path
import torch
import pickle
from wild_visual_navigation import WVN_ROOT_DIR

if __name__ == "__main__":

    folder = "time_extractor"
    warm_up_runs = 100
    hot_runs = 100
    special_key = ""

    # Setup folder
    ws = os.environ["ENV_WORKSTATION_NAME"]
    res_folder = os.path.join(WVN_ROOT_DIR, f"results/ablations/{folder}_ablation_{ws}")
    p = os.path.join(res_folder, f"{folder}_ablation_test_results{special_key}.pkl")
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    try:
        os.remove(p)
    except OSError as error:
        pass
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create a dictionary of feature extractors.
    fes = {}
    fes["slic100_dino448_8"] = FeatureExtractor(
        device, "slic", "dino", 448, model_type="vit_small", patch_size=8, slic_num_components=100
    )
    fes["slic100_dino448_16"] = FeatureExtractor(
        device, "slic", "dino", 448, model_type="vit_small", patch_size=16, slic_num_components=100
    )
    fes["slic100_dino224_8"] = FeatureExtractor(
        device, "slic", "dino", 224, model_type="vit_small", patch_size=8, slic_num_components=100
    )
    fes["slic100_dino224_16"] = FeatureExtractor(
        device, "slic", "dino", 224, model_type="vit_small", patch_size=16, slic_num_components=100
    )
    fes["slic100_dino112_8"] = FeatureExtractor(
        device, "slic", "dino", 112, model_type="vit_small", patch_size=8, slic_num_components=100
    )
    fes["slic100_dino112_16"] = FeatureExtractor(
        device, "slic", "dino", 112, model_type="vit_small", patch_size=16, slic_num_components=100
    )
    fes["slic100_sift"] = FeatureExtractor(device, "slic", "sift", slic_num_components=100)

    fes["slic100_efficientnet_b0"] = FeatureExtractor(
        device, "slic", "torchvision", (256, 224), model_type="efficientnet_b0", slic_num_components=100
    )
    fes["slic100_efficientnet_b4"] = FeatureExtractor(
        device, "slic", "torchvision", (384, 380), model_type="efficientnet_b4", slic_num_components=100
    )
    fes["slic100_efficientnet_b7"] = FeatureExtractor(
        device, "slic", "torchvision", (633, 600), model_type="efficientnet_b7", slic_num_components=100
    )
    fes["slic100_resnet50"] = FeatureExtractor(
        device, "slic", "torchvision", 448, model_type="resnet50", slic_num_components=100
    )
    fes["slic100_resnet18"] = FeatureExtractor(
        device, "slic", "torchvision", 448, model_type="resnet18", slic_num_components=100
    )
    fes["slic100_resnet50_dino"] = FeatureExtractor(
        device, "slic", "torchvision", 448, model_type="resnet50_dino", slic_num_components=100
    )

    res = {}
    for extractor_name, fe in fes.items():
        print(extractor_name)
        input_size = fe._input_size
        if not isinstance(input_size, tuple):
            input_size = (input_size, input_size)

        times_warmup = []
        for i in range(warm_up_runs):
            img = torch.rand((1, 3, input_size[0], input_size[1]))
            img.to(device)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            feat = fe.compute_features(img, None, None)
            end.record()
            torch.cuda.synchronize()
            t = start.elapsed_time(end)
            times_warmup.append(t)

        times_eval = []
        for i in range(hot_runs):
            img = torch.rand((1, 3, input_size[0], input_size[1]))
            img.to(device)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            feat = fe.compute_features(img, None, None)
            end.record()
            torch.cuda.synchronize()
            t = start.elapsed_time(end)
            times_eval.append(t)
        try:
            model = fe.extractor.model
            model.train()
            params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        except:
            try:
                model = fe._extractor
                params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            except:
                params = 0

        res[extractor_name] = {
            "times_eval": times_eval,
            "times_warmup": times_warmup,
            "parameter": params,
            "warm_up_runs": warm_up_runs,
            "hot_runs": hot_runs,
            "device": torch.cuda.get_device_name(),
            "input_size": input_size,
        }

    with open(p, "wb") as handle:
        pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)
