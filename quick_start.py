#
# Copyright (c) 2022-2024, ETH Zurich, Matias Mattamala, Jonas Frey.
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#
from wild_visual_navigation import WVN_ROOT_DIR
from wild_visual_navigation.feature_extractor import FeatureExtractor
from wild_visual_navigation.cfg import ExperimentParams
from wild_visual_navigation.image_projector import ImageProjector
from wild_visual_navigation.model import get_model
from wild_visual_navigation.utils import ConfidenceGenerator
from wild_visual_navigation.utils import AnomalyLoss
from PIL import Image
import torch
import numpy as np
import torch.nn.functional as F
from omegaconf import OmegaConf
from wild_visual_navigation.utils import Data
from os.path import join
import os
from argparse import ArgumentParser
from wild_visual_navigation.model import get_model
from pathlib import Path
from wild_visual_navigation.visu import LearningVisualizer


# Function to handle folder creation
def parse_folders(args):
    input_image_folder = args.input_image_folder
    output_folder = args.output_folder_name

    # Check if input folder is global or local
    if not os.path.isabs(input_image_folder):
        input_image_folder = os.path.join(WVN_ROOT_DIR, "assets", input_image_folder)

    # Check if output folder is global or local
    if not os.path.isabs(output_folder):
        output_folder = os.path.join(WVN_ROOT_DIR, "results", output_folder)

    # Create input folder if it doesn't exist
    if not os.path.exists(input_image_folder):
        raise ValueError(f"Input folder '{input_image_folder}' does not exist.")

    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    return input_image_folder, output_folder


if __name__ == "__main__":
    parser = ArgumentParser()

    # Define command line arguments

    parser.add_argument("--model_name", default="indoor_mpi", help="Description of model name argument")
    parser.add_argument(
        "--input_image_folder",
        default="demo_data",
        help="Gloabl path or folder name within the assests directory",
    )
    parser.add_argument(
        "--output_folder_name",
        default="demo_data",
        help="Gloabl path or folder name within the results directory",
    )

    # Fixed values
    parser.add_argument("--network_input_image_height", type=int, default=224, help="Height of the input image")
    parser.add_argument("--network_input_image_width", type=int, default=224, help="Width of the input image")
    parser.add_argument(
        "--segmentation_type",
        default="stego",
        choices=["slic", "grid", "random", "stego"],
        help="Options: slic, grid, random, stego",
    )
    parser.add_argument(
        "--feature_type", default="stego", choices=["dino", "dinov2", "stego"], help="Options: dino, dinov2, stego"
    )
    parser.add_argument("--dino_patch_size", type=int, default=8, choices=[8, 16], help="Options: 8, 16")
    parser.add_argument("--dino_backbone", default="vit_small", choices=["vit_small"], help="Options: vit_small")
    parser.add_argument(
        "--slic_num_components", type=int, default=100, help="Number of components for SLIC segmentation"
    )

    parser.add_argument(
        "--compute_confidence", action="store_true", help="Compute confidence for the traversability prediction"
    )
    parser.add_argument("--no-compute_confidence", dest="compute_confidence", action="store_false")
    parser.set_defaults(compute_confidence=True)

    parser.add_argument(
        "--prediction_per_pixel", action="store_true", help="Inference traversability per-pixel or per-segment"
    )
    parser.add_argument("--no-prediction_per_pixel", dest="prediction_per_pixel", action="store_false")
    parser.set_defaults(prediction_per_pixel=True)

    # Parse the command line arguments
    args = parser.parse_args()

    input_image_folder, output_folder = parse_folders(args)

    params = OmegaConf.structured(ExperimentParams)
    anomaly_detection = False

    # Update model from file if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    visualizer = LearningVisualizer(p_visu=output_folder, store=True)

    if anomaly_detection:
        confidence_generator = ConfidenceGenerator(
            method=params.loss_anomaly.method, std_factor=params.loss_anomaly.confidence_std_factor
        )
    else:
        confidence_generator = ConfidenceGenerator(
            method=params.loss.method, std_factor=params.loss.confidence_std_factor
        )

    # Load feature and segment extractor
    feature_extractor = FeatureExtractor(
        device=device,
        segmentation_type=args.segmentation_type,
        feature_type=args.feature_type,
        patch_size=args.dino_patch_size,
        backbone_type=args.dino_backbone,
        input_size=args.network_input_image_height,
        slic_num_components=args.slic_num_components,
    )

    # Sorry for that 💩
    params.model.simple_mlp_cfg.input_size = feature_extractor.feature_dim
    params.model.double_mlp_cfg.input_size = feature_extractor.feature_dim
    params.model.simple_gcn_cfg.input_size = feature_extractor.feature_dim
    params.model.linear_rnvp_cfg.input_size = feature_extractor.feature_dim

    # Load traversability model
    model = get_model(params.model).to(device)
    model.eval()
    torch.set_grad_enabled(False)

    p = join(WVN_ROOT_DIR, "assets", "checkpoints", f"{args.model_name}.pt")
    model_state_dict = torch.load(p)
    model.load_state_dict(model_state_dict, strict=False)
    print(f"\nLoaded model `{args.model_name}` successfully!")

    cg = model_state_dict["confidence_generator"]
    # Only mean and std are needed
    confidence_generator.var = cg["var"]
    confidence_generator.mean = cg["mean"]
    confidence_generator.std = cg["std"]

    images = [str(s) for s in Path(input_image_folder).rglob("*.png" or "*.jpg")]
    print(f"Found {len(images)} images in the folder! \n")

    H, W = args.network_input_image_height, args.network_input_image_width
    for i, img_p in enumerate(images):
        print(f"Processing image {i+1}/{len(images)}: {img_p}")
        img = Image.open(img_p)
        img = img.convert("RGB")
        torch_image = torch.from_numpy(np.array(img))
        torch_image = torch_image.to(device).permute(2, 0, 1).float() / 255.0

        C, H_in, W_in = torch_image.shape

        # K can be ignored given that no reprojection is performed
        image_projector = ImageProjector(
            K=torch.eye(4, device=device)[None],
            h=H_in,
            w=W_in,
            new_h=H,
            new_w=W,
        )

        torch_image = image_projector.resize_image(torch_image)
        # Extract features
        _, feat, seg, center, dense_feat = feature_extractor.extract(
            img=torch_image[None],
            return_centers=False,
            return_dense_features=True,
            n_random_pixels=100,
        )

        # Forward pass to predict traversability
        if args.prediction_per_pixel:
            # Pixel-wise traversability prediction using the dense features
            data = Data(x=dense_feat[0].permute(1, 2, 0).reshape(-1, dense_feat.shape[1]))
        else:
            # input_feat = dense_feat[0].permute(1, 2, 0).reshape(-1, dense_feat.shape[1])
            # Segment-wise traversability prediction using the average feature per segment
            input_feat = feat[seg.reshape(-1)]
            data = Data(x=input_feat)

        # Inference model
        prediction = model.forward(data)

        # Calculate traversability
        if not anomaly_detection:
            out_trav = prediction.reshape(H, W, -1)[:, :, 0]
        else:
            losses = prediction["logprob"].sum(1) + prediction["log_det"]
            confidence = confidence_generator.inference_without_update(x=-losses)
            trav = confidence
            out_trav = trav.reshape(H, W, -1)[:, :, 0]

        original_img = visualizer.plot_image(torch_image, store=False)
        img_ls = [original_img]

        if args.compute_confidence:
            # Calculate confidence
            loss_reco = F.mse_loss(prediction[:, 1:], data.x, reduction="none").mean(dim=1)
            confidence = confidence_generator.inference_without_update(x=loss_reco)
            out_confidence = confidence.reshape(H, W)
            conf_img = visualizer.plot_detectron_classification(torch_image, out_confidence, store=False)
            img_ls.append(conf_img)

        name = img_p.split("/")[-1].split(".")[0]
        trav_img = visualizer.plot_detectron_classification(torch_image, out_trav, store=False)
        print(out_trav.sum(), out_trav.max(), torch_image.sum(), data.x.sum(), dense_feat.sum(), torch_image.sum())

        img_ls.append(trav_img)
        visualizer.plot_list(img_ls, tag=f"{name}_original_conf_trav", store=True)
