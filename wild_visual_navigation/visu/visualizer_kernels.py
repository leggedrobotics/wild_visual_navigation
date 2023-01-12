import cupy as cp
import string


def plot_segmentation_kernel():
    plot_segmentation_kernel = cp.ElementwiseKernel(
        in_params="raw T seg_map, raw U col_mapping, raw T image_height, raw T image_width",
        out_params="raw U seg_image",
        preamble=string.Template(
            """
            __device__ int get_map_idx(int idx, int layer_n, int width, int height) {
                const int layer = width * height;
                return layer * layer_n + idx;
            }
            
            """
        ).substitute(),
        operation=string.Template(
            """
            int r_idx = get_map_idx(i, 0, image_height, image_width);
            int g_idx = get_map_idx(i, 1, image_height, image_width);
            int b_idx = get_map_idx(i, 2, image_height, image_width);
            
            seg_image[r_idx] = col_mapping[seg_map[i]*3];
            seg_image[g_idx] = col_mapping[seg_map[i]*3+1];
            seg_image[b_idx] = col_mapping[seg_map[i]*3+2];
            
            """
        ).substitute(),
        name="plot_segmentation_kernel",
    )
    return plot_segmentation_kernel


if __name__ == "__main__":
    import torch
    import numpy as np
    from wild_visual_navigation import WVN_ROOT_DIR
    from PIL import Image
    import os
    import seaborn as sns
    from wild_visual_navigation.utils import Timer

    segmentation_kernel = plot_segmentation_kernel()

    img = np.array(Image.open(os.path.join(WVN_ROOT_DIR, "assets/graph/img.png")))
    img = (torch.from_numpy(img).type(torch.float32) / 255).permute(2, 0, 1)
    trav_pred = torch.load(os.path.join(WVN_ROOT_DIR, "assets/graph/trav_pred.pt"))
    seg = torch.load(os.path.join(WVN_ROOT_DIR, "assets/graph/seg.pt"))
    H, W = seg.shape
    out_img = cp.zeros((3, H, W)).astype(cp.float32)
    c_map = sns.color_palette("Set2", 100)
    cp_c_map = cp.array(c_map).astype(cp.float32)
    cp_seg = cp.asarray(seg).astype(cp.uint32)

    with Timer("seg"):
        for i in range(100):
            segmentation_kernel(cp_seg, cp_c_map, cp.uint32(H), cp.uint32(W), out_img, size=(int(H * W)))
    with Timer("sing"):
        segmentation_kernel(cp_seg, cp_c_map, cp.uint32(H), cp.uint32(W), out_img, size=(int(H * W)))

    b = np.uint8(torch.as_tensor(out_img, device="cuda").cpu().permute(1, 2, 0).numpy() * 255)
    i = Image.fromarray(b)
    print("Loaded data!")
