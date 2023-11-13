import numpy as np
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt

def plot_overlay_image(img, alpha=0.5, overlay_mask=None, channel=0, **kwargs):
    """ 
    overlay_mask: (C,H,W) mask with C channels
    output: (H,W,3) RGB image with overlay mask
    
    """
    # Apply your existing plot_image function
    overlay_mask = overlay_mask.cpu().numpy()
    img = plot_image(img.squeeze(0))  
    H, W = img.shape[:2]

    # Prepare the background
    back = np.zeros((H, W, 4))
    back[:, :, :3] = img
    back[:, :, 3] = 255

    if overlay_mask is not None:
        mask_channel = overlay_mask[channel, :, :]

        # Handling NaNs for normalization
        valid_mask = ~np.isnan(mask_channel)
        # Define the value range based on the channel
        if channel == 0:  # Friction channel
            min_val, max_val = 0, 1
        else:  # Stiffness channel (or others if present)
            min_val, max_val = 1, 10

        norm_mask = np.zeros_like(mask_channel)
        norm_mask[valid_mask] = (mask_channel[valid_mask] - min_val) / (max_val - min_val)
        
        cmap = sns.color_palette(kwargs.get("cmap", "viridis"), as_cmap=True)
        colored_mask = plt.cm.ScalarMappable(cmap=cmap).to_rgba(norm_mask, bytes=True)[:,:,:3]

        fore = np.zeros((H, W, 4), dtype=np.uint8)
        fore[:, :, :3] = colored_mask
        fore[:, :, 3] = valid_mask * alpha * 255

        img_new = Image.alpha_composite(Image.fromarray(np.uint8(back)), Image.fromarray(fore))
        img_new = img_new.convert("RGB")
        return np.uint8(img_new)
    else:
        return img


def plot_image( img, **kwargs):
    """
    ----------
    img : CHW HWC accepts torch.tensor or numpy.array
        Range 0-1 or 0-255
    """
    try:
        img = img.clone().cpu().numpy()
    except Exception:
        pass

    if img.shape[2] == 3:
        pass
    elif img.shape[0] == 3:
        img = np.moveaxis(img, [0, 1, 2], [2, 0, 1])
    else:
        raise Exception("Invalid Shape")
    if img.max() <= 1:
        img = img * 255

    img = np.uint8(img)
    return img

if __name__ == "__main__":
    # Create a sample image
    sample_image = np.random.rand(100, 100, 3)

    # Create an overlay mask with two channels (friction and stiffness)
    overlay_mask = np.random.rand(2, 100, 100) * 10  # Random values between 0 and 10
    overlay_mask[:, 20:40, 40:60] = np.nan  # Some NaNs for demonstration

    # Plot the overlay image for friction (channel 0)
    overlay_result = plot_overlay_image(sample_image, alpha=1.0, overlay_mask=overlay_mask, channel=0, cmap="viridis")
    # Displaying the images
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(sample_image)
    axes[0].set_title("Original Image")
    axes[1].imshow(overlay_result)
    axes[1].set_title("Overlay Image (Friction)")
    for ax in axes:
        ax.axis('off')
    plt.show()