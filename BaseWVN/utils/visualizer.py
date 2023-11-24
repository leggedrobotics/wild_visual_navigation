import numpy as np
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import gridspec
from openTSNE import TSNE
import os
import torch

def plot_tsne(confidence, loss_reco_raw, title='t-SNE with Confidence Highlighting',path=None,sample_size=1000):
    """
    Plots a 2D t-SNE visualization of the given data, highlighting points based on a confidence mask.

    :param confidence: A 2D array (HxW) indicating confidence (True/False) for each data point.
    :param loss_reco_raw: An array of shape NxM where N is the number of samples and M is the number of features.
    :param title: Title of the plot.
    """
    if not os.path.exists(path):
        os.makedirs(path)
    if isinstance(loss_reco_raw, torch.Tensor):
        loss_reco_raw = loss_reco_raw.detach().cpu().numpy()
    if isinstance(confidence, torch.Tensor):
        confidence = confidence.cpu().numpy()
    # Flatten the confidence mask
    confidence_flat = confidence.flatten()

    # Sampling data if necessary
    if len(loss_reco_raw) > sample_size:
        indices = np.random.choice(len(loss_reco_raw), size=sample_size, replace=False)
        sampled_data = loss_reco_raw[indices]
        sampled_confidence = confidence_flat[indices]
    else:
        sampled_data = loss_reco_raw
        sampled_confidence = confidence_flat
    
    # Apply t-SNE to the data
    # tsne = TSNE(n_components=2, random_state=0)
    # data_2d = tsne.fit_transform(loss_reco_raw)
    data_2d=TSNE().fit(sampled_data)
    # Assign colors based on confidence
    colors = ['orange' if conf else 'red' for conf in sampled_confidence]

    # Plotting
    plt.figure(figsize=(10, 8))
    for i in range(len(data_2d)):
        plt.scatter(data_2d[i, 0], data_2d[i, 1], color=colors[i])

    plt.title(title)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    # plt.show()
    plt.savefig(os.path.join(path,title+".png"))
    plt.close()

def plot_overlay_image(img, alpha=0.5, overlay_mask=None, channel=0, **kwargs):
    """ 
    overlay_mask: (C,H,W) mask with C channels
    img:(1,C,H,W) image with C channels
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
            mask_channel=np.clip(mask_channel,0,1)
        else:  # Stiffness channel (or others if present)
            min_val, max_val = 1, 10
            mask_channel=np.clip(mask_channel,1,10)
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

def concatenate_images(images):
    """Concatenate a list of images horizontally, assuming they are already np.uint8 arrays."""
    images_pil = [Image.fromarray(img) for img in images]
    widths, heights = zip(*(i.size for i in images_pil))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images_pil:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]

    return new_im

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


def plot_images_side_by_side(images, titles, save_path='comparison.png'):
    """
    Plot a list of images side by side with titles and save the plot.

    Args:
    images (list of np.array): List of images (H,W,C) to plot. they need to rotate 180 degree for proper display
    titles (list of str): List of titles for each image.
    save_path (str): Path to save the image file.
    """
    num_images = len(images)
    fig, axes = plt.subplots(1, num_images, figsize=(num_images * 5, 5))

    for ax, img, title in zip(axes, images, titles):
        # Display image
        rotated_img = np.rot90(np.rot90(img))
        ax.imshow(rotated_img)
        ax.set_title(title)
        ax.axis('off')  # Turn off axis

    plt.tight_layout()
    plt.savefig(save_path)
    # plt.show()
    plt.close()

def plot_images_in_grid(images, titles, rows=10,cols=3, save_path='grid_comparison.png', show_plot=False):
    """
    Plot a list of images in a grid with titles and save the plot.

    Args:
    images (list of np.array): List of images (H,W,C) to plot. They need to rotate 180 degree for proper display.
    titles (list of str): List of titles for each image.
    rows (int): Number of rows in the grid.
    save_path (str): Path to save the image file.
    show_plot (bool): Whether to show the plot or not.
    """
    num_images = len(images)

    if num_images > cols * rows or num_images == 0:
        raise ValueError(f"Invalid number of images ({num_images}) for the grid of {rows}x{cols}")

     # Calculate number of columns
    # gs = gridspec.GridSpec(rows, cols, height_ratios=[1]*rows)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 3))
    axes = axes.flatten()  # Flatten the axes array for easy indexing

    for idx, (ax, img, title) in enumerate(zip(axes, images, titles)):
        # Rotate image by 180 degrees
        ax.margins(0)
        ax.axis('off')
        rotated_img = np.rot90(np.rot90(img))
        ax.imshow(rotated_img)
        if idx < cols:
            ax.set_title(title)
    
     # Turn off any unused subplots
    for ax in axes[num_images:]:
        ax.axis('off')   
    plt.tight_layout()
    plt.savefig(save_path)

    if show_plot:
        plt.show()
    plt.close()

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