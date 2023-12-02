
from BaseWVN.utils import *
import torch
width, height = 400, 300
channel = 0
img = torch.rand((1,3,height, width))

# Create a test mask with values in the range specific to the channel
if channel == 0:  # Friction channel
    mask = torch.ones((1,height, width))*0.8
else:  # Stiffness channel
    mask = np.random.uniform(1, 10, (height, width))

# Create overlay image
overlay_img = plot_overlay_image(img, overlay_mask=mask, alpha=1.0, channel=channel)

# Save image with color bar
output_path = "test_overlay_with_colorbar.png"
add_color_bar_and_save(overlay_img, channel, output_path)