from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image
import numpy as np


def get_img_from_fig(fig, dpi=180):
    """Returns an image as numpy array from figure

    Args:
        fig (matplotlib.figure.Figure): Input figure.
        dpi (int, optional): Resolution. Defaults to 180.

    Returns:
        buf (np.array, dtype=np.uint8 or PIL.Image.Image): Resulting image.
    """
    fig.set_dpi(dpi)
    canvas = FigureCanvasAgg(fig)
    # Retrieve a view on the renderer buffer
    canvas.draw()
    buf = canvas.buffer_rgba()
    # convert to a NumPy array
    buf = np.asarray(buf)
    buf = Image.fromarray(buf)
    buf = buf.convert("RGB")
    return buf
