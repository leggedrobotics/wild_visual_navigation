#
# Copyright (c) 2022-2024, ETH Zurich, Jonas Frey, Matias Mattamala.
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt


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


class PlotHelper:
    def __init__(self):
        self.data = []
        self.tag = []

    def add(self, img, tag="nan"):
        if torch.is_tensor(img):
            data = img.clone().cpu().numpy()
        else:
            data = np.copy(img)

        if len(data.shape) == 4:
            data = data[0]

        if data.shape[0] == 1 or data.shape[0] == 3:
            data = data.transpose((1, 2, 0))

        if data.dtype != np.uint8:
            if data.max() <= 1.0:
                data = np.uint8(data * 255)
            else:
                data = np.uint8(data)
        self.data.append(data)
        self.tag.append(tag)

    def show(self):
        figure = self.create_figure()
        plt.show()

    def create_figure(self):
        figure, axis = plt.subplots(1, len(self.data), figsize=(5 * len(self.data), 5))
        if len(self.data) == 1:
            axis = [axis]

        for i, (data, tag) in enumerate(zip(self.data, self.tag)):
            if len(data.shape) == 3 and data.shape[2] == 1:
                axis[i].imshow(data.squeeze(2))
            else:
                axis[i].imshow(data)
            axis[i].set_title(tag)
        return figure

    def get_img(self):
        return get_img_from_fig(self.create_figure())
