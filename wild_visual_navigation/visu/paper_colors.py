#
# Copyright (c) 2022-2024, ETH Zurich, Jonas Frey, Matias Mattamala.
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#
paper_colors_rgb_u8 = {
    "orange": (251, 151, 39),
    "mangenta": (150, 36, 145),
    "blue": (67, 110, 176),
    "red": (210, 43, 38),
    "cyan": (66, 173, 187),
    "green": (167, 204, 110),
    "red_light": (230, 121, 117),
    "orange_light": (252, 188, 115),
    "mangenta_light": (223, 124, 218),
    "blue_light": (137, 166, 210),
    "cyan_light": (164, 216, 223),
    "green_light": (192, 218, 152),
}
paper_colors_rgba_u8 = {k: (v[0], v[1], v[2], 255) for k, v in paper_colors_rgb_u8.items()}
paper_colors_rgb_f = {
    k: (float(v[0]) / 255.0, float(v[1]) / 255.0, float(v[2]) / 255.0) for k, v in paper_colors_rgb_u8.items()
}
paper_colors_rgba_f = {k: (v[0], v[1], v[2], 1.0) for k, v in paper_colors_rgb_f.items()}


def adjust_lightness(color, amount=0.5):
    """
    From https://stackoverflow.com/a/49601444
    """
    import matplotlib.colors as mc
    import colorsys

    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])


def darken(color, amount=0.0):
    """Darkens a color. 0.0 means no change"""
    import numpy as np

    return adjust_lightness(color, 1.0 - np.clip(amount, 0.0, 1.0))


def lighten(color, amount=0.0):
    """Darkens a color. 0.0 means no change"""
    import numpy as np

    return adjust_lightness(color, 1.0 + np.clip(amount, 0.0, 1.0))
