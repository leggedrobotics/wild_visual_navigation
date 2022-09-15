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
