import skimage
from wild_visual_navigation.feature_extractor import StegoInterface


class FeatureExtractor:
    def __init__(self):
        pass

    def extract(self, key, img, **kwargs):
        getattr(self, key)(img, **kwargs)

    def stego(self, img):
        pass

    def slic(self, img, n_segments=100, compactness=10.0):
        return skimage.segmentation.slic(img, n_segments=n_segments, compactness=compactness)
