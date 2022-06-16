from wild_visual_navigation.feature_extractor import StegoInterface
import skimage


class FeatureExtractor:
    def __init__(self, device):
        self.si = StegoInterface(device=device)

    def extract(self, key, img, **kwargs):
        return getattr(self, key)(img, **kwargs)

    def stego(self, img):
        return self.si.inference(img)

    def slic(self, img, n_segments=100, compactness=10.0):
        return skimage.segmentation.slic(img, n_segments=n_segments, compactness=compactness)
