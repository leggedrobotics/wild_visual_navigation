from wild_visual_navigation.feature_extractor import FeatureExtractor

import numpy as np

fe = FeatureExtractor()
img = np.zeros((640, 480, 3), dtype=np.float32)
res = fe.extract("slic", img, n_segments=100, compactness=10.0)
