from wild_visual_navigation.data_preprocessing import ImageExtractor
from wild_visual_navigation.data_preprocessing import GpsTrajectoryExtractor
from wild_visual_navigation.data_preprocessing import CompslamTrajectoryExtractor
from wild_visual_navigation.data_preprocessing import PointcloudExtractor

__all__ = ['extractor_register']

extractor_register = {
    "image": ImageExtractor,
    "gps": GpsTrajectoryExtractor,
    "compslam": CompslamTrajectoryExtractor,
    "pointcloud": PointcloudExtractor
    }
