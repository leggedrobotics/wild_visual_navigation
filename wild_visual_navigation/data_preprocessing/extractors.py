from bagpy import bagreader

class ExtractorBase(object):
    """
    Defines Extractor Interface
    """
    def __init__(self, bagdir, outdir, tag, **kwargs):
        pass

    def update(self):
        pass

class ImageExtractor(ExtractorBase):
    def __init__(self, bagdir, outdir, tag, **kwargs):
        self.outdir = outdir
        self.bagdir = bagdir
        
        self.reader = bagreader(self.bagdir + f"jetson_images_{tag}.bag")
        
    
    def update(self):
        self.reader
        
class PointcloudExtractor(ExtractorBase):
    def __init__(self, bagdir, outdir, tag, **kwargs):
        pass
    
    def update(self):
        pass
    
class CompslamTrajectoryExtractor(ExtractorBase):
    def __init__(self, bagdir, outdir, tag, **kwargs):
        pass
    
    def update(self):
        pass
    
class GpsTrajectoryExtractor(ExtractorBase):
    def __init__(self, outdir, tag, **kwargs):
        pass
    
    def update(self):
        pass