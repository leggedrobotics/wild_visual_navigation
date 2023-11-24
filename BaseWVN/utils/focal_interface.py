from seem_base import init_model,backbone_inference
import torch
class FocalInterface:
    def __init__(
        self,
        device: str,
        **kwargs
    ):
        self.device = device
        self.model=init_model().to(device)
        
    @torch.no_grad()
    def inference(self,img:torch.Tensor):
        # check if it has a batch dim or not
        if img.dim()==3:
            img=img.unsqueeze(0)
        if img.dtype!=torch.uint8:
            img=(img*255.0).type(torch.uint8)
        # Send to device
        img = img.to(self.device)
        # print("After transform shape is:",img.shape)
        features=backbone_inference(self.model,img)
        return features
    
    def change_device(self, device):
        """Changes the device of all the class members

        Args:
            device (str): new device
        """
        self.model.to(device)
        self.device = device