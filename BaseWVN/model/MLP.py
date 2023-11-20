import torch
import torch.nn.functional as F

class SimpleMLP(torch.nn.Module):
    def __init__(self, input_size: int = 64, hidden_sizes: [int] = [255], reconstruction: bool = False):
        super(SimpleMLP, self).__init__()
        layers = []
        self.nr_sigmoid_layers = hidden_sizes[-1]

        if reconstruction:
            hidden_sizes[-1] = hidden_sizes[-1] + input_size

        for hs in hidden_sizes[:-1]:
            layers.append(torch.nn.Linear(input_size, hs))
            layers.append(torch.nn.ReLU())
            input_size = hs
        layers.append(torch.nn.Linear(input_size, hidden_sizes[-1]))

        self.layers = torch.nn.Sequential(*layers)
        self.output_features = hidden_sizes[-1]

    def forward(self, x) -> torch.Tensor:
        # Checked data is correctly memory aligned and can be reshaped
        # If you change something in the dataloader make sure this is still working
        x = self.layers(x)
        return x
    
    
class SeperateMLP(torch.nn.Module):
    """ 
    Seperate reconstruction and regression net
    """
    def __init__(self, input_size: int = 64, hidden_sizes: [int] = [255]):
        super().__init__()
        
        self.nr_sigmoid_layers = hidden_sizes[-1]
        
        # reconstruction net
        layers = []
        hid_sizes=hidden_sizes.copy()
        ip_size=input_size
        hid_sizes[-1] = ip_size
        for hs in hid_sizes[:-1]:
            layers.append(torch.nn.Linear(ip_size, hs))
            layers.append(torch.nn.ReLU())
            ip_size = hs
        layers.append(torch.nn.Linear(ip_size, hid_sizes[-1]))
        self.reco_layers = torch.nn.Sequential(*layers)
        
        # regression net
        layers = []
        hid_sizes=hidden_sizes.copy()
        bottleneck=min(hid_sizes[:-1])
        ip_size=input_size
        for hs in hid_sizes[:-1]:
            if ip_size==bottleneck:
                break
            layers.append(torch.nn.Linear(ip_size, hs))
            layers.append(torch.nn.ReLU())
            ip_size = hs
        layers.append(torch.nn.Linear(ip_size, hid_sizes[-1]))
        self.reg_layers = torch.nn.Sequential(*layers)
    
    def forward(self, x) -> torch.Tensor:

        reco=self.reco_layers(x)
        reg=self.reg_layers(x)
        return torch.cat((reco,reg),dim=1)

class RndMLP(torch.nn.Module):
    def __init__(self, input_size: int = 64, 
                 hidden_sizes_target: [int] = [255], 
                 hidden_sizes_pred: [int] = [255],
                 pred_head:int=2):
        super().__init__()
        
        # last layer of target and pred must be the same
        assert hidden_sizes_target[-1]==hidden_sizes_pred[-1]
        
        # target network, frozen
        layers = []
        hid_sizes=hidden_sizes_target.copy()
        ip_size=input_size
        for hs in hid_sizes[:-1]:
            layers.append(torch.nn.Linear(ip_size, hs))
            layers.append(torch.nn.ReLU())
            ip_size = hs
        layers.append(torch.nn.Linear(ip_size, hid_sizes[-1]))
        self.target= torch.nn.Sequential(*layers)
        for param in self.target.parameters():
           param.requires_grad = False
        
        # predictor network
        layers = []
        hid_sizes=hidden_sizes_pred.copy()
        hid_sizes[-1] +=pred_head
        ip_size=input_size
        for hs in hid_sizes[:-1]:
            layers.append(torch.nn.Linear(ip_size, hs))
            layers.append(torch.nn.ReLU())
            ip_size = hs
        layers.append(torch.nn.Linear(ip_size, hid_sizes[-1]))
        self.predictor= torch.nn.Sequential(*layers)

    def forward(self, x) -> torch.Tensor:
        target=self.target(x)
        pred=self.predictor(x)
        return target,pred
        
class SeprndMLP(torch.nn.Module):
    def __init__(self, input_size: int = 64, 
                 hidden_sizes_target: [int] = [255], 
                 hidden_sizes_pred: [int] = [255],
                 pred_head:int=2):
        super().__init__()
        
        # last layer of target and pred must be the same
        assert hidden_sizes_target[-1]==hidden_sizes_pred[-1]
        
        # target network, frozen
        layers = []
        hid_sizes=hidden_sizes_target.copy()
        ip_size=input_size
        for hs in hid_sizes[:-1]:
            layers.append(torch.nn.Linear(ip_size, hs))
            layers.append(torch.nn.ReLU())
            ip_size = hs
        layers.append(torch.nn.Linear(ip_size, hid_sizes[-1]))
        self.target= torch.nn.Sequential(*layers)
        for param in self.target.parameters():
           param.requires_grad = False
        
        # predictor network
        layers = []
        hid_sizes=hidden_sizes_pred.copy()
        ip_size=input_size
        for hs in hid_sizes[:-1]:
            layers.append(torch.nn.Linear(ip_size, hs))
            layers.append(torch.nn.ReLU())
            ip_size = hs
        layers.append(torch.nn.Linear(ip_size, hid_sizes[-1]))
        self.predictor= torch.nn.Sequential(*layers)
        
        # regression network
        layers = []
        hid_sizes=hidden_sizes_pred.copy()
        hid_sizes.append(pred_head)
        ip_size=input_size
        for hs in hid_sizes[:-1]:
            layers.append(torch.nn.Linear(ip_size, hs))
            layers.append(torch.nn.ReLU())
            ip_size = hs
        layers.append(torch.nn.Linear(ip_size, hid_sizes[-1]))
        self.regressor= torch.nn.Sequential(*layers)
        

    def forward(self, x) -> torch.Tensor:
        target=self.target(x)
        pred=self.predictor(x)
        reg=self.regressor(x)
        
        return target,torch.cat((pred,reg),dim=1)      
        