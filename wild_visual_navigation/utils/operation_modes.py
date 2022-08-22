from enum import Enum

class WVNMode(Enum):
    DEAFULT = 0
    ONLINE = 1
    EXTRACT_LABELS = 2
    
    def from_string(string):
        if string == "default":
            return WVNMode.DEAFULT
        elif string == "online":
            return WVNMode.ONLINE
        elif string == "extract_labels":
            return WVNMode.EXTRACT_LABELS
        else:
            raise ValueError("Invalid WVNMode string")