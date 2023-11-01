# import rospy
# from std_msgs.msg import String

# def callback(data):
#     rospy.loginfo("I heard: %s", data.data)

# def listener():
#     # Initialize the node with a name 'listener'
#     rospy.init_node('listener', anonymous=True)

#     # Create a subscriber to the '/chatter' topic and specify the callback function
#     rospy.Subscriber("/chatter", String, callback)

#     # Keep the node running until it's shut down
#     rospy.spin()

# if __name__ == '__main__':
#     listener()
import json

class Serializable:
    def to_json(self):
        return json.dumps(self, default=lambda o: o.__dict__, indent=4)

    @classmethod
    def from_json(cls, json_str):
        params = json.loads(json_str)
        return cls(**params)

from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class ExperimentParams(Serializable):
    @dataclass
    class GeneralParams:
        name: str = "debug/debug"
        timestamp: bool = True
        tag_list: List[str] = field(default_factory=lambda: ["debug"])
        # ... [rest of the attributes]

    general: GeneralParams = GeneralParams()

# Create an experiment configuration
exp = ExperimentParams()

# Modify some attributes
exp.general.name = "experiment1/run1"
exp.general.tag_list = ["run1", "experiment1"]

# Serialize the experiment configuration to JSON
json_str = exp.to_json()
print(json_str)

# Save the configuration to a file
with open("experiment_config.json", "w") as f:
    f.write(json_str)

# Later, or in another part of the system...

# Load the configuration from a file
with open("experiment_config.json", "r") as f:
    loaded_json = f.read()

# Deserialize the configuration from JSON
loaded_exp = ExperimentParams.from_json(loaded_json)
print(loaded_exp)
