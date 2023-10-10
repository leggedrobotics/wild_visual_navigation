from dataclasses import dataclass, field, asdict
from typing import Tuple, Dict, List, Optional, Any

@dataclass
class ExtractionParams:
    wvn_topics: List[str] = field(default_factory=lambda: ["/state_estimator/anymal_state",
                                                           # "/wide_angle_camera_front/img_out",
                                                           "/v4l2_camera/image_raw_throttle/compressed",
                                                           "/depth_camera_front_upper/point_cloud_self_filtered",
                                                           "/depth_camera_rear_upper/point_cloud_self_filtered",
                                                           "/depth_camera_left/point_cloud_self_filtered",
                                                           "/depth_camera_right/point_cloud_self_filtered"])
    wvn_bags: List[str] = field(default_factory=lambda: [
        # "/home/rschmid/RosBags/6_proc/images.bag",
        "/home/rschmid/RosBags/uetliberg_small/2023-09-20-09-43-57_anymal-d020-jetson_0.bag",
        "/home/rschmid/RosBags/uetliberg_small/2023-09-20-09-43-57_anymal-d020-jetson_1.bag",
        "/home/rschmid/RosBags/uetliberg_small/2023-09-20-09-43-57_anymal-d020-lpc_0.bag",
        "/home/rschmid/RosBags/uetliberg_small/2023-09-20-09-43-57_anymal-d020-lpc_1.bag",
        "/home/rschmid/RosBags/uetliberg_small/2023-09-20-09-43-57_anymal-d020-npc_0.bag",
        "/home/rschmid/RosBags/uetliberg_small/2023-09-20-09-43-57_anymal-d020-npc_1.bag",
                                                         # "/home/rschmid/RosBags/6_proc/2023-03-02-11-13-08_anymal-d020-jetson_mission_0.bag",
                                                         # "/home/rschmid/RosBags/6_proc/2023-03-02-11-13-08_anymal-d020-jetson_mission_1.bag",
               # "/home/rschmid/RosBags/6_proc/2023-03-02-11-13-08_anymal-d020-lpc_mission_0.bag",
               # "/home/rschmid/RosBags/6_proc/2023-03-02-11-13-08_anymal-d020-lpc_mission_1.bag",
               #  "/home/rschmid/RosBags/6_proc/2023-03-02-11-13-08_anymal-d020-npc_mission_0.bag",
               #  "/home/rschmid/RosBags/6_proc/2023-03-02-11-13-08_anymal-d020-npc_mission_1.bag"])
    ])

data: ExtractionParams = ExtractionParams()