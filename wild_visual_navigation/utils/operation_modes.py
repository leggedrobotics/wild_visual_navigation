#
# Copyright (c) 2022-2024, ETH Zurich, Jonas Frey, Matias Mattamala.
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#
from enum import Enum


class WVNMode(Enum):
    DEBUG = 0
    ONLINE = 1
    EXTRACT_LABELS = 2

    def from_string(string):
        if string == "debug":
            # Learning theard on
            # Everything published via ROS
            #       current_traversability: True
            #       current_confidence: True
            #       mission_graph: False
            #       debug_topics: False

            return WVNMode.DEBUG
        elif string == "online":
            # Learning theard on
            # ROS Predictions Published:
            #       current_traversability: True
            #       current_confidence: True
            #       mission_graph: False
            #       debug_topics: False
            return WVNMode.ONLINE
        elif string == "extract_labels":
            return WVNMode.EXTRACT_LABELS
        else:
            raise ValueError("Invalid WVNMode string")
