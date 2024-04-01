#!/bin/bash
# This is a workaround to set the JACKAL_URDF_EXTRAS path
export JACKAL_URDF_EXTRAS="$(rospack find wild_visual_navigation_jackal)/urdf/extras.xacro"
xacro $(rospack find jackal_description)/urdf/jackal.urdf.xacro