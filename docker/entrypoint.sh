#!/bin/bash
set -e

# Installing wild visual navigation
install_wvn="pip3 install -e /root/git/wild_visual_navigation"


echo "Installing wvn: ${install_wvn}..."
$install_wvn > /dev/null
echo "Done!"

echo "source /root/catkin_ws/devel/setup.bash" >> /root/.bashrc

exec "$@"
