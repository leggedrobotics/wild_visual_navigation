#include <anymal_msgs/AnymalState.h>
#include <ros/console.h>
#include <ros/ros.h>
#include <wild_visual_navigation_msgs/CustomState.h>
#include <wild_visual_navigation_msgs/RobotState.h>

class AnymalMsgConverter {
 public:
  AnymalMsgConverter(ros::NodeHandle& nh) {
    // Get parameters
    std::string anymal_state_topic;
    nh.param<std::string>("anymal_state_topic", anymal_state_topic, "/state_estimator/anymal_state");

    std::string output_topic;
    nh.param<std::string>("output_topic", output_topic, "/wild_visual_navigation_node/robot_state");

    // Setup subscriber
    anymal_state_sub_ = nh.subscribe(anymal_state_topic, 20, &AnymalMsgConverter::anymalStateCallback, this);

    // Setup publishers
    robot_state_pub_ = nh.advertise<wild_visual_navigation_msgs::RobotState>(output_topic, 20);

    // Spin
    ros::spin();
  }

  void anymalStateCallback(const anymal_msgs::AnymalStateConstPtr& anymal_msg) {
    robot_state_msg_.header = anymal_msg->header;

    // Store pose
    robot_state_msg_.pose = anymal_msg->pose;

    // Store twist
    robot_state_msg_.twist = anymal_msg->twist;

    robot_state_pub_.publish(robot_state_msg_);
  }

 private:
  wild_visual_navigation_msgs::RobotState robot_state_msg_;
  ros::Subscriber anymal_state_sub_;
  ros::Publisher robot_state_pub_;
};

int main(int argc, char** argv) {
  ros::init(argc, argv, "anymal_msg_converter");
  ros::NodeHandle nh("~");
  AnymalMsgConverter converter(nh);
  return 0;
}