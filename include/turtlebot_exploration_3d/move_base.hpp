#pragma once
// ROS2 포팅: 원본 MoveBase 클래스와 동일 API (토픽만 ROS2 네임스페이스)

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <tf2/LinearMath/Quaternion.h>

class MoveBase {
public:
  explicit MoveBase(const rclcpp::Node::SharedPtr& node)
  : node_(node)
  {
    // 원본 토픽명 그대로 유지
    pub_ = node_->create_publisher<geometry_msgs::msg::Pose>("/mobile_base/commands/position", 10);
  }

  void move_to(float x, float y, float z, float yaw) {
    geometry_msgs::msg::Pose p;
    p.position.x = x;
    p.position.y = y;
    p.position.z = z;

    tf2::Quaternion q; q.setRPY(0.0, 0.0, yaw);
    q.normalize();
    p.orientation.x = q.x(); p.orientation.y = q.y();
    p.orientation.z = q.z(); p.orientation.w = q.w();

    pub_->publish(p);
  }

  void move_to(geometry_msgs::msg::Pose &p) {
    pub_->publish(p);
  }

private:
  rclcpp::Node::SharedPtr node_;
  rclcpp::Publisher<geometry_msgs::msg::Pose>::SharedPtr pub_;
};
