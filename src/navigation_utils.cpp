#include "turtlebot_exploration_3d/navigation_utils.hpp"

namespace navigation_utils {

bool goToDest(point3d go_posi, const tf2::Quaternion& q,
              std::chrono::seconds timeout)
{
  auto node = rclcpp::Node::make_shared("navigation_utils");
  auto logger = node->get_logger();

  using ClientT = rclcpp_action::Client<NavigateToPose>;
  auto client = rclcpp_action::create_client<NavigateToPose>(node, "navigate_to_pose");

  if (!client->wait_for_action_server(std::chrono::seconds(5))) {
    RCLCPP_INFO(logger, "Waiting for the navigate_to_pose action server to come up...");
    if (!client->wait_for_action_server(std::chrono::seconds(10))) {
      RCLCPP_ERROR(logger, "Nav2 action server not available");
      return false;
    }
  }

  auto cancel_all_future = client->async_cancel_all_goals();
  {
    rclcpp::executors::SingleThreadedExecutor exec;
    exec.add_node(node);
    exec.spin_until_future_complete(cancel_all_future, std::chrono::milliseconds(200));
  }

  NavigateToPose::Goal goal;
  goal.pose.header.frame_id = "map";
  goal.pose.header.stamp = node->now();
  goal.pose.pose.position.x = go_posi.x();
  goal.pose.pose.position.y = go_posi.y();
  goal.pose.pose.position.z = go_posi.z();
  geometry_msgs::msg::Quaternion qmsg;
  qmsg.x = q.x(); qmsg.y = q.y(); qmsg.z = q.z(); qmsg.w = q.w();
  goal.pose.pose.orientation = qmsg;

  RCLCPP_INFO(logger, "Sending robot to the viewpoint... (%.2f, %.2f, %.2f)",
              go_posi.x(), go_posi.y(), go_posi.z());

  // ✅ 올바른 SendGoalOptions 타입
  ClientT::SendGoalOptions opts;  // 콜백 지정 안 함 (동기 대기만 할 거라서)
  auto goal_handle_future = client->async_send_goal(goal, opts);

  rclcpp::executors::SingleThreadedExecutor exec;
  exec.add_node(node);

  if (exec.spin_until_future_complete(goal_handle_future, std::chrono::seconds(5))
        != rclcpp::FutureReturnCode::SUCCESS) {
    RCLCPP_ERROR(logger, "Failed to get goal handle");
    return false;
  }
  auto goal_handle = goal_handle_future.get();
  if (!goal_handle) {
    RCLCPP_ERROR(logger, "Goal was rejected by server");
    return false;
  }

  auto result_future = client->async_get_result(goal_handle);
  if (exec.spin_until_future_complete(result_future, timeout)
        != rclcpp::FutureReturnCode::SUCCESS) {
    RCLCPP_WARN(logger, "Navigation timed out, canceling goal...");
    client->async_cancel_goal(goal_handle);
    return false;
  }

  auto wrapped_result = result_future.get();
  switch (wrapped_result.code) {
    case rclcpp_action::ResultCode::SUCCEEDED:
      return true;
    case rclcpp_action::ResultCode::ABORTED:
      RCLCPP_WARN(logger, "Navigation aborted");
      return false;
    case rclcpp_action::ResultCode::CANCELED:
      RCLCPP_WARN(logger, "Navigation canceled");
      return false;
    default:
      RCLCPP_WARN(logger, "Navigation ended with unknown result code");
      return false;
  }

  // 🔒 안전망 (이 줄에 도달하지 않지만, 컴파일러 경고 방지용)
  return false;
}

} // namespace navigation_utils
