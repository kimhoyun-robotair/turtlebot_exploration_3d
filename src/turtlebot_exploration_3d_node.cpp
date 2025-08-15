// 원본 메인 로직의 ROS2 포팅 (불필요 추가 없음)

#include "turtlebot_exploration_3d/exploration.hpp"
#include "turtlebot_exploration_3d/navigation_utils.hpp"  // 원본과 동일 가정
#include "turtlebot_exploration_3d/gpregressor.h"
#include "turtlebot_exploration_3d/covMaterniso3.h"

#include <rclcpp/rclcpp.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <tf2/LinearMath/Quaternion.h>

#include <random>
#include <ctime>
#include <Eigen/Dense>

// ===== 글로벌 정의 (헤더 extern 매칭) =====
namespace jackal_exploration {
std::unique_ptr<octomap::OcTree> cur_tree;
std::string         octomap_name_3d;
jackal_exploration::point3d kinect_orig(0,0,0);
std::shared_ptr<tf2_ros::Buffer> g_tf_buffer;
rclcpp::Logger g_logger = rclcpp::get_logger("turtlebot_exploration_3d");
} // namespace jackal_exploration

using namespace jackal_exploration;

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  auto node = rclcpp::Node::make_shared("turtlebot_exploration_3d");
  g_logger = node->get_logger();

  // TF2
  g_tf_buffer = std::make_shared<tf2_ros::Buffer>(node->get_clock());
  tf2_ros::TransformListener tf_listener(*g_tf_buffer);

  // Octomap 파일명 (원본과 동일 포맷)
  {
    char buffer[80];
    std::time_t rawtime = std::time(nullptr);
    std::tm *ti = std::localtime(&rawtime);
    std::strftime(buffer, 80, "Octomap3D_%m%d_%H%M.ot", ti);
    octomap_name_3d = buffer;
  }

  // Octomap 초기화
  cur_tree = std::make_unique<octomap::OcTree>(octo_reso);

  // Pub/Sub
  auto sub_cloud = node->create_subscription<sensor_msgs::msg::PointCloud2>(
      "/mast_camera/points", rclcpp::SensorDataQoS(),
      &kinectCallbacks);

  auto pub_goal_marker = node->create_publisher<visualization_msgs::msg::Marker>("/Goal_Marker", 1);
  auto pub_candidates  = node->create_publisher<visualization_msgs::msg::MarkerArray>("/Candidate_MIs", 1);
  auto pub_frontiers   = node->create_publisher<visualization_msgs::msg::Marker>("/Frontier_points", 1);
  auto pub_twist       = node->create_publisher<geometry_msgs::msg::Twist>("/cmd_vel_mux/input/teleop", 1);
  auto pub_octomap     = node->create_publisher<octomap_msgs::msg::Octomap>("octomap_3d", 1);

  // 초기 6회 회전 스캔 (원본 로직 동일)
  for (int i=0; i<6; ++i) {
    // kinect_orig 업데이트 (TF)
    bool got_tf = false;
    while (!got_tf && rclcpp::ok()) {
      try {
        auto tf = g_tf_buffer->lookupTransform("map", "mast_rgbd_link", tf2::TimePointZero);
        kinect_orig = point3d(tf.transform.translation.x,
                              tf.transform.translation.y,
                              tf.transform.translation.z);
        got_tf = true;
      } catch (const tf2::TransformException &e) {
        RCLCPP_WARN(g_logger, "Wait for tf: Kinect frame (%s)", e.what());
      }
      rclcpp::sleep_for(std::chrono::milliseconds(50));
    }

    // 한 번 스캔 처리(ROS1의 ros::spinOnce)
    rclcpp::spin_some(node);

    // 옥토맵 퍼블리시(원본과 같은 방법)
    octomap_msgs::msg::Octomap msg;
    octomap_msgs::binaryMapToMsg(*cur_tree, msg);
    msg.header.frame_id = "map";
    msg.header.stamp = node->now();
    pub_octomap->publish(msg);

    // 3초 회전
    geometry_msgs::msg::Twist twist;
    RCLCPP_WARN(g_logger, "Rotate...");
    rclcpp::Time t0 = node->now();
    while ((node->now() - t0) < rclcpp::Duration::from_seconds(3.0)) {
      twist.angular.z = 0.6;
      pub_twist->publish(twist);
      rclcpp::sleep_for(std::chrono::milliseconds(50));
    }
    twist.angular.z = 0.0;
    pub_twist->publish(twist);
  }

  int robot_step_counter = 0;

  while (rclcpp::ok()) {
    // 프런티어 추출
    auto frontier_groups = extractFrontierPoints(cur_tree);

    // RViz CUBE_LIST (원본 동일)
    visualization_msgs::msg::Marker frontier_mk;
    frontier_mk.header.frame_id = "map";
    frontier_mk.header.stamp = node->now();
    frontier_mk.ns = "frontier_points_array";
    frontier_mk.id = 0;
    frontier_mk.type = visualization_msgs::msg::Marker::CUBE_LIST;
    frontier_mk.action = visualization_msgs::msg::Marker::ADD;
    frontier_mk.scale.x = frontier_mk.scale.y = frontier_mk.scale.z = octo_reso;
    frontier_mk.color.a = 1.0;
    frontier_mk.color.r = 1.0; frontier_mk.color.g = 0.0; frontier_mk.color.b = 0.0;
    for (const auto &grp : frontier_groups)
      for (const auto &p : grp) {
        geometry_msgs::msg::Point q; q.x=p.x(); q.y=p.y(); q.z=p.z()+octo_reso;
        frontier_mk.points.push_back(q);
      }
    pub_frontiers->publish(frontier_mk);

    // 후보 생성
    auto candidates = extractCandidateViewPoints(frontier_groups, kinect_orig, 30);
    std::shuffle(candidates.begin(), candidates.end(), std::mt19937{std::random_device{}()});
    auto gp_test_poses = candidates;

    RCLCPP_INFO(g_logger, "Candidate View Points: %zu Generated, %d evaluating...",
                candidates.size(), num_of_samples_eva);
    int temp_size = static_cast<int>(candidates.size()) - 3;
    if (temp_size < 1) {
      RCLCPP_ERROR(g_logger, "Very few candidates generated, finishing with exploration...");
      rclcpp::shutdown();
      return 0;
    }

    candidates.resize(std::min(num_of_samples_eva, temp_size));
    frontier_groups.clear();

    // MI 평가 (원본과 동일, 거리 정규화)
    std::vector<double> MIs(candidates.size(), 0.0);
    double before = countFreeVolume(cur_tree);
    double t_begin = node->now().seconds();

    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(candidates.size()); ++i) {
      auto c = candidates[i];
      auto hits = castSensorRays(cur_tree, c.first, c.second);
      double mi = calc_MI(cur_tree, c.first, hits, before);
      double dx = c.first.x()-kinect_orig.x(), dy = c.first.y()-kinect_orig.y();
      MIs[i] = mi / std::hypot(dx,dy);
    }

    // BO (원본 동일)
    GPRegressor g(100, 3, 0.01);
    for (int bay_itr=0; bay_itr<num_of_bay; ++bay_itr) {
      Eigen::MatrixXf gp_train_x(candidates.size(), 3),
                      gp_train_label(candidates.size(), 1),
                      gp_test_x(gp_test_poses.size(), 3);

      for (int i=0;i<(int)candidates.size();++i) {
        gp_train_x(i,0)=candidates[i].first.x();
        gp_train_x(i,1)=candidates[i].first.y();
        gp_train_x(i,2)=candidates[i].second.z();
        gp_train_label(i)=MIs[i];
      }
      for (int i=0;i<(int)gp_test_poses.size();++i) {
        gp_test_x(i,0)=gp_test_poses[i].first.x();
        gp_test_x(i,1)=gp_test_poses[i].first.y();
        gp_test_x(i,2)=gp_test_poses[i].second.z();
      }

      Eigen::MatrixXf gp_mean_MI, gp_var_MI;
      g.train(gp_train_x, gp_train_label);
      g.test(gp_test_x, gp_mean_MI, gp_var_MI);

      double beta = 2.4;
      std::vector<double> bay_acq_fun(gp_test_poses.size());
      for (int i=0;i<(int)gp_test_poses.size();++i)
        bay_acq_fun[i] = gp_mean_MI(i) + beta * gp_var_MI(i);

      auto idx_acq = sort_MIs(bay_acq_fun);
      auto c = gp_test_poses[idx_acq[0]];
      auto hits = castSensorRays(cur_tree, c.first, c.second);
      candidates.push_back(c);
      MIs.push_back(calc_MI(cur_tree, c.first, hits, before));
      gp_test_poses.erase(gp_test_poses.begin()+idx_acq[0]);
    }

    double t_end = node->now().seconds();
    RCLCPP_INFO(g_logger, "Mutual Information Eval took: %.3f s", (t_end - t_begin));

    // 정렬 인덱스
    auto idx_MI = sort_MIs(MIs);

    // 후보 마커(원본 스타일)
    tf2::Quaternion MI_heading; MI_heading.setRPY(0.0, -PI/2, 0.0); MI_heading.normalize();
    visualization_msgs::msg::MarkerArray cand_arr;
    cand_arr.markers.resize(candidates.size());
    for (int i=0;i<(int)candidates.size();++i) {
      auto &mk = cand_arr.markers[i];
      mk.header.frame_id = "map"; mk.header.stamp = node->now();
      mk.ns="candidates"; mk.id=i;
      mk.type=visualization_msgs::msg::Marker::ARROW; mk.action=visualization_msgs::msg::Marker::ADD;
      mk.pose.position.x = candidates[i].first.x();
      mk.pose.position.y = candidates[i].first.y();
      mk.pose.position.z = candidates[i].first.z();
      mk.pose.orientation.x = MI_heading.x();
      mk.pose.orientation.y = MI_heading.y();
      mk.pose.orientation.z = MI_heading.z();
      mk.pose.orientation.w = MI_heading.w();
      mk.scale.x = 2.0 * MIs[i] / std::max(1e-9, MIs[idx_MI[0]]);
      mk.scale.y = mk.scale.z = 0.2;
      mk.color.a = std::min(1.0, MIs[i] / std::max(1e-9, MIs[idx_MI[0]]));
      mk.color.r = 1.0; mk.color.g = 0.55; mk.color.b = 0.22;
    }
    pub_candidates->publish(cand_arr);
    cand_arr.markers.clear();

    // 최대 MI부터 시도 (원본 동일)
    bool arrived = false;
    int idx_ptr = 0;

    while (!arrived && rclcpp::ok()) {
      point3d next_vp = point3d(candidates[idx_MI[idx_ptr]].first.x(),
                                candidates[idx_MI[idx_ptr]].first.y(),
                                candidates[idx_MI[idx_ptr]].first.z());
      tf2::Quaternion Goal_heading; Goal_heading.setRPY(0,0, candidates[idx_MI[idx_ptr]].second.z());
      Goal_heading.normalize();
      RCLCPP_INFO(g_logger, "Max MI: %.3f @ (%.2f, %.2f, %.2f)",
                  MIs[idx_MI[idx_ptr]], next_vp.x(), next_vp.y(), next_vp.z());

      // Goal 마커
      visualization_msgs::msg::Marker goal;
      goal.header.frame_id="map"; goal.header.stamp=node->now();
      goal.ns="goal_marker"; goal.id=0;
      goal.type=visualization_msgs::msg::Marker::ARROW; goal.action=visualization_msgs::msg::Marker::ADD;
      goal.pose.position.x = next_vp.x();
      goal.pose.position.y = next_vp.y();
      goal.pose.position.z = 1.0;
      goal.pose.orientation.x = Goal_heading.x();
      goal.pose.orientation.y = Goal_heading.y();
      goal.pose.orientation.z = Goal_heading.z();
      goal.pose.orientation.w = Goal_heading.w();
      goal.scale.x=1.0; goal.scale.y=0.3; goal.scale.z=0.3;
      goal.color.a=1.0; goal.color.r=1.0; goal.color.g=0.0; goal.color.b=0.0;
      pub_goal_marker->publish(goal);

      // 이동 (원본 유틸 호출 가정)
      arrived = navigation_utils::goToDest(next_vp, Goal_heading);

      if (arrived) {
        // 도착 후 kinect_orig 갱신
        bool got_tf = false;
        while (!got_tf && rclcpp::ok()) {
          try {
            auto tf = g_tf_buffer->lookupTransform("map", "mast_rgbd_link", tf2::TimePointZero);
            kinect_orig = point3d(tf.transform.translation.x,
                                  tf.transform.translation.y,
                                  tf.transform.translation.z);
            got_tf = true;
          } catch (const tf2::TransformException &e) {
            RCLCPP_WARN(g_logger, "Wait for tf: Kinect frame (%s)", e.what());
          }
          rclcpp::sleep_for(std::chrono::milliseconds(50));
        }

        // 한 번 스캔
        rclcpp::spin_some(node);
        RCLCPP_INFO(g_logger, "Succeed, new Map Free Volume: %.3f", countFreeVolume(cur_tree));
        robot_step_counter++;

        // Octomap 퍼블리시
        octomap_msgs::msg::Octomap msg;
        octomap_msgs::binaryMapToMsg(*cur_tree, msg);
        msg.header.frame_id = "map";
        msg.header.stamp = node->now();
        pub_octomap->publish(msg);
      } else {
        RCLCPP_WARN(g_logger, "Failed to drive to the %d-th goal, switch to sub-optimal..", idx_ptr);
        idx_ptr++;
        if (idx_ptr > (int)MIs.size()) {
          RCLCPP_ERROR(g_logger, "None of the goals valid, shutting down.");
          rclcpp::shutdown();
        }
      }
    } // while(!arrived)
  } // while(ok)

  rclcpp::shutdown();
  return 0;
}
