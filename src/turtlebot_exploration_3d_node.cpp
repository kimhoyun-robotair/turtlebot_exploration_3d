// 클래스화된 ROS2 포팅 (원본 로직은 그대로, 하드코딩만 파라미터화)
#include "turtlebot_exploration_3d/exploration.hpp"
#include "turtlebot_exploration_3d/navigation_utils.hpp"
#include "turtlebot_exploration_3d/gpregressor.h"
#include "turtlebot_exploration_3d/covMaterniso3.h"

#include <rclcpp/rclcpp.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <octomap_msgs/msg/octomap.hpp>
#include <octomap_msgs/conversions.h>

#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>

#include <Eigen/Dense>
#include <random>
#include <ctime>

namespace daramg_exploration {
// 원본 전역 (exploration.hpp의 extern과 매칭)
std::unique_ptr<octomap::OcTree> cur_tree;
std::string octomap_name_3d;
daramg_exploration::point3d kinect_orig(0,0,0);
std::shared_ptr<tf2_ros::Buffer> g_tf_buffer;
rclcpp::Logger g_logger = rclcpp::get_logger("turtlebot_exploration_3d");
} // namespace daramg_exploration

using namespace daramg_exploration;

class TurtlebotExploration3DNode : public rclcpp::Node {
public:
  TurtlebotExploration3DNode()
  : rclcpp::Node("turtlebot_exploration_3d"),
    tf_listener_(*ensure_tf_buffer())
  {
    g_logger = this->get_logger();

    // 프레임 및 IO 토픽
    map_frame_ = this->declare_parameter<std::string>("map_frame", "map");
    camera_frame_ = this->declare_parameter<std::string>("camera_frame", "mast_rgbd_link");
    cloud_topic_ = this->declare_parameter<std::string>("cloud_topic", "/mast_camera/points");
    cmd_vel_topic_ = this->declare_parameter<std::string>("cmd_vel_topic", "/cmd_vel");
    goal_mk_topic_ = this->declare_parameter<std::string>("goal_marker_topic", "/Goal_Marker");
    cand_mk_topic_ = this->declare_parameter<std::string>("candidates_topic", "/Candidate_MIs");
    frnt_mk_topic_ = this->declare_parameter<std::string>("frontiers_topic", "/Frontier_points");
    octomap_topic_ = this->declare_parameter<std::string>("octomap_topic", "octomap_3d");

    // 센서 모델 생성용 파라미터
    sensor_width_ = this->declare_parameter<int>("sensor_width", 640);
    sensor_height_ = this->declare_parameter<int>("sensor_height",480);
    h_fov_deg_ = this->declare_parameter<double>("h_fov_deg", 57.0);
    v_fov_deg_ = this->declare_parameter<double>("v_fov_deg", 43.0);
    max_range_ = this->declare_parameter<double>("max_range", 8.0);

    // exploration.hpp의 센서모덜 런타임 재설정
    set_sensor_model(sensor_width_, sensor_height_, h_fov_deg_, v_fov_deg_, max_range_);

    // 전처리 필터용 파라미터
    g_voxel_leaf = this->declare_parameter<double>("voxel_leaf", 0.05);
    g_z_clip_min = this->declare_parameter<double>("z_clip_min", -1.0);
    g_z_clip_max = this->declare_parameter<double>("z_clip_max",  5.0);
    g_r_min = this->declare_parameter<double>("r_min", 0.3);
    g_r_max = this->declare_parameter<double>("r_max", 6.0);

    // Frontier 슬라이스용 파라미터
    g_slice_z = this->declare_parameter<double>("slice_z", 0.4);
    g_slice_thickness = this->declare_parameter<double>("slice_thickness", 0.05); // octo_reso와 같게
    g_cluster_R1 = this->declare_parameter<double>("cluster_radius_R1", 0.4);

    // 후보 Viewpoint 설정용 파라미터
    yaw_samples_ = this->declare_parameter<int>("yaw_samples", 30);
    g_ring_r_min = this->declare_parameter<double>("ring_r_min", 1.0);
    g_ring_r_max = this->declare_parameter<double>("ring_r_max", 5.0);
    g_ring_r_step = this->declare_parameter<double>("ring_r_step", 0.5);
    g_min_dist_sensor = this->declare_parameter<double>("min_dist_from_sensor", 0.25);
    g_clearance_R3 = this->declare_parameter<double>("clearance_R3", 0.3);

    // MI 및 BO를 위한 파라미터
    num_of_samples_eva_ = this->declare_parameter<int>("num_of_samples_eva", 15);
    num_of_bay_ = this->declare_parameter<int>("num_of_bay", 3);
    gp_sf2_ = this->declare_parameter<double>("gp_sf2", 100.0);
    gp_ell_ = this->declare_parameter<double>("gp_ell", 3.0);
    gp_noise_ = this->declare_parameter<double>("gp_noise", 0.01);
    bo_beta_ = this->declare_parameter<double>("bo_beta", 2.4);
    normalize_mi_by_dist_ = this->declare_parameter<bool>("normalize_mi_by_distance", true);

    // 초기 스캔을 위한 파라미터
    initial_scans_ = this->declare_parameter<int>("initial_rotate_scans", 6);
    rotate_speed_ = this->declare_parameter<double>("rotate_speed", 0.6);
    rotate_dur_sec_ = this->declare_parameter<double>("rotate_duration_sec", 3.0);
    nav_timeout_sec_ = this->declare_parameter<double>("nav_timeout_sec", 40.0);
    nav_action_name_ = this->declare_parameter<std::string>("nav_action_name", "navigate_to_pose");


    // 파일 저장용 파라미터
    write_octomap_to_disk_ = this->declare_parameter<bool>("write_octomap_to_disk", false);
    octomap_prefix_ = this->declare_parameter<std::string>("octomap_filename_prefix", "Octomap3D_");
    g_write_octomap = write_octomap_to_disk_;
    g_octomap_prefix = octomap_prefix_;

    // Octomap 파일명(원본 포맷)
    {
      char buffer[80];
      std::time_t rawtime = std::time(nullptr);
      std::tm *ti = std::localtime(&rawtime);
      std::strftime(buffer, 80, "Octomap3D_%m%d_%H%M.ot", ti);
      octomap_name_3d = buffer;
    }

    // Octomap 초기화(원본 유지: octo_reso는 exploration.hpp에 고정)
    cur_tree = std::make_unique<octomap::OcTree>(octo_reso);

    // Pub/Sub
    sub_cloud_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      cloud_topic_, rclcpp::SensorDataQoS(), &kinectCallbacks);
    pub_goal_marker_ = this->create_publisher<visualization_msgs::msg::Marker>(goal_mk_topic_, 1);
    pub_candidates_  = this->create_publisher<visualization_msgs::msg::MarkerArray>(cand_mk_topic_, 1);
    pub_frontiers_ = this->create_publisher<visualization_msgs::msg::Marker>(frnt_mk_topic_, 1);
    pub_twist_ = this->create_publisher<geometry_msgs::msg::Twist>(cmd_vel_topic_, 1);
    pub_octomap_ = this->create_publisher<octomap_msgs::msg::Octomap>(octomap_topic_, 1);
  }

  void run() {
    rclcpp::executors::SingleThreadedExecutor exec;
    exec.add_node(shared_from_this());

    // 초기 회전 스캔
    for (int i=0; rclcpp::ok() && i<initial_scans_; ++i) {
      bool got_tf = false;
      while (!got_tf && rclcpp::ok()) {
        try {
          auto tf = g_tf_buffer->lookupTransform(map_frame_, camera_frame_, tf2::TimePointZero);
          kinect_orig = point3d(tf.transform.translation.x,
                                tf.transform.translation.y,
                                tf.transform.translation.z);
          got_tf = true;
        } catch (const tf2::TransformException &e) {
          RCLCPP_WARN(g_logger, "Wait for tf: camera_frame (%s)", e.what());
        }
        rclcpp::sleep_for(std::chrono::milliseconds(50));
      }

      exec.spin_some();

      octomap_msgs::msg::Octomap msg;
      octomap_msgs::binaryMapToMsg(*cur_tree, msg);
      msg.header.frame_id = map_frame_;
      msg.header.stamp = this->now();
      pub_octomap_->publish(msg);

      geometry_msgs::msg::Twist twist;
      RCLCPP_WARN(g_logger, "Rotate...");
      rclcpp::Time t0 = this->now();
      while ((this->now() - t0) < rclcpp::Duration::from_seconds(rotate_dur_sec_) && rclcpp::ok()) {
        twist.angular.z = rotate_speed_;
        pub_twist_->publish(twist);
        rclcpp::sleep_for(std::chrono::milliseconds(50));
      }
      twist.angular.z = 0.0;
      pub_twist_->publish(twist);
    }

    int robot_step_counter = 0;

    while (rclcpp::ok()) {
      auto frontier_groups = extractFrontierPoints(cur_tree);

      visualization_msgs::msg::Marker frontier_mk;
      frontier_mk.header.frame_id = map_frame_;
      frontier_mk.header.stamp = this->now();
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
      pub_frontiers_->publish(frontier_mk);

      auto candidates = extractCandidateViewPoints(frontier_groups, kinect_orig, yaw_samples_);
      std::shuffle(candidates.begin(), candidates.end(), std::mt19937{std::random_device{}()});
      auto gp_test_poses = candidates;

      RCLCPP_INFO(g_logger, "Candidate View Points: %zu Generated, %d evaluating...",
                  candidates.size(), num_of_samples_eva_);
      int temp_size = static_cast<int>(candidates.size()) - 3;
      if (temp_size < 1) {
        RCLCPP_ERROR(g_logger, "Very few candidates; finish.");
        rclcpp::shutdown();
        return;
      }

      candidates.resize(std::min(num_of_samples_eva_, temp_size));
      frontier_groups.clear();

      std::vector<double> MIs(candidates.size(), 0.0);
      double before = countFreeVolume(cur_tree);
      double t_begin = this->now().seconds();

      #pragma omp parallel for
      for (int i = 0; i < static_cast<int>(candidates.size()); ++i) {
        auto c = candidates[i];
        auto hits = castSensorRays(cur_tree, c.first, c.second);
        double mi = calc_MI(cur_tree, c.first, hits, before);
        if (normalize_mi_by_dist_) {
          double dx = c.first.x()-kinect_orig.x(), dy = c.first.y()-kinect_orig.y();
          mi /= std::hypot(dx,dy);
        }
        MIs[i] = mi;
      }

      GPRegressor g(gp_sf2_, 3, gp_noise_); // 원본 인터페이스: (sf2, dim, noise)
      for (int bay_itr=0; bay_itr<num_of_bay_; ++bay_itr) {
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

        std::vector<double> bay_acq_fun(gp_test_poses.size());
        for (int i=0;i<(int)gp_test_poses.size();++i)
          bay_acq_fun[i] = gp_mean_MI(i) + bo_beta_ * gp_var_MI(i);

        auto idx_acq = sort_MIs(bay_acq_fun);
        auto c = gp_test_poses[idx_acq[0]];
        auto hits = castSensorRays(cur_tree, c.first, c.second);
        candidates.push_back(c);
        MIs.push_back(calc_MI(cur_tree, c.first, hits, before));
        gp_test_poses.erase(gp_test_poses.begin()+idx_acq[0]);
      }

      double t_end = this->now().seconds();
      RCLCPP_INFO(g_logger, "MI eval: %.3f s", (t_end - t_begin));

      auto idx_MI = sort_MIs(MIs);

      tf2::Quaternion MI_heading; MI_heading.setRPY(0.0, -PI/2, 0.0); MI_heading.normalize();
      visualization_msgs::msg::MarkerArray cand_arr;
      cand_arr.markers.resize(candidates.size());
      const double denom = std::max(1e-9, MIs[idx_MI[0]]);
      for (int i=0;i<(int)candidates.size();++i) {
        auto &mk = cand_arr.markers[i];
        mk.header.frame_id = map_frame_; mk.header.stamp = this->now();
        mk.ns="candidates"; mk.id=i;
        mk.type=visualization_msgs::msg::Marker::ARROW; mk.action=visualization_msgs::msg::Marker::ADD;
        mk.pose.position.x = candidates[i].first.x();
        mk.pose.position.y = candidates[i].first.y();
        mk.pose.position.z = candidates[i].first.z();
        mk.pose.orientation.x = MI_heading.x();
        mk.pose.orientation.y = MI_heading.y();
        mk.pose.orientation.z = MI_heading.z();
        mk.pose.orientation.w = MI_heading.w();
        mk.scale.x = 2.0 * MIs[i] / denom;
        mk.scale.y = mk.scale.z = 0.2;
        mk.color.a = std::min(1.0, MIs[i] / denom);
        mk.color.r = 1.0; mk.color.g = 0.55; mk.color.b = 0.22;
      }
      pub_candidates_->publish(cand_arr);
      cand_arr.markers.clear();

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

        visualization_msgs::msg::Marker goal;
        goal.header.frame_id=map_frame_; goal.header.stamp=this->now();
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
        pub_goal_marker_->publish(goal);

        // NOTE: nav_timeout_sec_/nav_action_name_을 navigation_utils에 전달하려면
        // navigation_utils에 setter 또는 확장 오버로드가 있어야 함.
        arrived = navigation_utils::goToDest(next_vp, Goal_heading);

        if (arrived) {
          bool got_tf = false;
          while (!got_tf && rclcpp::ok()) {
            try {
              auto tf = g_tf_buffer->lookupTransform(map_frame_, camera_frame_, tf2::TimePointZero);
              kinect_orig = point3d(tf.transform.translation.x,
                                    tf.transform.translation.y,
                                    tf.transform.translation.z);
              got_tf = true;
            } catch (const tf2::TransformException &e) {
              RCLCPP_WARN(g_logger, "Wait for tf: camera_frame (%s)", e.what());
            }
            rclcpp::sleep_for(std::chrono::milliseconds(50));
          }

          exec.spin_some();
          RCLCPP_INFO(g_logger, "Succeed, new Map Free Volume: %.3f", countFreeVolume(cur_tree));
          robot_step_counter++;

          octomap_msgs::msg::Octomap msg;
          octomap_msgs::binaryMapToMsg(*cur_tree, msg);
          msg.header.frame_id = map_frame_;
          msg.header.stamp = this->now();
          pub_octomap_->publish(msg);
        } else {
          RCLCPP_WARN(g_logger, "Failed to drive to the %d-th goal, switch next..", idx_ptr);
          idx_ptr++;
          if (idx_ptr > (int)MIs.size()) {
            RCLCPP_ERROR(g_logger, "None of the goals valid, shutting down.");
            rclcpp::shutdown();
          }
        }
      }
    }
  }

private:
  std::shared_ptr<tf2_ros::Buffer> ensure_tf_buffer() {
    if (!g_tf_buffer) g_tf_buffer = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    return g_tf_buffer;
  }

  // ---- params ----
  std::string map_frame_, camera_frame_, cloud_topic_, cmd_vel_topic_, goal_mk_topic_, cand_mk_topic_, frnt_mk_topic_, octomap_topic_;
  int sensor_width_{128}, sensor_height_{96};
  double h_fov_deg_{57.0}, v_fov_deg_{43.0}, max_range_{6.0};
  int yaw_samples_{30}, num_of_samples_eva_{15}, num_of_bay_{3}, initial_scans_{6};
  double gp_sf2_{100.0}, gp_ell_{3.0}, gp_noise_{0.01}, bo_beta_{2.4};
  bool normalize_mi_by_dist_{true};
  double rotate_speed_{0.6}, rotate_dur_sec_{3.0}, nav_timeout_sec_{120.0};
  std::string nav_action_name_;
  bool write_octomap_to_disk_{false};
  std::string octomap_prefix_;

  // ROS I/O
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_cloud_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr pub_goal_marker_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_candidates_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr pub_frontiers_;
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr pub_twist_;
  rclcpp::Publisher<octomap_msgs::msg::Octomap>::SharedPtr pub_octomap_;
  tf2_ros::TransformListener tf_listener_;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<TurtlebotExploration3DNode>();
  node->run();
  rclcpp::shutdown();
  return 0;
}