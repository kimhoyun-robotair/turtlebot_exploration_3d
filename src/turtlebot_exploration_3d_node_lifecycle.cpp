// 라이프사이클 + 타이머 스텝 머신 적용 (원본 로직/이름/변수 유지)
// - exploration.hpp의 전역/함수는 그대로 사용
// - navigation_utils::goToDest(point3d, tf2::Quaternion) 블로킹 → std::async로 비동기화
// - 긴 while(run) 루프를 timer step() 상태머신으로 분해

#include "turtlebot_exploration_3d/exploration.hpp"
#include "turtlebot_exploration_3d/navigation_utils.hpp"
#include "turtlebot_exploration_3d/gpregressor.h"
#include "turtlebot_exploration_3d/covMaterniso3.h"

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/lifecycle_node.hpp>
#include <rclcpp_lifecycle/lifecycle_publisher.hpp>
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
#include <future>
#include <cmath>

namespace daramg_exploration {
// exploration.hpp의 extern과 매칭되는 전역 정의
std::unique_ptr<octomap::OcTree> cur_tree;
std::string octomap_name_3d;
daramg_exploration::point3d kinect_orig(0,0,0);
std::shared_ptr<tf2_ros::Buffer> g_tf_buffer;
rclcpp::Logger g_logger = rclcpp::get_logger("turtlebot_exploration_3d");
} // namespace daramg_exploration

using namespace daramg_exploration;

class TurtlebotExploration3DNode
: public rclcpp_lifecycle::LifecycleNode
{
public:
  using LNI = rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface;
  using CallbackReturn = LNI::CallbackReturn;

  TurtlebotExploration3DNode()
  : rclcpp_lifecycle::LifecycleNode("turtlebot_exploration_3d"),
    tf_listener_(*ensure_tf_buffer())
  {
    g_logger = this->get_logger();

    // ===== 파라미터 =====
    // Frame & Topics
    map_frame_ = this->declare_parameter<std::string>("map_frame", "map");
    camera_frame_ = this->declare_parameter<std::string>("camera_frame", "mast_rgbd_link");
    cloud_topic_ = this->declare_parameter<std::string>("cloud_topic", "/mast_camera/points");
    cmd_vel_topic_ = this->declare_parameter<std::string>("cmd_vel_topic", "/cmd_vel");
    goal_mk_topic_ = this->declare_parameter<std::string>("goal_marker_topic", "/Goal_Marker");
    cand_mk_topic_ = this->declare_parameter<std::string>("candidates_topic", "/Candidate_MIs");
    frnt_mk_topic_ = this->declare_parameter<std::string>("frontiers_topic", "/Frontier_points");
    octomap_topic_ = this->declare_parameter<std::string>("octomap_topic", "octomap_3d");

    // Sensor model
    sensor_width_ = this->declare_parameter<int>("sensor_width", 640);
    sensor_height_ = this->declare_parameter<int>("sensor_height", 480);
    h_fov_deg_ = this->declare_parameter<double>("h_fov_deg", 57.0);
    v_fov_deg_ = this->declare_parameter<double>("v_fov_deg", 43.0);
    max_range_ = this->declare_parameter<double>("max_range", 8.0);

    // exploration.hpp의 센서 모델 런타임 재설정
    set_sensor_model(sensor_width_, sensor_height_, h_fov_deg_, v_fov_deg_, max_range_);

    // 전처리/후보/프런티어 관련 글로벌( exploration.hpp 안의 inline 변수 사용 )
    g_voxel_leaf = this->declare_parameter<double>("voxel_leaf", 0.05);
    g_z_clip_min = this->declare_parameter<double>("z_clip_min", -1.0);
    g_z_clip_max = this->declare_parameter<double>("z_clip_max",  5.0);
    g_r_min = this->declare_parameter<double>("r_min", 0.3);
    g_r_max = this->declare_parameter<double>("r_max", 6.0);

    g_slice_z = this->declare_parameter<double>("slice_z", 0.4);
    g_slice_thickness = this->declare_parameter<double>("slice_thickness", 0.05);
    g_cluster_R1 = this->declare_parameter<double>("cluster_radius_R1", 0.4);

    yaw_samples_ = this->declare_parameter<int>("yaw_samples", 30);
    g_ring_r_min = this->declare_parameter<double>("ring_r_min", 1.0);
    g_ring_r_max = this->declare_parameter<double>("ring_r_max", 5.0);
    g_ring_r_step = this->declare_parameter<double>("ring_r_step", 0.5);
    g_min_dist_sensor = this->declare_parameter<double>("min_dist_from_sensor", 0.25);
    g_clearance_R3 = this->declare_parameter<double>("clearance_R3", 0.3);

    // MI/BO
    num_of_samples_eva_ = this->declare_parameter<int>("num_of_samples_eva", 15);
    num_of_bay_ = this->declare_parameter<int>("num_of_bay", 3);
    gp_sf2_ = this->declare_parameter<double>("gp_sf2", 100.0);
    gp_ell_ = this->declare_parameter<double>("gp_ell", 3.0);   // (사용 안해도 원본 호환 위해 남김)
    gp_noise_  = this->declare_parameter<double>("gp_noise", 0.01);
    bo_beta_ = this->declare_parameter<double>("bo_beta", 2.4);
    normalize_mi_by_dist_ = this->declare_parameter<bool>("normalize_mi_by_distance", true);

    // 초기 회전 스캔
    initial_scans_ = this->declare_parameter<int>("initial_rotate_scans", 6);
    rotate_speed_ = this->declare_parameter<double>("rotate_speed", 0.6);
    rotate_dur_sec_  = this->declare_parameter<double>("rotate_duration_sec", 3.0);

    // (선택) 내비 타임아웃(여기서는 비동기 future 확인만 하므로 직접 사용 안함)
    nav_timeout_sec_ = this->declare_parameter<double>("nav_timeout_sec", 40.0);

    // 파일 저장 관련( exploration.hpp의 inline 전역에 반영 )
    write_octomap_to_disk_ = this->declare_parameter<bool>("write_octomap_to_disk", false);
    octomap_prefix_ = this->declare_parameter<std::string>("octomap_filename_prefix", "Octomap3D_");
    g_write_octomap = write_octomap_to_disk_;
    g_octomap_prefix = octomap_prefix_;

    // 타이머에서 MI를 청크로 나눠 계산
    mi_chunk_ = this->declare_parameter<int>("mi_chunk", 4);

    // Octomap 파일명(원본 포맷)
    {
      char buffer[80];
      std::time_t raw = std::time(nullptr);
      std::tm *ti = std::localtime(&raw);
      std::strftime(buffer, 80, "Octomap3D_%m%d_%H%M.ot", ti);
      octomap_name_3d = buffer;
    }

    // Octomap 초기화 (원본: octo_reso는 exploration.hpp 상수)
    cur_tree = std::make_unique<octomap::OcTree>(octo_reso);

    // ===== IO 생성 =====
    // PointCloud 입력은 라이프사이클 상태와 무관하게 구독(필요하면 활성 상태에서만 쓰도록 별 플래그로 관리 가능)
    sub_cloud_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      cloud_topic_, rclcpp::SensorDataQoS(), &kinectCallbacks);

    // Lifecycle Publisher들
    pub_goal_marker_ = this->create_publisher<visualization_msgs::msg::Marker>(goal_mk_topic_, 10);
    pub_candidates_  = this->create_publisher<visualization_msgs::msg::MarkerArray>(cand_mk_topic_, 10);
    pub_frontiers_ = this->create_publisher<visualization_msgs::msg::Marker>(frnt_mk_topic_, 10);
    pub_twist_ = this->create_publisher<geometry_msgs::msg::Twist>(cmd_vel_topic_, 10);
    pub_octomap_ = this->create_publisher<octomap_msgs::msg::Octomap>(octomap_topic_, 10);
  }

  // ===== Lifecycle Hooks =====
  CallbackReturn on_configure(const rclcpp_lifecycle::State&) override {
    // 상태 초기화
    state_ = State::INIT_TF;
    init_scan_done_ = 0;
    rotating_ = false;
    candidates_.clear();
    gp_test_poses_.clear();
    MIs_.clear();
    return CallbackReturn::SUCCESS;
  }

  CallbackReturn on_activate(const rclcpp_lifecycle::State&) override {
    pub_goal_marker_->on_activate();
    pub_candidates_->on_activate();
    pub_frontiers_->on_activate();
    pub_twist_->on_activate();
    pub_octomap_->on_activate();

    RCLCPP_INFO(get_logger(), "[LC] Activated. Starting exploration timer loop.");

    // 50ms 주기로 step() 호출 (비블로킹)
    timer_ = this->create_wall_timer(
      std::chrono::milliseconds(50),
      std::bind(&TurtlebotExploration3DNode::step, this));

    return CallbackReturn::SUCCESS;
  }

  CallbackReturn on_deactivate(const rclcpp_lifecycle::State&) override {
    if (timer_) timer_.reset();
    pub_goal_marker_->on_deactivate();
    pub_candidates_->on_deactivate();
    pub_frontiers_->on_deactivate();
    pub_twist_->on_deactivate();
    pub_octomap_->on_deactivate();
    // 네비 비동기 작업 정리
    if (nav_future_.valid()) {
      // 필요 시 취소 로직 추가 가능 (여기서는 단순 무시)
    }
    return CallbackReturn::SUCCESS;
  }

  CallbackReturn on_cleanup(const rclcpp_lifecycle::State&) override {
    candidates_.clear();
    gp_test_poses_.clear();
    MIs_.clear();
    return CallbackReturn::SUCCESS;
  }

  CallbackReturn on_shutdown(const rclcpp_lifecycle::State&) override {
    if (timer_) timer_.reset();
    return CallbackReturn::SUCCESS;
  }

private:
  // ===== 타이머 기반 상태 머신 =====
  enum class State {
    INIT_TF,
    INIT_ROTATE,
    PLAN_FRONTIERS,
    PLAN_MI_CHUNK,
    BO_STEP,
    SELECT_AND_SEND_GOAL,
    WAIT_NAV,
    UPDATE_MAP
  };

  void step() {
    switch (state_) {
      case State::INIT_TF:              stepInitTf(); break;
      case State::INIT_ROTATE:          stepInitRotate(); break;
      case State::PLAN_FRONTIERS:       stepPlanFrontiers(); break;
      case State::PLAN_MI_CHUNK:        stepPlanMiChunk(); break;
      case State::BO_STEP:              stepBoStep(); break;
      case State::SELECT_AND_SEND_GOAL: stepSelectAndSend(); break;
      case State::WAIT_NAV:             stepWaitNav(); break;
      case State::UPDATE_MAP:           stepUpdateMap(); break;
    }
  }

  void stepInitTf() {
    if (!g_tf_buffer->canTransform(map_frame_, camera_frame_, tf2::TimePointZero, tf2::durationFromSec(0.0))) {
      RCLCPP_DEBUG(get_logger(), "[INIT_TF] Waiting for TF %s -> %s", camera_frame_.c_str(), map_frame_.c_str());
      return;
    }
    try {
      auto tf = g_tf_buffer->lookupTransform(map_frame_, camera_frame_, tf2::TimePointZero);
      kinect_orig = point3d(tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z);
      RCLCPP_INFO(get_logger(), "[INIT_TF] Sensor origin: (%.3f, %.3f, %.3f)", kinect_orig.x(), kinect_orig.y(), kinect_orig.z());
      rotate_start_ = now();
      rotating_ = true;
      state_ = State::INIT_ROTATE;
    } catch (const tf2::TransformException &e) {
      RCLCPP_WARN(get_logger(), "[INIT_TF] TF lookup failed: %s", e.what());
    }
  }

  void stepInitRotate() {
    if (init_scan_done_ >= initial_scans_) {
      RCLCPP_INFO(get_logger(), "[INIT_ROTATE] Completed %d initial scans.", initial_scans_);
      state_ = State::PLAN_FRONTIERS;
      return;
    }
    rclcpp::Duration elapsed = now() - rotate_start_;
    geometry_msgs::msg::Twist twist;
    if (elapsed < rclcpp::Duration::from_seconds(rotate_dur_sec_)) {
      twist.angular.z = rotate_speed_;
      pub_twist_->publish(twist);
      RCLCPP_DEBUG(get_logger(), "[INIT_ROTATE] Rotating... (scan %d/%d) elapsed=%.2f",
                  init_scan_done_+1, initial_scans_, elapsed.seconds());
    } else {
      twist.angular.z = 0.0;
      pub_twist_->publish(twist);

      octomap_msgs::msg::Octomap msg;
      octomap_msgs::binaryMapToMsg(*cur_tree, msg);
      msg.header.frame_id = map_frame_;
      msg.header.stamp = now();
      pub_octomap_->publish(msg);

      RCLCPP_INFO(get_logger(), "[INIT_ROTATE] Scan %d done. Octomap snapshot published.", init_scan_done_+1);

      init_scan_done_++;
      rotate_start_ = now();
    }
  }

  void stepPlanFrontiers() {
    auto frontier_groups = extractFrontierPoints(cur_tree);

    size_t groups = frontier_groups.size();
    size_t points = 0;
    for (const auto& g : frontier_groups) points += g.size();

    last_frontier_groups_ = groups;
    last_frontier_points_ = points;

    RCLCPP_INFO(get_logger(), "[PLAN] Frontiers: %zu groups / %zu points (slice_z=%.2f ± %.2f)",
                groups, points, g_slice_z, g_slice_thickness);

    // 시각화
    visualization_msgs::msg::Marker mk;
    mk.header.frame_id = map_frame_;
    mk.header.stamp = now();
    mk.ns = "frontier_points_array";
    mk.id = 0;
    mk.type = visualization_msgs::msg::Marker::CUBE_LIST;
    mk.action = visualization_msgs::msg::Marker::ADD;
    mk.scale.x = mk.scale.y = mk.scale.z = octo_reso;
    mk.color.a = 1.0; mk.color.r = 1.0; mk.color.g = 0.0; mk.color.b = 0.0;
    for (const auto &grp : frontier_groups)
      for (const auto &p : grp) {
        geometry_msgs::msg::Point q; q.x=p.x(); q.y=p.y(); q.z=p.z()+octo_reso;
        mk.points.push_back(q);
      }
    pub_frontiers_->publish(mk);

    candidates_ = extractCandidateViewPoints(frontier_groups, kinect_orig, yaw_samples_);
    std::shuffle(candidates_.begin(), candidates_.end(), std::mt19937{std::random_device{}()});
    gp_test_poses_ = candidates_;

    RCLCPP_INFO(get_logger(), "[PLAN] Candidates generated: %zu (yaw_samples=%d, r=[%.1f, %.1f], step=%.1f)",
                candidates_.size(), yaw_samples_, g_ring_r_min, g_ring_r_max, g_ring_r_step);

    const int temp_size = static_cast<int>(candidates_.size()) - 3;
    if (temp_size < 1) {
      RCLCPP_ERROR(get_logger(), "[PLAN] Very few candidates; deactivating.");
      transition_to_inactive_();
      return;
    }
    if ((int)candidates_.size() > num_of_samples_eva_) {
      candidates_.resize(std::min(num_of_samples_eva_, temp_size));
    }
    RCLCPP_INFO(get_logger(), "[PLAN] Evaluating %zu candidates (num_of_samples_eva=%d).",
                candidates_.size(), num_of_samples_eva_);

    before_volume_ = countFreeVolume(cur_tree);
    last_before_volume_ = before_volume_;
    RCLCPP_INFO(get_logger(), "[PLAN] Baseline free-volume(before): %.3f", before_volume_);

    MIs_.assign(candidates_.size(), 0.0);
    mi_index_ = 0;
    mi_eval_start_ = now();

    state_ = State::PLAN_MI_CHUNK;
  }

  void stepPlanMiChunk() {
    const int N = static_cast<int>(candidates_.size());
    const int end = std::min(N, mi_index_ + mi_chunk_);
    for (int i = mi_index_; i < end; ++i) {
      auto &c = candidates_[i];
      auto hits = castSensorRays(cur_tree, c.first, c.second);
      double mi = calc_MI(cur_tree, c.first, hits, before_volume_);
      if (normalize_mi_by_dist_) {
        double dx = c.first.x()-kinect_orig.x(), dy = c.first.y()-kinect_orig.y();
        mi /= std::hypot(dx,dy);
      }
      MIs_[i] = mi;
      RCLCPP_DEBUG(get_logger(), "[MI] cand[%d] pos(%.2f,%.2f,%.2f) yaw=%.2f -> MI=%.4f",
                  i, c.first.x(), c.first.y(), c.first.z(), c.second.z(), mi);
    }
    RCLCPP_INFO(get_logger(), "[MI] Progress: %d/%d evaluated.", end, N);
    mi_index_ = end;

    if (mi_index_ >= N) {
      const double dt = (now() - mi_eval_start_).seconds();
      RCLCPP_INFO(get_logger(), "[MI] Completed MI evaluation of %d cand in %.3f s.", N, dt);
      bo_iter_ = 0;
      state_ = (num_of_bay_ > 0) ? State::BO_STEP : State::SELECT_AND_SEND_GOAL;
      if (num_of_bay_ > 0) {
        bo_start_ = now();
        RCLCPP_INFO(get_logger(), "[BO] Starting Bayesian Optimization (%d iterations).", num_of_bay_);
      }
    }
  }

  void stepBoStep() {
    GPRegressor g(gp_sf2_, 3, gp_noise_);
    Eigen::MatrixXf gp_train_x(candidates_.size(), 3),
                    gp_train_label(candidates_.size(), 1),
                    gp_test_x(gp_test_poses_.size(), 3);

    for (int i=0;i<(int)candidates_.size();++i) {
      gp_train_x(i,0)=candidates_[i].first.x();
      gp_train_x(i,1)=candidates_[i].first.y();
      gp_train_x(i,2)=candidates_[i].second.z();
      gp_train_label(i)=MIs_[i];
    }
    for (int i=0;i<(int)gp_test_poses_.size();++i) {
      gp_test_x(i,0)=gp_test_poses_[i].first.x();
      gp_test_x(i,1)=gp_test_poses_[i].first.y();
      gp_test_x(i,2)=gp_test_poses_[i].second.z();
    }

    Eigen::MatrixXf gp_mean_MI, gp_var_MI;
    g.train(gp_train_x, gp_train_label);
    g.test(gp_test_x, gp_mean_MI, gp_var_MI);

    std::vector<double> acq(gp_test_poses_.size());
    for (int i=0;i<(int)gp_test_poses_.size();++i)
      acq[i] = gp_mean_MI(i) + bo_beta_ * gp_var_MI(i);

    auto order = sort_MIs(acq);
    int best_i = order[0];
    double m = gp_mean_MI(best_i);
    double v = gp_var_MI(best_i);
    double a = acq[best_i];
    auto c = gp_test_poses_[best_i];

    RCLCPP_INFO(get_logger(), "[BO] iter %d/%d -> pick #%d pos(%.2f,%.2f,%.2f) yaw=%.2f mean=%.4f var=%.4f acq=%.4f",
                bo_iter_+1, num_of_bay_, best_i, c.first.x(), c.first.y(), c.first.z(), c.second.z(), m, v, a);

    auto hits = castSensorRays(cur_tree, c.first, c.second);
    candidates_.push_back(c);
    double mi_new = calc_MI(cur_tree, c.first, hits, before_volume_);
    MIs_.push_back(mi_new);
    gp_test_poses_.erase(gp_test_poses_.begin()+best_i);

    bo_iter_++;
    if (bo_iter_ >= num_of_bay_) {
      double dt_bo = (now() - bo_start_).seconds();
      RCLCPP_INFO(get_logger(), "[BO] Finished %d iterations in %.3f s. Total candidates now %zu.",
                  num_of_bay_, dt_bo, candidates_.size());
      state_ = State::SELECT_AND_SEND_GOAL;
    }
  }


  void stepSelectAndSend() {
    auto idx = sort_MIs(MIs_);

    // 상위 5개 요약
    int topk = std::min<int>(5, (int)idx.size());
    std::string bests;
    for (int i=0;i<topk;++i) {
      int j = idx[i];
      bests += " #" + std::to_string(j) + "(MI=" + std::to_string(MIs_[j]) + ")";
    }
    RCLCPP_INFO(get_logger(), "[SELECT] Top MI%s", bests.c_str());

    // 후보 마커 발행
    tf2::Quaternion MI_heading; MI_heading.setRPY(0.0, -M_PI/2, 0.0); MI_heading.normalize();
    visualization_msgs::msg::MarkerArray arr;
    arr.markers.resize(candidates_.size());
    const double denom = std::max(1e-9, MIs_[idx[0]]);
    for (int i=0;i<(int)candidates_.size();++i) {
      auto &mk = arr.markers[i];
      mk.header.frame_id = map_frame_; mk.header.stamp = now();
      mk.ns="candidates"; mk.id=i;
      mk.type=visualization_msgs::msg::Marker::ARROW; mk.action=visualization_msgs::msg::Marker::ADD;
      mk.pose.position.x = candidates_[i].first.x();
      mk.pose.position.y = candidates_[i].first.y();
      mk.pose.position.z = candidates_[i].first.z();
      mk.pose.orientation.x = MI_heading.x();
      mk.pose.orientation.y = MI_heading.y();
      mk.pose.orientation.z = MI_heading.z();
      mk.pose.orientation.w = MI_heading.w();
      mk.scale.x = 2.0 * MIs_[i] / denom;
      mk.scale.y = mk.scale.z = 0.2;
      mk.color.a = std::min(1.0, MIs_[i] / denom);
      mk.color.r = 1.0; mk.color.g = 0.55; mk.color.b = 0.22;
    }
    pub_candidates_->publish(arr);
    RCLCPP_INFO(get_logger(), "[SELECT] Candidate markers published: %zu", candidates_.size());

    // 최상위부터 내비 전송 준비
    sorted_idx_ = std::move(idx);
    idx_ptr_ = 0;
    sendCurrentGoal_();
    arrival_entropy_logged_ = false;   // 이번 주행에 도착하면 한 번만 entropy 로그
    state_ = State::WAIT_NAV;
  }


  void stepWaitNav() {
    if (!nav_future_.valid()) {
      sendCurrentGoal_();
      return;
    }
    using namespace std::chrono_literals;
    if (nav_future_.wait_for(0ms) == std::future_status::ready) {
      bool arrived = nav_future_.get();
      if (arrived) {
        state_ = State::UPDATE_MAP;
      } else {
        idx_ptr_++;
        if (idx_ptr_ >= (int)sorted_idx_.size()) {
          RCLCPP_ERROR(get_logger(), "None of the goals valid; replanning.");
          state_ = State::PLAN_FRONTIERS;
        } else {
          sendCurrentGoal_();
        }
      }
    }
  }

  void stepUpdateMap() {
    // 도착 후 센서 위치 갱신 시도
    if (g_tf_buffer->canTransform(map_frame_, camera_frame_, tf2::TimePointZero, tf2::durationFromSec(0.0))) {
      try {
        auto tf = g_tf_buffer->lookupTransform(map_frame_, camera_frame_, tf2::TimePointZero);
        kinect_orig = point3d(tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z);
        RCLCPP_INFO(get_logger(), "[MAP] Updated sensor origin after arrival: (%.3f, %.3f, %.3f)",
                    kinect_orig.x(), kinect_orig.y(), kinect_orig.z());
      } catch (...) {
        RCLCPP_WARN(get_logger(), "[MAP] Failed to update sensor origin after arrival (TF).");
      }
    }

    // 도착 시 1회만 Entropy(=free volume) 출력
    if (!arrival_entropy_logged_) {
      double free_vol = countFreeVolume(cur_tree);
      RCLCPP_INFO(get_logger(), "Entropy(3d map) : %.3f", free_vol);
      arrival_entropy_logged_ = true;
    }

    octomap_msgs::msg::Octomap msg;
    octomap_msgs::binaryMapToMsg(*cur_tree, msg);
    msg.header.frame_id = map_frame_;
    msg.header.stamp = now();
    pub_octomap_->publish(msg);

    RCLCPP_INFO(get_logger(), "[MAP] Octomap published after arrival. Replanning next step...");

    state_ = State::PLAN_FRONTIERS;
  }

  void sendCurrentGoal_() {
    if (idx_ptr_ >= (int)sorted_idx_.size()) return;
    const auto &best = candidates_[sorted_idx_[idx_ptr_]];
    point3d next_vp(best.first.x(), best.first.y(), best.first.z());
    tf2::Quaternion heading; heading.setRPY(0,0,best.second.z()); heading.normalize();

    // goal marker
    visualization_msgs::msg::Marker goal;
    goal.header.frame_id=map_frame_; goal.header.stamp=now();
    goal.ns="goal_marker"; goal.id=0;
    goal.type=visualization_msgs::msg::Marker::ARROW; goal.action=visualization_msgs::msg::Marker::ADD;
    goal.pose.position.x = next_vp.x();
    goal.pose.position.y = next_vp.y();
    goal.pose.position.z = 1.0;
    goal.pose.orientation.x = heading.x();
    goal.pose.orientation.y = heading.y();
    goal.pose.orientation.z = heading.z();
    goal.pose.orientation.w = heading.w();
    goal.scale.x=1.0; goal.scale.y=0.3; goal.scale.z=0.3;
    goal.color.a=1.0; goal.color.r=1.0; goal.color.g=0.0; goal.color.b=0.0;
    pub_goal_marker_->publish(goal);

    // 블로킹 goToDest() → 비동기 실행
    nav_future_ = std::async(std::launch::async, [next_vp, heading]() {
      // navigation_utils에 (point3d, quat, timeout) 오버로드가 있으면 그걸로 바꿔도 됨.
      return navigation_utils::goToDest(next_vp, heading);
    });
  }

  // 활성→비활성 전이 유틸 (에러 시 사용)
  void transition_to_inactive_() {
    // 여기서는 간단히 deactivate → activate 재시작도 가능하지만,
    // 일단은 deactivate만 호출 (외부에서 재활성화)
    this->deactivate();
  }

  // TF Buffer
  std::shared_ptr<tf2_ros::Buffer> ensure_tf_buffer() {
    if (!g_tf_buffer) g_tf_buffer = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    return g_tf_buffer;
  }

private:
  // ===== 파라미터 멤버 =====
  std::string map_frame_, camera_frame_, cloud_topic_, cmd_vel_topic_;
  std::string goal_mk_topic_, cand_mk_topic_, frnt_mk_topic_, octomap_topic_;
  int sensor_width_{640}, sensor_height_{480};
  double h_fov_deg_{57.0}, v_fov_deg_{43.0}, max_range_{8.0};
  int yaw_samples_{30}, num_of_samples_eva_{15}, num_of_bay_{3}, initial_scans_{6};
  double gp_sf2_{100.0}, gp_ell_{3.0}, gp_noise_{0.01}, bo_beta_{2.4};
  bool normalize_mi_by_dist_{true};
  double rotate_speed_{0.6}, rotate_dur_sec_{3.0}, nav_timeout_sec_{40.0};
  bool write_octomap_to_disk_{false};
  std::string octomap_prefix_;
  int mi_chunk_{4};

  // --- logging helpers / last-knowns (ADD THESE) ---
  rclcpp::Time mi_eval_start_;
  rclcpp::Time bo_start_;
  size_t last_frontier_groups_{0};
  size_t last_frontier_points_{0};
  bool arrival_entropy_logged_{false};  // print free-volume once per arrival
  double last_before_volume_{0.0};

  // ===== ROS IO =====
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_cloud_;
  rclcpp_lifecycle::LifecyclePublisher<visualization_msgs::msg::Marker>::SharedPtr pub_goal_marker_;
  rclcpp_lifecycle::LifecyclePublisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_candidates_;
  rclcpp_lifecycle::LifecyclePublisher<visualization_msgs::msg::Marker>::SharedPtr pub_frontiers_;
  rclcpp_lifecycle::LifecyclePublisher<geometry_msgs::msg::Twist>::SharedPtr pub_twist_;
  rclcpp_lifecycle::LifecyclePublisher<octomap_msgs::msg::Octomap>::SharedPtr pub_octomap_;
  tf2_ros::TransformListener tf_listener_;
  rclcpp::TimerBase::SharedPtr timer_;

  // ===== 상태 머신 =====
  enum State state_{State::INIT_TF};

  // INIT_ROTATE
  int init_scan_done_{0};
  rclcpp::Time rotate_start_;
  bool rotating_{false};

  // PLAN
  std::vector<std::pair<point3d, point3d>> candidates_;
  std::vector<std::pair<point3d, point3d>> gp_test_poses_;
  std::vector<double> MIs_;
  double before_volume_{0.0};
  int mi_index_{0};
  int bo_iter_{0};

  // NAV
  std::vector<int> sorted_idx_;
  int idx_ptr_{0};
  std::future<bool> nav_future_;
};

// main: Lifecycle 노드 등록(일반 실행도 가능)
int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<TurtlebotExploration3DNode>();
  rclcpp::executors::SingleThreadedExecutor exec;
  exec.add_node(node->get_node_base_interface());
  exec.spin();
  rclcpp::shutdown();
  return 0;
}
