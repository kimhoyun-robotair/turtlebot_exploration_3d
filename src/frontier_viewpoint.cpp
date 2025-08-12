#include "turtlebot_exploration_3d/map_store.hpp"

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <rclcpp_action/rclcpp_action.hpp>

#include <std_msgs/msg/u_int64.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <nav2_msgs/action/navigate_to_pose.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <unordered_set>
#include <random>
#include <algorithm>
#include <cmath>

namespace explore {

using NavigateToPose = nav2_msgs::action::NavigateToPose;
using point3d = octomap::point3d;

class FrontierViewpointNode : public rclcpp::Node {
public:
  explicit FrontierViewpointNode(const rclcpp::NodeOptions &opts)
  : rclcpp::Node("frontier_viewpoint", opts)
  {
    // params
    map_frame_   = declare_parameter<std::string>("map_frame", "map");
    yaw_samples_ = declare_parameter<int>("yaw_samples", 30);
    ring_r_min_  = declare_parameter<double>("ring_r_min", 1.0);
    ring_r_max_  = declare_parameter<double>("ring_r_max", 5.0);
    ring_r_step_ = declare_parameter<double>("ring_r_step", 0.5);
    keep_top_k_  = declare_parameter<int>("keep_top_k", 15);

    width_ = declare_parameter<int>("width", 320);
    height_ = declare_parameter<int>("height", 240);
    double hdeg = declare_parameter<double>("h_fov_deg", 87.0);
    double vdeg = declare_parameter<double>("v_fov_deg", 58.0);

    nav_timeout_sec_ = declare_parameter<double>("nav_timeout_sec", 180.0);

    // pubs/subs
    sub_stamp_ = create_subscription<std_msgs::msg::UInt64>(
      "/map_stamp", rclcpp::QoS(1),
      std::bind(&FrontierViewpointNode::onMapUpdate, this, std::placeholders::_1));

    pub_frontiers_ = create_publisher<visualization_msgs::msg::Marker>("/frontiers", 1);
    pub_candidates_ = create_publisher<visualization_msgs::msg::MarkerArray>("/candidates", 1);
    pub_goal_marker_ = create_publisher<visualization_msgs::msg::Marker>("/goal_marker", 1);

    nav_client_ = rclcpp_action::create_client<NavigateToPose>(this, "navigate_to_pose");

    // 레이 테이블(카메라 optical 기준 Z-forward)
    buildRayTable(hdeg * M_PI/180.0, vdeg * M_PI/180.0);

    RCLCPP_INFO(get_logger(), "FrontierViewpointNode ready.");
  }

private:
  inline MapStore& store() { return global_map_store(); }

  // params/state
  std::string map_frame_;
  int yaw_samples_, keep_top_k_;
  double ring_r_min_, ring_r_max_, ring_r_step_;
  int width_, height_;
  double nav_timeout_sec_;

  // IO
  rclcpp::Subscription<std_msgs::msg::UInt64>::SharedPtr sub_stamp_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr pub_frontiers_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_candidates_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr pub_goal_marker_;
  rclcpp_action::Client<NavigateToPose>::SharedPtr nav_client_;

  // ray table
  std::vector<point3d> rays_;

  void buildRayTable(double hfov, double vfov) {
    rays_.clear(); rays_.reserve(static_cast<size_t>(width_)*height_);
    const double inc_h = hfov / static_cast<double>(width_);
    const double inc_v = vfov / static_cast<double>(height_);
    const int w2 = width_/2, h2 = height_/2;
    for (int v = -h2; v < h2; ++v) {
      for (int u = -w2; u < w2; ++u) {
        const double yaw   = (static_cast<double>(u)+0.5)*inc_h;
        const double pitch = (static_cast<double>(v)+0.5)*inc_v;
        point3d d(0,0,1); d.rotate_IP(pitch, 0.0, yaw); d.normalize();
        rays_.push_back(d);
      }
    }
  }

  void onMapUpdate(const std_msgs::msg::UInt64::SharedPtr) {
    // 스냅샷 포인터 취득 (짧게 read lock)
    std::shared_ptr<octomap::OcTree> tree;
    point3d sensor;
    {
    auto& ms = explore::global_map_store();                     // 확실하게 non-const
    std::shared_lock<std::shared_mutex> lk(ms.mtx);            // 타입을 명시해도 좋음
    tree   = ms.snapshot;
    sensor = ms.snap_origin;
    }
    if (!tree) return;

    // 프런티어 추출
    auto fronts = extractFrontiers(*tree, sensor.z(), tree->getResolution());
    publishFrontiers(fronts, *tree);

    if (fronts.empty()) {
      RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 5000, "No frontiers.");
      return;
    }

    // 후보 샘플링
    auto cands = sampleCandidates(*tree, fronts, sensor);
    if (cands.size() < 3) {
      RCLCPP_WARN(get_logger(), "Too few candidates.");
      return;
    }
    std::shuffle(cands.begin(), cands.end(), std::mt19937{std::random_device{}()});
    if ((int)cands.size() > keep_top_k_) cands.resize(keep_top_k_);

    // MI 근사(KeyRay unknown count)
    std::vector<double> mi(cands.size(), 0.0);
    for (size_t i=0; i<cands.size(); ++i) {
      std::vector<point3d> r; r.reserve(rays_.size());
      for (const auto &d : rays_) { point3d rr = d; rr.rotate_IP(0,0,cands[i].second.z()); r.push_back(rr); }
      mi[i] = estimateMI_KeyRay(cands[i].first, r, *tree, store().max_range);
    }

    publishCandidates(cands, mi);

    // 최대 MI goal 선택
    const auto it = std::max_element(mi.begin(), mi.end());
    if (it == mi.end()) return;
    const size_t idx = static_cast<size_t>(std::distance(mi.begin(), it));
    const auto best = cands[idx];

    publishGoal(best);
    (void) sendNavGoal(best.first, best.second.z()); // 실패시 다음 이벤트에서 재시도
  }

  /* ======== 알고리즘 유틸 ======== */
  std::vector<std::vector<point3d>>
  extractFrontiers(const octomap::OcTree &tree, double z0, double thickness) {
    std::vector<std::vector<point3d>> groups;
    const double res = tree.getResolution();
    const double R1 = 0.4;
    for (auto it = tree.begin_leafs(tree.getTreeDepth()); it != tree.end_leafs(); ++it) {
      if (tree.isNodeOccupied(*it)) continue;
      const double z = it.getZ();
      if (z < z0 || z > z0 + thickness) continue;
      const double x = it.getX(), y = it.getY();

      bool frontier = false;
      for (double dx=-res; dx<=res && !frontier; dx+=res) {
        for (double dy=-res; dy<=res && !frontier; dy+=res) {
          if (!tree.search(point3d(x+dx, y+dy, z))) frontier = true;
        }
      }
      if (!frontier) continue;

      bool added = false;
      for (auto &g : groups) {
        if (std::hypot(g.front().x()-x, g.front().y()-y) < R1) {
          g.emplace_back(x,y,z); added=true; break;
        }
      }
      if (!added) groups.push_back({point3d(x,y,z)});
    }
    return groups;
  }

  std::vector<std::pair<point3d, point3d>>
  sampleCandidates(const octomap::OcTree &tree,
                   const std::vector<std::vector<point3d>> &frontiers,
                   const point3d &sensor_orig) {
    std::vector<std::pair<point3d, point3d>> cands;
    const double z = sensor_orig.z();
    const double R3 = 0.3;
    for (const auto &g : frontiers) {
      for (int k=0; k<yaw_samples_; ++k) {
        double yaw = 2.0*M_PI * static_cast<double>(k)/static_cast<double>(yaw_samples_);
        for (double R=ring_r_min_; R<=ring_r_max_+1e-6; R+=ring_r_step_) {
          const double x = g.front().x() - R*std::cos(yaw);
          const double y = g.front().y() - R*std::sin(yaw);
          auto n = tree.search(point3d(x,y,z));
          if (!n || tree.isNodeOccupied(n)) continue;
          if (std::hypot(x-sensor_orig.x(), y-sensor_orig.y()) < 0.25) continue;
          bool ok=true;
          for (const auto &g2 : frontiers) {
            for (const auto &p : g2) { if (std::hypot(x-p.x(), y-p.y()) < R3) { ok=false; break; } }
            if (!ok) break;
          }
          if (!ok) continue;
          cands.emplace_back(point3d(x,y,z), point3d(0,0,yaw));
        }
      }
    }
    return cands;
  }

  double estimateMI_KeyRay(const point3d &origin,
                           const std::vector<point3d> &rays,
                           const octomap::OcTree &tree,
                           double max_range) {
    octomap::KeyRay keyray;
    std::unordered_set<octomap::OcTreeKey, octomap::OcTreeKey::KeyHash> unknown;
    unknown.reserve(rays.size()*8);
    for (const auto &dir : rays) {
      point3d end;
      bool hit = tree.castRay(origin, dir, end, /*ignoreUnknown=*/false, max_range);
      if (tree.computeRayKeys(origin, end, keyray)) {
        for (const auto &k : keyray) if (!tree.search(k)) unknown.insert(k);
      }
      if (!hit) { // go to max range
        point3d far = origin + dir * max_range;
        if (tree.computeRayKeys(origin, far, keyray)) {
          for (const auto &k : keyray) if (!tree.search(k)) unknown.insert(k);
        }
      }
    }
    return static_cast<double>(unknown.size()) * std::pow(tree.getResolution(), 3);
  }

  /* ======== 시각화/네비 ======== */
  void publishFrontiers(const std::vector<std::vector<point3d>> &groups,
                        const octomap::OcTree &tree) {
    visualization_msgs::msg::Marker mk;
    mk.header.frame_id = map_frame_; mk.header.stamp = now();
    mk.ns = "frontiers"; mk.id = 0;
    mk.type = visualization_msgs::msg::Marker::CUBE_LIST;
    mk.action = visualization_msgs::msg::Marker::ADD;
    mk.scale.x = mk.scale.y = mk.scale.z = tree.getResolution();
    mk.color.r = 1.0; mk.color.g = 0.4; mk.color.b = 0.0; mk.color.a = 1.0;
    for (const auto &grp : groups)
      for (const auto &p : grp) {
        geometry_msgs::msg::Point pt; pt.x=p.x(); pt.y=p.y(); pt.z=p.z();
        mk.points.push_back(pt);
      }
    pub_frontiers_->publish(mk);
  }

  void publishCandidates(const std::vector<std::pair<point3d,point3d>> &cands,
                         const std::vector<double> &mi) {
    visualization_msgs::msg::MarkerArray arr;
    arr.markers.resize(cands.size());
    const double max_mi = std::max(1e-9, *std::max_element(mi.begin(), mi.end()));
    for (size_t i=0; i<cands.size(); ++i) {
      auto &mk = arr.markers[i];
      mk.header.frame_id = map_frame_; mk.header.stamp = now();
      mk.ns="candidates"; mk.id=static_cast<int>(i);
      mk.type = visualization_msgs::msg::Marker::ARROW;
      mk.action = visualization_msgs::msg::Marker::ADD;
      mk.pose.position.x = cands[i].first.x();
      mk.pose.position.y = cands[i].first.y();
      mk.pose.position.z = cands[i].first.z();
      tf2::Quaternion q; q.setRPY(0,0,cands[i].second.z()); q.normalize();
      mk.pose.orientation = tf2::toMsg(q);
      mk.scale.x = 0.5 + 1.5*(mi[i]/max_mi);
      mk.scale.y = mk.scale.z = 0.2;
      mk.color.r = 1.0; mk.color.g = 0.6; mk.color.b = 0.2;
      mk.color.a = std::min(1.0, 0.2 + mi[i]/max_mi);
    }
    pub_candidates_->publish(arr);
  }

  void publishGoal(const std::pair<point3d,point3d> &best) {
    visualization_msgs::msg::Marker goal;
    goal.header.frame_id = map_frame_; goal.header.stamp = now();
    goal.ns="goal"; goal.id=0; goal.type=visualization_msgs::msg::Marker::ARROW;
    goal.pose.position.x = best.first.x();
    goal.pose.position.y = best.first.y();
    goal.pose.position.z = best.first.z();
    tf2::Quaternion q; q.setRPY(0,0,best.second.z()); q.normalize();
    goal.pose.orientation = tf2::toMsg(q);
    goal.scale.x=1.0; goal.scale.y=goal.scale.z=0.3;
    goal.color.r=0.1; goal.color.g=0.9; goal.color.b=0.1; goal.color.a=1.0;
    pub_goal_marker_->publish(goal);
  }

  bool sendNavGoal(const point3d &p, double yaw) {
    if (!nav_client_->wait_for_action_server(std::chrono::seconds(2))) {
      RCLCPP_WARN(get_logger(), "Nav2 action server not available");
      return false;
    }
    NavigateToPose::Goal goal;
    goal.pose.header.frame_id = map_frame_;
    goal.pose.header.stamp = now();
    goal.pose.pose.position.x = p.x();
    goal.pose.pose.position.y = p.y();
    goal.pose.pose.position.z = 0.0;
    tf2::Quaternion q; q.setRPY(0,0,yaw); q.normalize();
    goal.pose.pose.orientation = tf2::toMsg(q);

    auto gh_future = nav_client_->async_send_goal(goal);
    if (gh_future.wait_for(std::chrono::seconds(1)) != std::future_status::ready) return false;
    auto gh = gh_future.get(); if (!gh) return false;

    auto res_fut = nav_client_->async_get_result(gh);
    if (res_fut.wait_for(std::chrono::duration<double>(nav_timeout_sec_)) != std::future_status::ready) {
      nav_client_->async_cancel_goal(gh);
      return false;
    }
    return res_fut.get().code == rclcpp_action::ResultCode::SUCCEEDED;
  }
};

} // namespace explore

RCLCPP_COMPONENTS_REGISTER_NODE(explore::FrontierViewpointNode)
