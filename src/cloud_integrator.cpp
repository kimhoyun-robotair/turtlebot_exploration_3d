#include "turtlebot_exploration_3d/map_store.hpp"

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/u_int64.hpp>
#include <octomap_msgs/msg/octomap.hpp>
#include <octomap_msgs/conversions.h>

#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_eigen/tf2_eigen.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl_conversions/pcl_conversions.h>

#include <Eigen/Core>
#include <chrono>

namespace explore {

class CloudIntegratorNode : public rclcpp::Node {
public:
  explicit CloudIntegratorNode(const rclcpp::NodeOptions &opts)
  : rclcpp::Node("cloud_integrator", opts),
    tf_buffer_(this->get_clock()),
    tf_listener_(tf_buffer_)
  {
    // params
    map_frame_   = declare_parameter<std::string>("map_frame", "map");
    cloud_topic_ = declare_parameter<std::string>("cloud_topic", "/mast_camera/points");
    r_min_ = declare_parameter<double>("r_min", 0.3);
    r_max_ = declare_parameter<double>("r_max", 6.0);
    z_min_ = declare_parameter<double>("z_min", -1.0);
    z_max_ = declare_parameter<double>("z_max", 2.0);
    store().resolution = declare_parameter<double>("octo_reso", 0.05);
    store().max_range  = declare_parameter<double>("max_range", 6.0);

    // init trees
    store().live     = std::make_shared<octomap::OcTree>(store().resolution);
    store().snapshot = std::make_shared<octomap::OcTree>(store().resolution);

    // pubs/subs
    stamp_pub_ = create_publisher<std_msgs::msg::UInt64>("/map_stamp", rclcpp::QoS(1));
    // RViz 보기용(희망 시): transient_local로 최근 맵 유지
    auto latched = rclcpp::QoS(rclcpp::KeepLast(1)).reliable();
    octomap_pub_ = create_publisher<octomap_msgs::msg::Octomap>("/octomap_3d", latched);

    sub_cloud_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      cloud_topic_, rclcpp::SensorDataQoS(),
      std::bind(&CloudIntegratorNode::cloudCb, this, std::placeholders::_1));

    // 1Hz 스냅샷 생성 + stamp publish
    snapshot_timer_ = create_wall_timer(std::chrono::milliseconds(1000),
      std::bind(&CloudIntegratorNode::makeSnapshotAndPublish, this));

    RCLCPP_INFO(get_logger(), "CloudIntegratorNode ready. topic=%s", cloud_topic_.c_str());
  }

private:
  // shared store alias
  inline MapStore& store() { return global_map_store(); }

  // params
  std::string map_frame_, cloud_topic_;
  double r_min_, r_max_, z_min_, z_max_;

  // io
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_cloud_;
  rclcpp::Publisher<std_msgs::msg::UInt64>::SharedPtr stamp_pub_;
  rclcpp::Publisher<octomap_msgs::msg::Octomap>::SharedPtr octomap_pub_;
  rclcpp::TimerBase::SharedPtr snapshot_timer_;

  // tf
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;

  void cloudCb(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    // TF 얻기
    geometry_msgs::msg::TransformStamped tf;
    auto stamp = rclcpp::Time(msg->header.stamp);
    try {
      // 최대 0.2s 기다림
      tf = tf_buffer_.lookupTransform(map_frame_, msg->header.frame_id,
                                      stamp, tf2::durationFromSec(0.2));
    } catch (const tf2::ExtrapolationException &e) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
        "TF extrapolation at %.3f s (%s). Falling back to latest TF.",
        stamp.seconds(), e.what());
      // 최신 TF로 폴백
      tf = tf_buffer_.lookupTransform(map_frame_, msg->header.frame_id,
                                      tf2::TimePointZero);
    }

    // PCL 변환
    pcl::PointCloud<pcl::PointXYZ> pcl_in, pcl_map;
    pcl::fromROSMsg(*msg, pcl_in);
    Eigen::Matrix4f T = tf2::transformToEigen(tf).matrix().cast<float>();
    pcl::transformPointCloud(pcl_in, pcl_map, T);

    // hit 구성
    octomap::Pointcloud hits;
    hits.reserve(pcl_map.size());
    for (const auto &p : pcl_map) {
      if (std::isnan(p.x) || std::isnan(p.y) || std::isnan(p.z)) continue;
      const double r2 = p.x*p.x + p.y*p.y + p.z*p.z;
      if (r2 < r_min_*r_min_ || r2 > r_max_*r_max_) continue;
      if (p.z < z_min_ || p.z > z_max_) continue;
      hits.push_back(octomap::point3d(p.x, p.y, p.z));
    }

    // live tree에 적재 (writer only)
    store().live_origin = octomap::point3d(T(0,3), T(1,3), T(2,3));
    store().live->insertPointCloud(hits, store().live_origin, store().max_range, true, true);
  }

  void makeSnapshotAndPublish() {
    // snapshot 교체는 짧게 unique_lock
    {
      std::unique_lock lk(store().mtx);
      store().snapshot = std::make_shared<octomap::OcTree>(*store().live); // deep copy
      store().snap_origin = store().live_origin;
      store().stamp.fetch_add(1, std::memory_order_relaxed);
    }
    // 이벤트 송신
    std_msgs::msg::UInt64 s; s.data = store().stamp.load(std::memory_order_relaxed);
    stamp_pub_->publish(s);

    // RViz용 OctoMap(저주기)
    octomap_msgs::msg::Octomap msg;
    msg.header.frame_id = map_frame_;
    msg.header.stamp = now();
    octomap_msgs::binaryMapToMsg(*store().snapshot, msg);
    octomap_pub_->publish(msg);
  }
};

} // namespace explore

RCLCPP_COMPONENTS_REGISTER_NODE(explore::CloudIntegratorNode)
