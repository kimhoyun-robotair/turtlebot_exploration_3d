#ifndef JACKALEXPLORATION_EXPLORATION_HPP_
#define JACKALEXPLORATION_EXPLORATION_HPP_

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include <octomap/octomap.h>
#include <octomap_msgs/msg/octomap.hpp>
#include <octomap_msgs/conversions.h>

#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_eigen/tf2_eigen.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl_conversions/pcl_conversions.h>

#include <algorithm>
#include <numeric>
#include <cmath>
#include <string>
#include <vector>
#include <fstream>

namespace jackal_exploration {

using point3d   = octomap::point3d;
using PointType = pcl::PointXYZ;
using PointCloud= pcl::PointCloud<pcl::PointXYZ>;

constexpr double PI        = 3.1415926;
constexpr double octo_reso = 0.05;
constexpr int    num_of_samples_eva = 15;
constexpr int    num_of_bay = 3;

// ===== 글로벌 상태 (원본 유지) =====
extern std::unique_ptr<octomap::OcTree> cur_tree;
extern std::string         octomap_name_3d;
extern point3d             kinect_orig;
extern std::shared_ptr<tf2_ros::Buffer> g_tf_buffer;
extern rclcpp::Logger      g_logger;

// 원본 sort
inline std::vector<int> sort_MIs(const std::vector<double> &v){
  std::vector<int> idx(v.size());
  std::iota(idx.begin(), idx.end(),0);
  std::sort(idx.begin(), idx.end(),
            [&v](int i1, int i2){ return v[i1] > v[i2]; });
  return idx;
}

// 원본 sensor model
struct sensorModel {
  double width;
  double height;
  double horizontal_fov;
  double vertical_fov;
  double angle_inc_hor;
  double angle_inc_vel;
  double max_range;
  octomap::Pointcloud SensorRays;
  point3d InitialVector;

  sensorModel(double _width, double _height, double _horizontal_fov,
              double _vertical_fov, double _max_range)
  : width(_width), height(_height),
    horizontal_fov(_horizontal_fov), vertical_fov(_vertical_fov),
    max_range(_max_range)
  {
    angle_inc_hor = horizontal_fov / width;
    angle_inc_vel = vertical_fov / height;
    for(double j = -height/2; j < height/2; ++j)
      for(double i = -width/2; i < width/2; ++i) {
        InitialVector = point3d(1.0, 0.0, 0.0);
        InitialVector.rotate_IP(0.0, j*angle_inc_vel, i*angle_inc_hor);
        SensorRays.push_back(InitialVector);
      }
  }
};

// Kinect 모델 (원본과 동일)
static sensorModel Kinect_360(128, 96, 2*PI*57/360, 2*PI*43/360, 6.0);

// ===== 원본 함수 =====
inline double countFreeVolume(const std::unique_ptr<octomap::OcTree> &octree) {
  double volume = 0.0;
  for (auto it = octree->begin_leafs(octree->getTreeDepth());
       it != octree->end_leafs(); ++it)
    if (!octree->isNodeOccupied(*it))
      volume += std::pow(it.getSize(), 3);
  return volume;
}

inline octomap::Pointcloud castSensorRays(const std::unique_ptr<octomap::OcTree> &octree,
                                          const point3d &position,
                                          const point3d &sensor_orientation)
{
  octomap::Pointcloud hits;
  octomap::Pointcloud RaysToCast;
  RaysToCast.push_back(Kinect_360.SensorRays);
  RaysToCast.rotate(sensor_orientation.x(), sensor_orientation.y(), sensor_orientation.z());
  point3d end;
  for (size_t i=0; i<RaysToCast.size(); ++i) {
    if (octree->castRay(position, RaysToCast.getPoint(i), end, true, Kinect_360.max_range))
      hits.push_back(end);
    else {
      end = RaysToCast.getPoint(i) * Kinect_360.max_range;
      end += position;
      hits.push_back(end);
    }
  }
  return hits;
}

// 2D 프런티어 (z=0.4±reso 한 장)
inline std::vector<std::vector<point3d>>
extractFrontierPoints(const std::unique_ptr<octomap::OcTree> &octree) {
  std::vector<std::vector<point3d>> frontier_groups;
  std::vector<point3d> frontier_points;

  double R1 = 0.4;
  octomap::OcTreeNode *n_cur_frontier = nullptr;
  for (auto n = octree->begin_leafs(octree->getTreeDepth());
       n != octree->end_leafs(); ++n)
  {
    bool frontier_true = false;
    if (!octree->isNodeOccupied(*n)) {
      double x_cur = n.getX(), y_cur = n.getY(), z_cur = n.getZ();
      if (z_cur < 0.4) continue;
      if (z_cur > 0.4 + octo_reso) continue;

      for (double xb = x_cur - octo_reso; xb < x_cur + octo_reso; xb += octo_reso)
        for (double yb = y_cur - octo_reso; yb < y_cur + octo_reso; yb += octo_reso)
        {
          n_cur_frontier = octree->search(point3d(xb, yb, z_cur));
          if (!n_cur_frontier) { frontier_true = true; continue; }
        }

      if (frontier_true) {
        if (frontier_groups.empty()) {
          frontier_points.resize(1);
          frontier_points[0] = point3d(x_cur, y_cur, z_cur);
          frontier_groups.push_back(frontier_points);
          frontier_points.clear();
        } else {
          bool belong_old = false;
          for (size_t u=0; u<frontier_groups.size(); ++u) {
            double dist = std::hypot(frontier_groups[u][0].x()-x_cur,
                                     frontier_groups[u][0].y()-y_cur);
            if (dist < R1) { frontier_groups[u].push_back(point3d(x_cur,y_cur,z_cur)); belong_old=true; break; }
          }
          if (!belong_old) {
            frontier_points.resize(1);
            frontier_points[0] = point3d(x_cur, y_cur, z_cur);
            frontier_groups.push_back(frontier_points);
            frontier_points.clear();
          }
        }
      }
    }
  }
  return frontier_groups;
}

// 후보 뷰포인트 (원본과 동일)
inline std::vector<std::pair<point3d, point3d>>
extractCandidateViewPoints(const std::vector<std::vector<point3d>> &frontier_groups,
                           const point3d &sensor_orig, int n)
{
  double R2_min = 1.0, R2_max = 5.0, R3 = 0.3;
  std::vector<std::pair<point3d, point3d>> candidates;
  double z = sensor_orig.z();

  for (size_t u=0; u<frontier_groups.size(); ++u) {
    for (double yaw=0; yaw<2*PI; yaw += PI*2/n)
      for (double R2=R2_min; R2<=R2_max; R2+=0.5)
      {
        double x = frontier_groups[u][0].x() - R2 * std::cos(yaw);
        double y = frontier_groups[u][0].y() - R2 * std::sin(yaw);

        bool candidate_valid = true;
        auto n_cur_3d = cur_tree->search(point3d(x,y,z));
        if (!n_cur_3d) { candidate_valid = false; continue; }

        if (std::hypot(x - sensor_orig.x(), y - sensor_orig.y()) < 0.25) {
          candidate_valid = false; continue;
        } else {
          for (size_t a=0; a<frontier_groups.size(); ++a)
            for (size_t b=0; b<frontier_groups[a].size(); ++b) {
              double d = std::hypot(x - frontier_groups[a][b].x(),
                                    y - frontier_groups[a][b].y());
              if (d < R3) { candidate_valid = false; break; }
            }

          for (double xb = x-0.3; xb < x+0.3; xb += octo_reso)
            for (double yb = y-0.3; yb < y+0.3; yb += octo_reso)
              for (double zb = sensor_orig.z()-0.1; zb < sensor_orig.z()+0.3; zb += octo_reso)
              {
                auto m = cur_tree->search(point3d(xb,yb,zb));
                if (!m) continue;
                else if (cur_tree->isNodeOccupied(m)) { candidate_valid=false; }
              }
        }

        if (candidate_valid)
          candidates.emplace_back(point3d(x,y,z), point3d(0.0,0.0,yaw));
      }
  }
  return candidates;
}

// MI (원본: 복사+가상삽입 → 자유체적 차)
inline double calc_MI(const std::unique_ptr<octomap::OcTree> &octree,
                      const point3d &sensor_orig,
                      const octomap::Pointcloud &hits,
                      const double before)
{
  auto octree_copy = std::make_unique<octomap::OcTree>(*octree);
  octree_copy->insertPointCloud(hits, sensor_orig, Kinect_360.max_range, true, true);
  double after = countFreeVolume(octree_copy);
  return after - before;
}

// ROS2 콜백 (원본과 동일 로직)
inline void kinectCallbacks(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
{
  if (!g_tf_buffer) return;

  // ROS1: ros::Duration(0.07).sleep();
  rclcpp::sleep_for(std::chrono::milliseconds(70));

  // TF: msg->frame → map
  geometry_msgs::msg::TransformStamped tf;
  try {
    tf = g_tf_buffer->lookupTransform("map", msg->header.frame_id, tf2::TimePointZero);
  } catch (const tf2::TransformException &e) {
    RCLCPP_WARN(g_logger, "TF error: %s", e.what());
    return;
  }

  // PCL 변환
  pcl::PointCloud<pcl::PointXYZ> pcl_in, pcl_map;
  pcl::fromROSMsg(*msg, pcl_in);
  Eigen::Matrix4f T = tf2::transformToEigen(tf).matrix().cast<float>();
  pcl::transformPointCloud(pcl_in, pcl_map, T);

  // hits 구성(간단 전처리: NaN, z<-1 제거)
  octomap::Pointcloud hits;
  hits.reserve(pcl_map.size());
  for (const auto &p : pcl_map.points) {
    if (std::isnan(p.x) || p.z < -1.0) continue;
    hits.push_back(point3d(p.x, p.y, p.z));
  }

  // 삽입 (원본과 동일)
  cur_tree->insertPointCloud(hits, kinect_orig, Kinect_360.max_range);

  // 디스크 기록(원본 유지)
  cur_tree->write(octomap_name_3d);

  RCLCPP_INFO(g_logger, "Entropy(3d map): %.3f", countFreeVolume(cur_tree));
}

} // namespace jackal_exploration
#endif
