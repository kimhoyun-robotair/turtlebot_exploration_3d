#pragma once
#ifndef _NAVIGATION_UTILS_H_
#define _NAVIGATION_UTILS_H_

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>

#include <nav2_msgs/action/navigate_to_pose.hpp>
#include <octomap/octomap.h>
#include <tf2/LinearMath/Quaternion.h>

namespace navigation_utils {

using NavigateToPose = nav2_msgs::action::NavigateToPose;
using point3d = octomap::point3d;

/**
 * @brief 원본과 동일한 시그니처(노드 인자 추가 없이).
 *        내부에서 임시 ROS2 노드/액션클라이언트를 생성해 동기 블로킹 호출합니다.
 * @param go_posi  목표 위치 (map 좌표)
 * @param q        목표 오리엔테이션(tf2 Quaternion, Yaw 포함)
 * @return true    성공적으로 도달
 * @return false   실패(타임아웃/거절/취소/중단 등)
 */
bool goToDest(point3d go_posi, const tf2::Quaternion& q,
              std::chrono::seconds timeout = std::chrono::seconds(120));

} // namespace navigation_utils

#endif // _NAVIGATION_UTILS_H_
