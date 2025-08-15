cmake_minimum_required(VERSION 3.8)
project(turtlebot_exploration_3d)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(eigen3_cmake_module REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(nav2_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(octomap REQUIRED)
find_package(octomap_msgs REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io)
find_package(pcl_conversions REQUIRED)
find_package(OpenMP)

include_directories(
  include
  ${Eigen3_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

add_executable(turtlebot_exploration_3d_node
  src/turtlebot_exploration_3d_node.cpp
  src/gpregressor.cpp
  src/covMaterniso3.cpp
  src/navigation_utils.cpp
)
target_include_directories(turtlebot_exploration_3d_node PUBLIC ${PCL_INCLUDE_DIRS})
target_link_libraries(turtlebot_exploration_3d_node
  ${PCL_LIBRARIES}
  octomap
)
if(OpenMP_CXX_FOUND)
  target_link_libraries(turtlebot_exploration_3d_node OpenMP::OpenMP_CXX)
endif()

ament_target_dependencies(turtlebot_exploration_3d_node
  rclcpp rclcpp_components rclcpp_action std_msgs sensor_msgs geometry_msgs
  visualization_msgs nav2_msgs tf2 tf2_ros tf2_geometry_msgs tf2_eigen
  octomap_msgs pcl_conversions
)
install(TARGETS turtlebot_exploration_3d_node
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/ DESTINATION include)
# install(DIRECTORY config DESTINATION share/${PROJECT_NAME})
# install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()