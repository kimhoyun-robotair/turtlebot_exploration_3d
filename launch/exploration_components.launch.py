from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')

    container = ComposableNodeContainer(
        name='exploration_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container_mt',  # 멀티스레드 컨테이너
        output='screen',
        emulate_tty=True,
        composable_node_descriptions=[
            ComposableNode(
                package='turtlebot_exploration_3d',
                plugin='explore::CloudIntegratorNode',
                name='cloud_integrator',
                parameters=[{'use_sim_time': use_sim_time},
                            {'octo_reso': 0.05, 'max_range': 6.0}],
                extra_arguments=[{'use_intra_process_comms': True}],
            ),
            ComposableNode(
                package='turtlebot_exploration_3d',
                plugin='explore::FrontierViewpointNode',
                name='frontier_viewpoint',
                parameters=[{'use_sim_time': use_sim_time}],
                extra_arguments=[{'use_intra_process_comms': True}],
            ),
        ],
    )

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        container
    ])
