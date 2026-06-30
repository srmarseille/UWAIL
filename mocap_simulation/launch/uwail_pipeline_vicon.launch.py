import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    mocap_share = get_package_share_directory("mocap_simulation")
    config_file = os.path.join(mocap_share, "config", "uwail_params.yaml")

    start_foxglove_bridge = LaunchConfiguration("start_foxglove_bridge")
    error_mode = LaunchConfiguration("error_mode")
    max_error = LaunchConfiguration("max_error")
    dynamic_error_period = LaunchConfiguration("dynamic_error_period")
    sigma_scale = LaunchConfiguration("sigma_scale")

    max_error_param = ParameterValue(max_error, value_type=float)
    dynamic_error_period_param = ParameterValue(dynamic_error_period, value_type=float)
    sigma_scale_param = ParameterValue(sigma_scale, value_type=float)

    static_tf_launch = IncludeLaunchDescription(PythonLaunchDescriptionSource(os.path.join(mocap_share, "launch", "static_sensor_tf.launch.py")))

    vicon_to_tf_node = Node(package="mocap_simulation",
                            executable="vicon_to_tf_node",
                            name="vicon_to_tf_node",
                            output="screen",
                            parameters=[config_file,
                                        {"error_mode": error_mode},
                                        {"max_error_x": max_error_param},
                                        {"max_error_y": max_error_param},
                                        {"max_error_yaw": 0.0},
                                        {"dynamic_error_period": dynamic_error_period_param},
                                        {"sigma_scale": sigma_scale_param}])
    
    depth_republish_node = Node(package="image_transport",
                            executable="republish",
                            name="depth_republish",
                            arguments=["compressedDepth", "raw"],
                            remappings=[
                                ("in/compressedDepth", "/camera/depth/image_raw/compressedDepth"),
                                ("out", "/camera/depth/image_raw"),],
                            parameters=[config_file],
                            output="screen")

    yolo_seg_node = Node(package="yolo_seg_ros2",
                        executable="yolo_seg_node",
                        name="yolo_seg_node",
                        output="screen",
                        parameters=[config_file])

    back_projection_node = Node(package="yolo_backprojection_ros2",
                                executable="back_projection_node",
                                name="back_projection_node",
                                output="screen",
                                parameters=[config_file])
    
    bbox_marker_node = Node(package="yolo_backprojection_ros2",
                            executable="bbox_marker_node",
                            name="bbox_marker_node",
                            output="screen",
                            parameters=[config_file])

    object_map_node = Node(package="object_map",
                            executable="object_map_node",
                            name="object_map_node",
                            output="screen",
                            parameters=[config_file])
    
    foxglove_bridge_node = Node(package="foxglove_bridge",
                                executable="foxglove_bridge",
                                name="foxglove_bridge",
                                output="log",
                                condition=IfCondition(start_foxglove_bridge),
                                arguments=["--ros-args", "--log-level", "error"],
                                parameters=[config_file])

    return LaunchDescription([DeclareLaunchArgument("start_foxglove_bridge", default_value="true"),
                                DeclareLaunchArgument("error_mode", default_value="gaussian"),
                                DeclareLaunchArgument("max_error", default_value="0.0"),
                                DeclareLaunchArgument("dynamic_error_period", default_value="5.0"),
                                DeclareLaunchArgument("sigma_scale", default_value="2.0"),
                                static_tf_launch,
                                vicon_to_tf_node,
                                depth_republish_node,
                                yolo_seg_node,
                                back_projection_node,
                                bbox_marker_node,
                                object_map_node,
                                foxglove_bridge_node])