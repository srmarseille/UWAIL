import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    mocap_share = get_package_share_directory("mocap_simulation")

    config_file = os.path.join(mocap_share, "config", "uwail_params.yaml")
    amcl_params_file = os.path.join(mocap_share, "config", "amcl_params.yaml")
    default_map_file = os.path.join(mocap_share, "maps", "delft1803_exp_map_3_full.yaml")

    use_sim_time = LaunchConfiguration("use_sim_time")
    map_file = LaunchConfiguration("map")
    start_foxglove_bridge = LaunchConfiguration("start_foxglove_bridge")
    publish_static_sensor_tf = LaunchConfiguration("publish_static_sensor_tf")

    static_sensor_tf_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(mocap_share, "launch", "static_sensor_tf.launch.py")),
        condition=IfCondition(publish_static_sensor_tf),
    )

    map_server_node = Node(package="nav2_map_server",
                            executable="map_server",
                            name="map_server",
                            output="screen",
                            parameters=[
                                amcl_params_file,
                                {"use_sim_time": use_sim_time},
                                {"yaml_filename": map_file}])

    amcl_node = Node(package="nav2_amcl",
                    executable="amcl",
                    name="amcl",
                    output="screen",
                    parameters=[
                        amcl_params_file,
                        {"use_sim_time": use_sim_time}])

    lifecycle_manager_node = Node(package="nav2_lifecycle_manager",
                                    executable="lifecycle_manager",
                                    name="lifecycle_manager_localization",
                                    output="screen",
                                    parameters=[
                                        amcl_params_file,
                                        {"use_sim_time": use_sim_time},
                                        {"autostart": True},
                                        {"node_names": ["map_server", "amcl"]}])

    depth_republish_node = Node(package="image_transport",
                                executable="republish",
                                name="depth_republish",
                                arguments=["compressedDepth", "raw"],
                                remappings=[
                                    ("in/compressedDepth", "/camera/depth/image_raw/compressedDepth"),
                                    ("out", "/camera/depth/image_raw")],
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
                                parameters=[
                                    config_file,
                                    {"target_frame": "map"},
                                    {"map_frame": "map"}])

    bbox_marker_node = Node(package="yolo_backprojection_ros2",
                            executable="bbox_marker_node",
                            name="bbox_marker_node",
                            output="screen",
                            parameters=[config_file])

    object_map_node = Node(package="object_map",
                            executable="object_map_node",
                            name="object_map_node",
                            output="screen",
                            parameters=[
                                config_file,
                                {"target_frame": "map"},
                                {"map_frame": "map"}])

    foxglove_bridge_node = Node(package="foxglove_bridge",
                                executable="foxglove_bridge",
                                name="foxglove_bridge",
                                output="log",
                                condition=IfCondition(start_foxglove_bridge),
                                arguments=["--ros-args", "--log-level", "error"],
                                parameters=[config_file])

    return LaunchDescription([
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument("start_foxglove_bridge", default_value="true"),
        DeclareLaunchArgument("publish_static_sensor_tf", default_value="false"),
        DeclareLaunchArgument("map", default_value=default_map_file, description="Path to the map yaml file"),

        static_sensor_tf_launch,

        map_server_node,
        amcl_node,
        lifecycle_manager_node,

        depth_republish_node,
        yolo_seg_node,
        back_projection_node,
        bbox_marker_node,
        object_map_node,
        foxglove_bridge_node,
    ])