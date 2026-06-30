from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    sim_time = {"use_sim_time": True}

    return LaunchDescription([
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="base_to_depth_optical_tf",
            arguments=[
                "--x", "0.147",
                "--y", "-0.003",
                "--z", "0.128",
                "--yaw", "-1.57079632679",
                "--pitch", "0.0",
                "--roll", "-1.57079632679",
                "--frame-id", "base_link",
                "--child-frame-id", "camera_depth_optical_frame",
            ],
            parameters=[sim_time],
            output="screen",
        ),

        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="base_to_color_optical_tf",
            arguments=[
                "--x", "0.144",
                "--y", "0.022",
                "--z", "0.128",
                "--yaw", "-1.569",
                "--pitch", "-0.04",
                "--roll", "-1.572",
                "--frame-id", "base_link",
                "--child-frame-id", "camera_color_optical_frame",
            ],
            parameters=[sim_time],
            output="screen",
        ),

        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="base_to_laser_tf",
            arguments=[
                "--x", "0.101",
                "--y", "0.0",
                "--z", "0.107",
                "--yaw", "1.551",
                "--pitch", "0.0",
                "--roll", "0.0",
                "--frame-id", "base_link",
                "--child-frame-id", "laser",
            ],
            parameters=[sim_time],
            output="screen",
        ),
    ])