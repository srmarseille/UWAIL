from setuptools import find_packages, setup
from glob import glob
import os

package_name = "yolo_backprojection_ros2"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "params"), glob("params/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="smarseille",
    maintainer_email="s.r.marseille@student.tudelft.nl",
    description="Back-project YOLO instance masks with depth to 3D observations",
    license="TODO",
    entry_points={
        "console_scripts": [
            "back_projection_node = yolo_backprojection_ros2.back_projection_node:main",
            "bbox_marker_node = yolo_backprojection_ros2.3d_bbox_marker_node:main",
        ],
    },
)