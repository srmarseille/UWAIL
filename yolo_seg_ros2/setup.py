from setuptools import setup

package_name = "yolo_seg_ros2"

setup(
    name=package_name,
    version="0.0.1",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="smarseille",
    maintainer_email="s.r.marseille@student.tudelft.nl",
    description="YOLO segmentation node publishing Detection2DArray + instance mask image",
    license="MIT",
    entry_points={
        "console_scripts": [
            "yolo_seg_node = yolo_seg_ros2.yolo_seg_node:main",
        ],
    },
)
