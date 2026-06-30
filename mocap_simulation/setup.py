import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'mocap_simulation'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='smarseille',
    maintainer_email='s.r.marseille@student.tudelft.nl',
    description='Motion-capture based localization and launch setup for UWAIL.',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': ["vicon_to_tf_node = mocap_simulation.vicon_to_tf_node:main",
        ],
    },
)
