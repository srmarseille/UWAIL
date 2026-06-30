#!/usr/bin/env python3

import os
import signal
import subprocess
import time


bag_paths = [# "/home/smarseille/Thesis/Experiments/bags/expr4_1103/delft1103_expr4_o12_d15_r1",
                # "/home/smarseille/Thesis/Experiments/bags/expr4_1103/delft1103_expr4_o12_d15_r2",
                # "/home/smarseille/Thesis/Experiments/bags/expr4_1103/delft1103_expr4_o12_d1_r1",
                # "/home/smarseille/Thesis/Experiments/bags/expr4_1103/delft1103_expr4_o12_d1_r2",
                # "/home/smarseille/Thesis/Experiments/bags/expr4_1103/delft1103_expr4_o12_d30_r1",
                # "/home/smarseille/Thesis/Experiments/bags/expr4_1103/delft1103_expr4_o12_d30_r2",
                #"/home/smarseille/Thesis/Experiments/bags/expr9_1803/delft1803_expr9_Oa_r1_tourpose",
                "/home/smarseille/Thesis/Experiments/bags/expr10_1803/delft1803_expr10_Oa_r3_2out2cycl",
                # "/home/smarseille/Thesis/Experiments/bags/expr10_1803/delft1803_expr10_Oa_r5_2out2cycl"
]

use_vicon_localization = True
synthetic_error = True

error_size = 0.35
error_period = 1.0
sigma_scale = 2.0
replay_speed = 1.5
start_delay_s = 6.0

bag_topics = ["/joint_states",
                "/robot_description",
                "/mirte_base_controller/odom",
                "/scan",
                "/io/imu/movement/data",
                "/camera/color/camera_info",
                "/camera/color/image_raw/compressed",
                "/camera/depth/camera_info",
                "/camera/depth/image_raw/compressedDepth",
                "/mirte_base_controller/cmd_vel",
                "/vrpn_mocap/stan_mirte_1/pose",
                "/vrpn_mocap/stan_asbtl_1/pose",
                "/vrpn_mocap/stan_asbtl_2/pose",
                "/vrpn_mocap/stan_book_1/pose",
                "/vrpn_mocap/stan_symbtlcr_1/pose",
                "/vrpn_mocap/stan_symbtlvi_2/pose"]

for bag_path in bag_paths:
    if not os.path.exists(bag_path):
        print(f"Bag path does not exist: {bag_path}")
        continue

    error_mode = "gaussian" if synthetic_error else "none"
    max_error = error_size if synthetic_error else 0.0

    launch_file = "uwail_pipeline_vicon.launch.py" if use_vicon_localization else "uwail_pipeline_amcl.launch.py"

    launch_cmd = ["ros2", "launch", "mocap_simulation", launch_file]

    if use_vicon_localization:
        launch_cmd += [f"error_mode:={error_mode}",
                    f"max_error:={max_error}",
                    f"dynamic_error_period:={error_period}",
                    f"sigma_scale:={sigma_scale}"]

    bag_cmd = ["ros2", "bag", "play", bag_path,
                "--clock",
                "--rate", str(replay_speed)]

    if use_vicon_localization:
        bag_cmd += ["--topics"] + bag_topics

    print("")
    print(f"Running bag: {bag_path}")

    launch_process = subprocess.Popen(launch_cmd, start_new_session=True)

    try:
        time.sleep(start_delay_s)
        subprocess.run(bag_cmd)
    finally:
        print("Stopping launch")
        os.killpg(os.getpgid(launch_process.pid), signal.SIGINT)
        launch_process.wait()

print("")
print("Finished all bags")