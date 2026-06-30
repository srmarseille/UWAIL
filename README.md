# UWAIL Object Mapping

This repository contains the ROS 2 implementation of the UWAIL object mapping pipeline.

The setup instructions below are intended for testing the pipeline out of the box using recorded ROS 2 bags. After the setup section, the README explains how to run the pipeline and which main settings can be changed.

## 1. Requirements

This project was developed and tested with:

- Ubuntu 22.04
- ROS 2 Humble
- Python 3.10
- `colcon`
- Recorded ROS 2 bags with RGB-D, LiDAR, odometry, TF, and motion-capture topics
- Python dependencies listed in `requirements.txt`

The instructions below assume that ROS 2 Humble is already installed.

## 2. Setup instructions

### 2.1 Preliminaries

Download the ROS 2 bags from Google Drive:

```text
[LINK]
```

For example:

```text
~/bags/
└── delft1803_expr10_Oa_r5_2out2cycl/
```


### 2.2 Create a ROS 2 workspace

```bash
mkdir -p ~/uwail_ws/src
cd ~/uwail_ws/src
git clone git@github.com:srmarseille/UWAIL.git
```

If SSH is not configured, use HTTPS instead.


### 2.3 Create a Python virtual environment (Recommended)

```bash
cd ~/uwail_ws
python3 -m venv uwail_venv
source uwail_venv/bin/activate
```


### 2.5 Install Python requirements

```bash
cd ~/uwail_ws
pip install -r src/UWAIL/requirements.txt
```

### 2.6 Install missing ROS dependencies

```bash
cd ~/uwail_ws
rosdep update
rosdep install --from-paths src --ignore-src -r -y
```

### 2.7 Build the workspace

```bash
cd ~/uwail_ws
colcon build --symlink-install
source install/setup.bash
```

For every new terminal, use:

```bash
source /opt/ros/humble/setup.bash
cd ~/uwail_ws
source uwail_venv/bin/activate
source install/setup.bash
```

## 3. Running the mocap simulation pipeline

The simulation script starts the required launch file and replays a selected ROS 2 bag.

Open:

```text
src/UWAIL/mocap_simulation.py
```

Set the bag path in `bag_paths`, for example:

```python
bag_paths = [
    "/home/.../bags/delft1803_expr10_Oa_r5_2out2cycl"
]
```

Then run:

```bash
cd ~/uwail_ws
source /opt/ros/humble/setup.bash
source uwail_venv/bin/activate
source install/setup.bash
python3 src/UWAIL/mocap_simulation.py
```

Multiple bags can be replayed by adding paths to `bag_paths`.

## 4. Main parameter settings

The most important settings are split over three files:

### 1. `mocap_simulation.py`

- `bag_paths`: list of ROS 2 bags to replay.
- `use_vicon_localization`: selects the localization pipeline. Use `True` for Vicon localization, or `False` for AMCL.
- `synthetic_error`: enables or disables synthetic localization error. Use `True` to add error to the Vicon pose, or `False` to use the Vicon pose directly.
- `error_size`: planar +- localization error offset in meters. For Gaussian error, this is used as the approximate two-sigma in both `x` and `y` separately.
- `error_period`: time in seconds between new Gaussian error samples. The sampled error is kept constant between samples.
- `sigma_scale`: converts `error_size` to standard deviation using `sigma = error_size / sigma_scale`.
- `replay_speed`: ROS bag replay speed. Use `1.0` for real time, higher values for faster replay.
- `start_delay_s`: delay between starting the lauch file, and the bag replay (allows launch file to correctly and fully start).
- `bag_topics`: topics replayed when `use_vicon_localization = True`. When `use_vicon_localization = False`, all topics in the bag are replayed. (Supresses `tf` and `tf_static`)

## AMCL localization

TODO


## Acknoledgement:
- OpenAI ChatGPT was used to help with the implementation of "supporting" parts of the system. Some examples are JSON state logger, visualization markers in `3d_bbox_marker_node.py` and `visualization_helpers.py`, setting up the ApproximateTimeSynchronizer, setting up launch files, some inline comments, and parts of the README.
- The core perception and mapping logic was not generated with ChatGPT. For example: RGB-D back-projection, depth and point filtering, object observation construction, track definition, data association, Kalman updates, log-odds existence update, and visibility/free-space reasoning.
- The design, implementation choices, validation, and final responsibility for the software remain my own.
- The use of ChatGPT was used to support implemenation; to speed up repetitive work. It was not used to design, replace, or validate the core perception and mapping logic.
- All outputs were verified by me, and I take full responsability.