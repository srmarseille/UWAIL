import rclpy

from geometry_msgs.msg import PoseStamped, TransformStamped, PoseWithCovarianceStamped
from rclpy.node import Node
from tf2_ros import TransformBroadcaster
from rclpy.qos import qos_profile_sensor_data
import numpy as np
import math
import json


def stamp_to_sec(stamp):
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9

def quat_from_yaw(yaw):
    half = 0.5 * yaw
    qz = math.sin(half)
    qw = math.cos(half)
    return 0.0, 0.0, qz, qw

def yaw_from_quat(qx, qy, qz, qw):
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def add_quats(ax, ay, az, aw, bx, by, bz, bw):
    qx = aw * bx + ax * bw + ay * bz - az * by
    qy = aw * by - ax * bz + ay * bw + az * bx
    qz = aw * bz + ax * by - ay * bx + az * bw
    qw = aw * bw - ax * bx - ay * by - az * bz

    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)

    if norm < 1e-9:
        return 0.0, 0.0, 0.0, 1.0

    return qx / norm, qy / norm, qz / norm, qw / norm

def pose_to_dict(x, y, z, qx, qy, qz, qw):
    return {"x": x,
            "y": y,
            "z": z,
            "yaw": yaw_from_quat(qx, qy, qz, qw)}



class ViconToTfNode(Node):
    def __init__(self):
        super().__init__("vicon_to_tf_node")

        self.declare_parameter("error_mode", "none")
        
        self.declare_parameter("max_error_x", 0.0)
        self.declare_parameter("max_error_y", 0.0)
        self.declare_parameter("max_error_yaw", 0.0)
        self.declare_parameter("dynamic_error_period", 5.0)
        self.declare_parameter("sigma_scale", 2.0) # = 95% certainty (gaussian)
        self.declare_parameter("random_seed", 0)
        
        self.declare_parameter("pose_with_cov_topic", "/synthetic_localization_pose_with_covariance")
        self.declare_parameter("robot_pose_log_path", "") 
        
        self.error_mode = self.get_parameter("error_mode").value
        self.max_error_x = self.get_parameter("max_error_x").value
        self.max_error_y = self.get_parameter("max_error_y").value
        self.max_error_yaw = self.get_parameter("max_error_yaw").value
        self.dynamic_error_period = max(float(self.get_parameter("dynamic_error_period").value), 1e-9)

        self.sigma_scale = max(self.get_parameter("sigma_scale").value, 1e-9) # avoid division by zero
        
        self.pose_with_cov_topic = self.get_parameter("pose_with_cov_topic").value

        # If path string not empty, will report robot GT and altered pose continuosly
        self.robot_pose_log_path = str(self.get_parameter("robot_pose_log_path").value)
        self.robot_pose_log_file = None
        if self.robot_pose_log_path:
            self.robot_pose_log_file = open(self.robot_pose_log_path, "w")

        # Init random for gaussian random error
        seed = self.get_parameter("random_seed").value
        self.random_generator = np.random.default_rng(seed)

        # tf 
        
        self.tf_broadcaster = TransformBroadcaster(self)

        # subs and pubs
        self.subscription = self.create_subscription(PoseStamped, "/vrpn_mocap/stan_mirte_1/pose", self.pose_callback, qos_profile_sensor_data)

        self.pose_with_cov_pub = self.create_publisher(PoseWithCovarianceStamped, self.pose_with_cov_topic, 10)

        # first stamp 
        self.first_stamp_sec = None

        # last stamp and current error (for gaussian timing)
        self.last_stamp_sec = None
        self.current_x_error = 0.0
        self.current_y_error = 0.0
        self.current_yaw_error = 0.0
    


    def pose_callback(self, msg):
        stamp_sec = stamp_to_sec(msg.header.stamp)

        gt_x, gt_y, gt_z = msg.pose.position.x, msg.pose.position.y, msg.pose.position.z
        gt_qx, gt_qy, gt_qz, gt_qw = msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w

        x, y, z = gt_x, gt_y, gt_z
        qx, qy, qz, qw = gt_qx, gt_qy, gt_qz, gt_qw

        if self.error_mode == "static":
            x += self.max_error_x
            y += self.max_error_y

            err_qx, err_qy, err_qz, err_qw = quat_from_yaw(self.max_error_yaw)

            qx, qy, qz, qw = add_quats(err_qx, err_qy, err_qz, err_qw, qx, qy, qz, qw)

        if self.error_mode == "sine":
            

            if self.first_stamp_sec is None:
                self.first_stamp_sec = stamp_sec

            elapsed = stamp_sec - self.first_stamp_sec
            phase = 2.0 * math.pi * elapsed / self.dynamic_error_period
            error_scale = 0.5 * math.sin(phase)

            x += error_scale * self.max_error_x
            y += error_scale * self.max_error_y

            err_qx, err_qy, err_qz, err_qw = quat_from_yaw(error_scale * self.max_error_yaw)
            qx, qy, qz, qw = add_quats(err_qx, err_qy, err_qz, err_qw, qx, qy, qz, qw)

        # add gaussian error based on max and sigma scale
        if self.error_mode == "gaussian":
            if self.last_stamp_sec is None or stamp_sec - self.last_stamp_sec >= self.dynamic_error_period:
                sigma_x = self.max_error_x / self.sigma_scale
                sigma_y = self.max_error_y / self.sigma_scale
                sigma_yaw = self.max_error_yaw / self.sigma_scale

                self.current_x_error = self.random_generator.normal(0.0, sigma_x)
                self.current_y_error = self.random_generator.normal(0.0, sigma_y)
                self.current_yaw_error = self.random_generator.normal(0.0, sigma_yaw)

                self.last_stamp_sec = stamp_sec

            x += self.current_x_error
            y += self.current_y_error

            err_qx, err_qy, err_qz, err_qw = quat_from_yaw(self.current_yaw_error)
            qx, qy, qz, qw = add_quats(err_qx, err_qy, err_qz, err_qw, qx, qy, qz, qw)

        self.publish_tf(msg.header.stamp, x, y, z, qx, qy, qz, qw)

        cov = self.make_pose_covariance()
        self.publish_pose_with_cov(msg.header.stamp, x, y, z, qx, qy, qz, qw, cov)

        self.write_pose_log(stamp_sec, gt_x, gt_y, gt_z, gt_qx, gt_qy, gt_qz, gt_qw, x, y, z, qx, qy, qz, qw, cov)

    def make_pose_covariance(self):
        cov = np.zeros(36)

        if self.error_mode == "static" or self.error_mode == "sine" or self.error_mode == "gaussian":
            var_x = abs(self.max_error_x / self.sigma_scale) ** 2
            var_y = abs(self.max_error_y / self.sigma_scale) ** 2
            var_yaw = abs(self.max_error_yaw / self.sigma_scale) ** 2

            cov[0] = var_x
            cov[7] = var_y
            cov[35] = var_yaw

        return cov

        
    def publish_tf(self, stamp, x, y, z, qx, qy, qz, qw):
        tf_msg = TransformStamped()
        tf_msg.header.stamp = stamp
        tf_msg.header.frame_id = "world"
        tf_msg.child_frame_id = "base_link"
        tf_msg.transform.translation.x = x
        tf_msg.transform.translation.y = y
        tf_msg.transform.translation.z = z        
        tf_msg.transform.rotation.x = qx
        tf_msg.transform.rotation.y = qy
        tf_msg.transform.rotation.z = qz
        tf_msg.transform.rotation.w = qw

        self.tf_broadcaster.sendTransform(tf_msg)


    def publish_pose_with_cov(self, stamp, x, y, z, qx, qy, qz, qw, cov):
        pose_with_cov_msg = PoseWithCovarianceStamped()
        pose_with_cov_msg.header.stamp = stamp
        pose_with_cov_msg.header.frame_id = "world"
        pose_with_cov_msg.pose.pose.position.x = x
        pose_with_cov_msg.pose.pose.position.y = y
        pose_with_cov_msg.pose.pose.position.z = z
        pose_with_cov_msg.pose.pose.orientation.x = qx
        pose_with_cov_msg.pose.pose.orientation.y = qy
        pose_with_cov_msg.pose.pose.orientation.z = qz
        pose_with_cov_msg.pose.pose.orientation.w = qw
        pose_with_cov_msg.pose.covariance = cov.tolist()

        self.pose_with_cov_pub.publish(pose_with_cov_msg)
        
    def write_pose_log(self, stamp_sec, gt_x, gt_y, gt_z, gt_qx, gt_qy, gt_qz, gt_qw, x, y, z, qx, qy, qz, qw, cov):
        if self.robot_pose_log_file is None:
            return

        pose_log = {"stamp": stamp_sec,
                 "gt": pose_to_dict(gt_x, gt_y, gt_z, gt_qx, gt_qy, gt_qz, gt_qw),
                 "synthetic_pose": pose_to_dict(x, y, z, qx, qy, qz, qw),
                 "synthetic_cov": {"x": cov[0], 
                                   "y": cov[7], 
                                   "z": cov[14], 
                                   "yaw": cov[35]}}

        self.robot_pose_log_file.write(json.dumps(pose_log) + "\n")
        self.robot_pose_log_file.flush()

    def destroy_node(self):
        if self.robot_pose_log_file is not None:
            self.robot_pose_log_file.close()
            self.robot_pose_log_file = None

        super().destroy_node()


def main():
    rclpy.init()
    node = ViconToTfNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()