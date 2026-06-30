import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
import message_filters
import numpy as np
from cv_bridge import CvBridge
from scipy.spatial.transform import Rotation

from sensor_msgs.msg import Image, CameraInfo, PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from nav_msgs.msg import Odometry
from std_msgs.msg import Header
from vision_msgs.msg import Detection2DArray

from mirte_mapping_interfaces.msg import Observation3DArray
from tf2_ros import Buffer, TransformListener, TransformException

from .back_projection_core import PoseEstimationCore


class PoseEstimationNode(Node):
    def __init__(self):
        super().__init__("pose_estimation_node")

        self.declare_parameter("depth_topic", "/camera/depth/image_raw")
        self.declare_parameter("color_info_topic", "/camera/color/camera_info")
        self.declare_parameter("detections_topic", "/yolo/detections")
        self.declare_parameter("mask_topic", "/yolo/mask")
        self.declare_parameter("output_topic", "/yolo/detections_3d")
        self.declare_parameter("points_topic", "/yolo/object_points")
        self.declare_parameter("target_frame", "map")
        self.declare_parameter("max_detection_speed", 0.01)
        self.declare_parameter("max_detection_depth_m", 2.0)
        self.declare_parameter("max_object_z", 0.3)
        self.declare_parameter("odom_topic", "/mirte_base_controller/odom")
        self.declare_parameter("debug_perception", False)
        

        self.depth_topic = self.get_parameter("depth_topic").value
        self.color_info_topic = self.get_parameter("color_info_topic").value
        self.detections_topic = self.get_parameter("detections_topic").value
        self.mask_topic = self.get_parameter("mask_topic").value
        self.output_topic = self.get_parameter("output_topic").value
        self.points_topic = self.get_parameter("points_topic").value
        self.target_frame = self.get_parameter("target_frame").value
        self.max_detection_speed = float(self.get_parameter("max_detection_speed").value)
        self.stationary_delay_s = 0.7
        self.max_detection_depth_m = float(self.get_parameter("max_detection_depth_m").value)
        self.max_object_z = float(self.get_parameter("max_object_z").value)
        self.odom_topic = self.get_parameter("odom_topic").value
        self.debug_perception = self.get_parameter("debug_perception").value

        # CV bridge converts ROS images to numpy arrays
        self.bridge = CvBridge()
        self.intrinsics = None # camera intrinsics

        # lin and angular speed for checking if robot is moving
        self.current_lin_speed = 0.0
        self.current_ang_vel = 0.0
        self.stationary_since = None

        self.asymmetry_by_class = {"0": True,
                                   "1": False,
                                   "2": True,
                                   "3": False}
       
        # intstantiate "core" logic
        self.core = PoseEstimationCore(self.max_detection_depth_m, self.max_object_z, asymmetry_by_class=self.asymmetry_by_class, debug_perception=self.debug_perception)
        
        # tf listener and buffer
        self.tf_buffer = Buffer(node=self)
        self.tf_listener = TransformListener(self.tf_buffer, self)

        ## subscriptions
        self.info_sub = self.create_subscription(CameraInfo, self.color_info_topic, self.on_camera_info, 10) # camera intrinsics
        self.odom_sub = self.create_subscription(Odometry, self.odom_topic, self.on_odom, 10) # robot speed

        self.depth_sub = message_filters.Subscriber(self, Image, self.depth_topic, qos_profile=qos_profile_sensor_data) # depth images
        self.mask_sub = message_filters.Subscriber(self, Image, self.mask_topic, qos_profile=10)                        # yolo segmentation masks
        self.det_sub = message_filters.Subscriber(self, Detection2DArray, self.detections_topic, qos_profile=10)        # yolo detections

        # synchronizer: syncs depth, mask and detection messages
        self.sync = message_filters.ApproximateTimeSynchronizer([self.depth_sub, self.mask_sub, self.det_sub], queue_size=10, slop=0.25)
        self.sync.registerCallback(self.on_synced_data)
        
        ## publishers
        self.pub_3d = self.create_publisher(Observation3DArray, self.output_topic, 10) # 3d observations
        if self.debug_perception:
            self.pub_points = self.create_publisher(PointCloud2, self.points_topic, 10) # 3d points (selected by masks + outlier filter)

        self.get_logger().info(f"Pose estimation node started. Max detection depth: {self.max_detection_depth_m:.2f} m")

    def on_camera_info(self, msg):
        """Callback for camera info message"""
        if self.intrinsics is None:
            self.intrinsics = {"fx": msg.k[0],
                               "cx": msg.k[2],
                               "fy": msg.k[4],
                               "cy": msg.k[5]}

    def on_odom(self, msg):
        """
        Callback for odometry message
        - gets linear and angular velocity
        - updates current_lin_speed and current_ang_vel
        - for checking if robot is stationary before returning observations
        
        """
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        wz = msg.twist.twist.angular.z

        self.current_lin_speed = float(np.sqrt(vx * vx + vy * vy))
        self.current_ang_vel = float(wz)

        is_stationary = self.current_lin_speed <= self.max_detection_speed and abs(self.current_ang_vel) <= self.max_detection_speed
        now = self.get_clock().now()

        if is_stationary:
            if self.stationary_since is None:
                self.stationary_since = now
        else:
            self.stationary_since = None

    def has_been_stationary_long_enough(self):
        """Returns True if robot has been stationary for at least self.stationary_delay_s seconds"""
        if self.stationary_since is None:
            return False

        stationary_time = self.get_clock().now() - self.stationary_since
        return stationary_time.nanoseconds * 1e-9 >= self.stationary_delay_s

    def get_transform_matrix(self, depth_msg):
        """
        returns transform matrix (4x4) from camera to target (map or world) frame
        - also returns camera position and camera yaw for detection message
        """
        transform_msg = self.tf_buffer.lookup_transform(self.target_frame, depth_msg.header.frame_id, depth_msg.header.stamp)

        quat = [transform_msg.transform.rotation.x,
                transform_msg.transform.rotation.y,
                transform_msg.transform.rotation.z,
                transform_msg.transform.rotation.w]
        
        # Convert quaternion to rotation matrix
        rotation_mat = Rotation.from_quat(quat).as_matrix()
        # extract yaw
        sensor_yaw = Rotation.from_quat(quat).as_euler("xyz")[2]

        transform = np.eye(4)
        transform[:3, :3] = rotation_mat
        transform[0, 3] = transform_msg.transform.translation.x
        transform[1, 3] = transform_msg.transform.translation.y
        transform[2, 3] = transform_msg.transform.translation.z

        camera_pos = np.array([transform_msg.transform.translation.x,
                               transform_msg.transform.translation.y,
                               transform_msg.transform.translation.z])

        return transform, camera_pos, sensor_yaw

    def on_synced_data(self, depth_msg, mask_msg, det_msg):
        """
        Main callback for synchronized depth image + yolo detections & mask
        """
        ## 0. First check
        # if camera intrinsics are available
        if self.intrinsics is None:
            return

        # check if robot is not moving for 1.5 seconds
        if not self.has_been_stationary_long_enough():
            return


        ## 1. Get transform
        try:
            transform, camera_pos, sensor_yaw = self.get_transform_matrix(depth_msg)
        except TransformException:
            # if not available, skip frame
            return

        ## 2. convert depth and mask to numpy arrays
        depth_img = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="16UC1")
        mask_img = self.bridge.imgmsg_to_cv2(mask_msg, desired_encoding="32SC1")

        ## 3. call core to process, returns 3d observations (and points for debug)
        out_msg_array, all_object_points = self.core.process_frame(depth_img, 
                                                                   mask_img,
                                                                   det_msg.detections,
                                                                   self.intrinsics,
                                                                   transform,
                                                                   camera_pos,
                                                                   sensor_yaw,
                                                                   depth_msg.header)

        ## 4. publish
        out_msg_array.header.frame_id = self.target_frame # original stamp 
        self.pub_3d.publish(out_msg_array)

        if self.debug_perception:
            if all_object_points:
                header = Header()
                header.stamp = depth_msg.header.stamp
                header.frame_id = self.target_frame
                cloud_msg = pc2.create_cloud_xyz32(header, all_object_points)
                self.pub_points.publish(cloud_msg)


def main():
    rclpy.init()
    node = PoseEstimationNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

