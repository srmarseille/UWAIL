#!/usr/bin/env python3
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

from mirte_mapping_interfaces.msg import ObjectMap3D, TrackedObject3D, Observation3DArray
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PoseWithCovarianceStamped
from tf2_ros import Buffer, TransformListener, TransformException  # type: ignore[attr-defined]

import message_filters
from sensor_msgs.msg import LaserScan
from rclpy.qos import qos_profile_sensor_data

from .math_helpers import yaw_from_quat, quat_from_yaw, get_fov_from_k, transform_msg_to_matrix
from .tracked_object import Observation
from .object_map import ObjectMap
from .visibility import VisibilityEvaluator
from .visualization_helpers import make_shape_marker, make_position_gaussian_marker, make_yaw_gaussian_marker, make_extent_cov_marker, make_text_marker
from .map_state_logger import write_map_state_jsonl


class ObjectMappingNode(Node):
    def __init__(self):
        super().__init__("object_mapping_node")
        
        self.declare_parameter("map_frame", "map")

        self.declare_parameter("max_match_cost", 14.16)
        self.declare_parameter("occlusion_scan_margin", 0.05)
        self.declare_parameter("map_state_log_path", "")
        self.declare_parameter("localization_pose_cov_topic", "/synthetic_localization_pose_with_covariance")


        self.map_frame = self.get_parameter("map_frame").value
        self.max_match_cost = float(self.get_parameter("max_match_cost").value)
        self.occlusion_scan_margin = float(self.get_parameter("occlusion_scan_margin").value)

        # map state logging (if not "", will log map state to file)
        self.map_state_log_path = str(self.get_parameter("map_state_log_path").value)
        self.map_state_log_file = None

        if self.map_state_log_path:
            self.map_state_log_file = open(self.map_state_log_path, "w")
            self.get_logger().info(f"Logging map state to {self.map_state_log_path}")
        
        self.detections_topic = "/yolo/detections_3d"
        self.optical_frame = "camera_color_optical_frame" # Using cam color opti frame for now (also intrinsics)
        self.scan_frame = "laser"

        # debuggin bools
        self.debug = False
        self.debug_vis = True

        # visualization cov params
        self.visualize_with_markers = True
        self.visualize_covariance = True
        self.covariance_sigma_scale = 2.0

        # localization covariance published by localization/vicon node
        self.localization_pose_cov_topic = self.get_parameter("localization_pose_cov_topic").value
        self.latest_localization_pose_cov = np.zeros((4, 4), dtype=float)


        ### Camera Intrinsics ###
        self.min_visible_range = 0.3
        self.max_visible_range = 2.0

        # default K, can be overwritten with camera info topic callback
        self.k = np.array([
                        [546.291259765625, 0.0, 314.9955749511719],
                        [0.0, 546.291259765625, 245.8639373779297],
                        [0.0, 0.0, 1.0]]) 
        self.image_width = 640
        self.image_height = 480
        # FOV & halve angles from default (placeholders automatic cam info callback)
        self.horizontal_fov, self.vertical_fov = get_fov_from_k(self.k, self.image_width,self.image_height)


        self.prior_prob_existence = 0.4
        self.min_prob_existence = -1.0

        ## Instantiate empty object map class
        self.object_map = ObjectMap(self.max_match_cost, min_prob_existence=self.min_prob_existence)


        ### Pubs & Subs ###
        # Tf buffer and sub
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.localization_pose_sub = self.create_subscription(PoseWithCovarianceStamped, self.localization_pose_cov_topic, self.on_localization_pose, 10)
        


        ## 3D observations and /scan sub, with approx time sync, for visibility check
        self.detection_sub = message_filters.Subscriber(self,
                                                        Observation3DArray,
                                                        self.detections_topic)

        self.scan_sub = message_filters.Subscriber(self,
                                                    LaserScan,
                                                    "/scan",
                                                    qos_profile=qos_profile_sensor_data)

        # Detections (3d observations) and /scan (for negative evidence) synched callback
        self.sync = message_filters.ApproximateTimeSynchronizer([self.detection_sub, self.scan_sub],
                                                                queue_size=10,
                                                                slop=0.2)
        
        self.sync.registerCallback(self.on_detections_synced_scan)

        # map state pub
        self.map_pub = self.create_publisher(ObjectMap3D,
                                            "/object_map/state",
                                            10)
        
        if self.visualize_with_markers: # visualization published if bool true
            self.marker_pub = self.create_publisher(MarkerArray,
                                                    "/object_map/markers",
                                                    10)

        # logger (it reached here, for launch file start)
        self.get_logger().info("object_mapping_node started")

    def on_localization_pose(self, msg):
        """
        Callback for localization covariance
        - (Robot pose is received from TF tree)
        - message is [x, y, z, roll, pitch, yaw]
        - mapping node only uses [x, y, z, yaw]
        """
        msg_cov = np.array(msg.pose.covariance, dtype=float).reshape(6, 6)

        pose_cov = np.zeros((4, 4), dtype=float)
        pose_cov[0, 0] = msg_cov[0, 0]
        pose_cov[0, 1] = msg_cov[0, 1]
        pose_cov[1, 0] = msg_cov[1, 0]
        pose_cov[1, 1] = msg_cov[1, 1]
        pose_cov[2, 2] = msg_cov[2, 2]
        pose_cov[3, 3] = msg_cov[5, 5]

        self.latest_localization_pose_cov = pose_cov


    def get_object_pose_cov_from_sensor_cov(self, base_link_position, object_pose, sensor_pose_cov):
        """
        This function returns the object pose covariance that results from (amcl) localization covariance on sensor pose
        - object position relative to baselink (robot) is dist_x, dist_y
        - small angle approximation
        """
        dist_x = object_pose[0] - base_link_position[0]
        dist_y = object_pose[1] - base_link_position[1]

        # Jacobian
        # dx = -dist_y * d_yaw
        # dy = dist_x * d_yaw
        J = np.eye(4, dtype=float)
        J[0, 3] = -dist_y
        J[1, 3] = dist_x
    

        return J @ sensor_pose_cov @ J.T



    def on_detections_synced_scan(self, detections_msg, scan_msg):
        """
        Main mapping callback for synced /yolo/detections_3d and /scan
         - detections_msg is Observation3DArray (from BackProjection node)
         - scan_msg is LaserScan
         - then: creates visibility object; updates object map; logs the state (optional); publishes the map state
        """

        # 1. Convert detections to observations + get transform between map (world)->camera frames and map->scan frame
        observations = self.make_observations_list(detections_msg)
        T_map_camera = self.get_transform_matrix(self.optical_frame, self.map_frame, stamp=detections_msg.header.stamp)
        T_map_laser = self.get_transform_matrix(self.scan_frame, self.map_frame, stamp=scan_msg.header.stamp)

        if T_map_camera is None: # handle no transform
            return
        
        if T_map_laser is None: # handle no transform
            self.get_logger().warning(f"No transform from {self.scan_frame} to {self.map_frame}, skip occlusion check")
            scan_msg = None # Make scan_msg None, to skip occlusion check (assume not occluded)

        # Make one vis eval object per observations list
        visibility_evaluator = VisibilityEvaluator(T_map_camera, 
                                                   self.horizontal_fov, 
                                                   self.vertical_fov, 
                                                   self.min_visible_range, 
                                                   self.max_visible_range,
                                                   T_map_laser=T_map_laser,
                                                   scan_msg=scan_msg,
                                                   scan_distance_margin=self.occlusion_scan_margin,
                                                   debug_vis=self.debug_vis)
        
        # 2. Update object map
        localization_cov = self.latest_localization_pose_cov # get latest loc cov
        self.object_map.update_map(observations, 
                                   visibility_evaluator, 
                                   time_observed=detections_msg.header.stamp, 
                                   localization_cov=localization_cov,
                                    prior_prob_existence=self.prior_prob_existence)
        
        # log map state to json nl (optional)
        write_map_state_jsonl(self.map_state_log_file,
                            detections_msg,
                            observations,
                            self.object_map.tracks,
                            self.map_frame)

        self.publish_map(detections_msg.header.stamp)
        self.log_tracks()

    def get_transform_matrix(self, target_frame, source_frame, stamp):
        """
        Look up TF transform and convert to 4x4 matrix
        - Only used for visibility evaluation
        """
        # try except handles TF lookup errors 
        try: 
            tf_msg = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                rclpy.time.Time(),
                #rclpy.time.Time.from_msg(stamp),   # type: ignore[attr-defined]
                timeout=Duration(seconds=0.8))   # type: ignore[call-arg]
            return transform_msg_to_matrix(tf_msg.transform)
        except TransformException as ex:
            self.get_logger().warning(f"Could not get transform {target_frame} <- {source_frame}: {ex}")
            return None


    def make_observations_list(self, msg):
        """
        Convert all 3d observations into list of Observation objects (defined in object_map.py)
        """

        observations = []

        for det in msg.observations:
            obs = self.observation_msg_to_observation(det, msg.header, msg.header.stamp)
            if obs is not None: # skip if None, then no transform (obs_msg_to_observation returns None if no baselink_tf)
                observations.append(obs)

        return observations


    def observation_msg_to_observation(self, det, header, stamp):
        """
        Convert observation message from BackProjection node to Observation object
        - R_{cause}_{frame} is covariance matric
        - R_pos is covariance from measurement model + (amcl) localization covaraince (added)
        """

        # Oject pose and covariance from detection message
        # published in back projection as [x, y, z, yaw]
        object_pose_in_source_frame = self.pose_from_detection_msg(det)
        R_pose_measurement_in_source_frame = self.pose_covariance_from_msg(det)

        # Baselink position in target frame, to propagate localization covariance

        base_link_tf = self.lookup_transform(self.map_frame, "base_link", stamp)
        if base_link_tf is None:
            return None
        
        base_link_position = np.array([base_link_tf.transform.translation.x, 
                                       base_link_tf.transform.translation.y, 
                                       base_link_tf.transform.translation.z])
        
        pose_in_target_frame = object_pose_in_source_frame.copy()
        pose_cov_in_target_frame = R_pose_measurement_in_source_frame.copy()


            

        # propagate localization covariance to object
        R_loc_base_link = self.latest_localization_pose_cov
        R_pose_loc_object = self.get_object_pose_cov_from_sensor_cov(base_link_position=base_link_position, 
                                                                     object_pose=pose_in_target_frame, 
                                                                     sensor_pose_cov=R_loc_base_link)
        
        # final pose cov = cov from measurement model + cov from localization 
        R_pose_in_target_frame = R_pose_loc_object + pose_cov_in_target_frame

        if self.debug:
            self.get_logger().info(f"meas diag = {np.diag(pose_cov_in_target_frame)}")
            self.get_logger().info(f"loc base diag = {np.diag(R_loc_base_link)}")
            self.get_logger().info(f"loc object diag = {np.diag(R_pose_loc_object)}")
            self.get_logger().info(f"final diag = {np.diag(R_pose_in_target_frame)}")

        # extent is in object dimensions
        extent = np.array([det.size_x, det.size_y, det.size_z])
        extent_cov = self.extent_covariance_from_msg(det)

        # class and conf directly from observation
        class_id = det.class_id
        confidence = float(det.confidence)


        # build one observation
        return Observation(
            header=header,
            pose=pose_in_target_frame,
            pose_cov=R_pose_in_target_frame,
            extent=extent,
            extent_cov=extent_cov,
            class_id=class_id,
            confidence=confidence,
            asymmetric=bool(det.asymmetric),
            sensor_xyz=np.array([det.sensor_position.x, 
                                 det.sensor_position.y, 
                                 det.sensor_position.z]))



    def pose_from_detection_msg(self, det):
        """Extract pose from detection message"""

        position = np.array([det.object_pose.pose.position.x,
                            det.object_pose.pose.position.y,
                            det.object_pose.pose.position.z])

        q = det.object_pose.pose.orientation
        yaw = yaw_from_quat(q.x, q.y, q.z, q.w)

        return np.array([position[0],
                        position[1],
                        position[2],
                        yaw])
    



    def pose_covariance_from_msg(self, det):
        msg_cov = np.array(det.object_pose.covariance, dtype=float).reshape(6, 6)

        pose_cov = np.zeros((4, 4), dtype=float)
        pose_cov[0, 0] = msg_cov[0, 0]
        pose_cov[0, 1] = msg_cov[0, 1]
        pose_cov[0, 2] = msg_cov[0, 2]

        pose_cov[1, 0] = msg_cov[1, 0]
        pose_cov[1, 1] = msg_cov[1, 1]
        pose_cov[1, 2] = msg_cov[1, 2]

        pose_cov[2, 0] = msg_cov[2, 0]
        pose_cov[2, 1] = msg_cov[2, 1]
        pose_cov[2, 2] = msg_cov[2, 2]

        pose_cov[3, 3] = msg_cov[5, 5]
        return pose_cov


    def extent_covariance_from_msg(self, det):
        extent_cov = np.array(det.extent_covariance, dtype=float).reshape(3, 3)
        return extent_cov


    def lookup_transform(self, target_frame, source_frame, stamp):
        try:
            return self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                rclpy.time.Time.from_msg(stamp),   # type: ignore[attr-defined]
                timeout=Duration(seconds=0.1))   # type: ignore[attr-defined]
            
        except TransformException as e:
            if self.debug:
                self.get_logger().warn(f"TF lookup failed: {target_frame} <- {source_frame}: {str(e)}")
            return None

    def publish_map(self, stamp):
        map_msg = self.make_map_msg(stamp)
        self.map_pub.publish(map_msg)

        if self.visualize_with_markers:
            self.publish_markers(stamp)
    
    def make_map_msg(self, stamp):
        map_msg = ObjectMap3D()
        map_msg.header.frame_id = self.map_frame
        map_msg.header.stamp = stamp

        for track in self.object_map.tracks:
            object_msg = self.track_to_map_object_msg(track)
            map_msg.objects.append(object_msg) # type: ignore[attr-defined]

        return map_msg
    

    def track_to_map_object_msg(self, track):
        """
        Convert Track (internally defined) to TrackedObject3D message (defined in object_map.msg)
        """
        track_msg = TrackedObject3D()

        track_msg.track_id = int(track.track_id)

        track_msg.pose.position.x = float(track.pose[0])
        track_msg.pose.position.y = float(track.pose[1])
        track_msg.pose.position.z = float(track.pose[2])

        if track.asymmetric:
            qx, qy, qz, qw = quat_from_yaw(track.pose[3])
            track_msg.pose.orientation.x = qx
            track_msg.pose.orientation.y = qy
            track_msg.pose.orientation.z = qz
            track_msg.pose.orientation.w = qw
        else: # if symmetric, pubish unit quaternion
            track_msg.pose.orientation.w = 1.0
        # pose covariance is flattened list     
        track_msg.pose_covariance = track.pose_cov.flatten().tolist()

        track_msg.size_x = float(track.extent[0])
        track_msg.size_y = float(track.extent[1])
        track_msg.size_z = float(track.extent[2])
        # extent covariance is flattened list
        track_msg.extent_covariance = track.extent_cov.flatten().tolist()

        track_msg.class_id = str(track.class_id)
        track_msg.class_conf = float(track.class_conf)
        track_msg.prob_existence = float(track.prob_existence)
        track_msg.asymmetric = bool(track.asymmetric)

        # times for debuggin only, not used in logic
        track_msg.first_seen_time = track.first_seen_time
        track_msg.last_seen_time = track.last_seen_time
        track_msg.last_update_time = track.last_update_time

        return track_msg
    

    def publish_markers(self, stamp):
        """
        Publish the map as markers for visualization (e.g. Foxglove)
        - uses functions defined in visualization_helpers.py
        """
        marker_array = MarkerArray()

        # clear previous markers, 
        clear_marker = Marker()
        clear_marker.action = Marker.DELETEALL
        marker_array.markers.append(clear_marker)   # type: ignore[attr-defined]

        # counter for each made marker, preventing overwritign eachother
        marker_id = 0

        for track in self.object_map.tracks:

            # shape marker: cube or cylinder
            shape_marker = make_shape_marker(track, self.map_frame, stamp, marker_id)
            marker_array.markers.append(shape_marker)   # type: ignore[attr-defined]
            marker_id += 1

            if self.visualize_covariance:
                # gaussian position cov marker
                position_gaussian_marker = make_position_gaussian_marker(track, self.map_frame, stamp, marker_id)
                marker_array.markers.append(position_gaussian_marker)   # type: ignore[attr-defined]
                marker_id += 1

                # extent standard deviation marker
                extent_cov_marker = make_extent_cov_marker(track, self.map_frame, stamp, marker_id, sigma_scale=self.covariance_sigma_scale)
                marker_array.markers.append(extent_cov_marker)   # type: ignore[attr-defined]
                marker_id += 1

                # yaw gaussian cov marker
                yaw_gaussian_marker = make_yaw_gaussian_marker(track, self.map_frame, stamp, marker_id)

                if yaw_gaussian_marker is not None:
                    marker_array.markers.append(yaw_gaussian_marker)   # type: ignore[attr-defined]
                    marker_id += 1
            
            # text label
            text_marker = make_text_marker(track, self.map_frame, stamp, marker_id)
            marker_array.markers.append(text_marker)   # type: ignore[attr-defined]
            marker_id += 1

        self.marker_pub.publish(marker_array)



    def log_tracks(self):
        # if in debug, print track info in terminal 
        if not self.debug:
            return

        if len(self.object_map.tracks) == 0:
            self.get_logger().info("tracks: 0")
            return

        parts = []
        for track in self.object_map.tracks:
            parts.append(f"[id={track.track_id}, class={track.class_id}, p={track.prob_existence:.2f}, xyz=({track.pose[0]:.2f}, {track.pose[1]:.2f}, {track.pose[2]:.2f})]")

        self.get_logger().info(" ".join(parts))

    def destroy_node(self):
        # if logging map state to JSON, first close file before destroying node
        if self.map_state_log_file is not None:
            self.map_state_log_file.close()
            self.map_state_log_file = None

        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ObjectMappingNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()