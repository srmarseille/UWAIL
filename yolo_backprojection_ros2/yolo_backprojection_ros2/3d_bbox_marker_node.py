import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from builtin_interfaces.msg import Duration

from mirte_mapping_interfaces.msg import Observation3DArray


class BBoxVisualizerNode(Node):
    def __init__(self):
        super().__init__("bbox_visualizer_node")

        self.declare_parameter("detections_topic", "/yolo/detections_3d")
        self.declare_parameter("markers_topic", "/yolo/bbox_markers")
        self.declare_parameter("lifetime_sec", 4.0)
        self.declare_parameter("alpha", 0.5)

        self.detections_topic = self.get_parameter("detections_topic").value
        self.markers_topic = self.get_parameter("markers_topic").value
        self.lifetime_sec = self.get_parameter("lifetime_sec").value
        self.alpha = float(self.get_parameter("alpha").value)

        self.sub = self.create_subscription(Observation3DArray, self.detections_topic, self.on_detections, 10)
        self.pub = self.create_publisher(MarkerArray, self.markers_topic, 10)

        # Used to delete markers that were visible in the previous frame but not in the current one.
        self.last_marker_count = 0

        self.get_logger().info("bbox_visualizer_node started")

    def on_detections(self, msg):
        marker_array = MarkerArray()

        sec = int(self.lifetime_sec)
        nanosec = int((self.lifetime_sec - sec) * 1e9)
        marker_duration = Duration(sec=sec, nanosec=nanosec)

        current_count = len(msg.observations)

        for i, det in enumerate(msg.observations):
            # The cube pose is stored at the box center, while the observation position is the bottom center.
            bbox_marker = Marker()
            bbox_marker.header = msg.header
            bbox_marker.ns = "yolo_bboxes"
            bbox_marker.id = i * 3
            bbox_marker.type = Marker.CUBE
            bbox_marker.action = Marker.ADD

            bbox_marker.pose.position.x = det.object_pose.pose.position.x
            bbox_marker.pose.position.y = det.object_pose.pose.position.y
            bbox_marker.pose.position.z = det.object_pose.pose.position.z + (det.size_z / 2.0)
            bbox_marker.pose.orientation = det.object_pose.pose.orientation

            bbox_marker.scale.x = det.size_x
            bbox_marker.scale.y = det.size_y
            bbox_marker.scale.z = det.size_z

            bbox_marker.color.r = 0.0
            bbox_marker.color.g = 1.0
            bbox_marker.color.b = 0.0
            bbox_marker.color.a = self.alpha

            bbox_marker.lifetime = marker_duration
            marker_array.markers.append(bbox_marker)

            # Put a small class label above the object.
            text_marker = Marker()
            text_marker.header = msg.header
            text_marker.ns = "yolo_labels"
            text_marker.id = (i * 3) + 1
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD

            text_marker.pose.position.x = det.object_pose.pose.position.x
            text_marker.pose.position.y = det.object_pose.pose.position.y
            text_marker.pose.position.z = det.object_pose.pose.position.z + det.size_z + 0.1
            text_marker.pose.orientation.w = 1.0

            text_marker.scale.z = 0.15

            text_marker.color.r = 1.0
            text_marker.color.g = 1.0
            text_marker.color.b = 1.0
            text_marker.color.a = 1.0

            text_marker.lifetime = marker_duration

            label = det.class_id
            score = det.confidence
            text_marker.text = f"{label} ({score:.2f})"
            marker_array.markers.append(text_marker)

            # Show the local x-direction of the observation frame.
            arrow_marker = Marker()
            arrow_marker.header = msg.header
            arrow_marker.ns = "yolo_orientation"
            arrow_marker.id = (i * 3) + 2
            arrow_marker.type = Marker.ARROW
            arrow_marker.action = Marker.ADD

            arrow_marker.pose.position.x = det.object_pose.pose.position.x
            arrow_marker.pose.position.y = det.object_pose.pose.position.y
            arrow_marker.pose.position.z = det.object_pose.pose.position.z
            arrow_marker.pose.orientation = det.object_pose.pose.orientation

            arrow_marker.scale.x = max(0.2, det.size_x * 1.5) # shaft length
            arrow_marker.scale.y = 0.03 # shaft diameter
            arrow_marker.scale.z = 0.03 # head diameter

            arrow_marker.color.r = 1.0
            arrow_marker.color.g = 0.0
            arrow_marker.color.b = 0.0
            arrow_marker.color.a = 1.0

            arrow_marker.lifetime = marker_duration
            marker_array.markers.append(arrow_marker)

        # Remove old markers when the number of detections decreases.
        for i in range(current_count, self.last_marker_count):
            del_bbox = Marker()
            del_bbox.ns = "yolo_bboxes"
            del_bbox.id = i * 3
            del_bbox.action = Marker.DELETE
            marker_array.markers.append(del_bbox)

            del_text = Marker()
            del_text.ns = "yolo_labels"
            del_text.id = (i * 3) + 1
            del_text.action = Marker.DELETE
            marker_array.markers.append(del_text)

            del_arrow = Marker()
            del_arrow.ns = "yolo_orientation"
            del_arrow.id = (i * 3) + 2
            del_arrow.action = Marker.DELETE
            marker_array.markers.append(del_arrow)

        self.last_marker_count = current_count

        if marker_array.markers:
            self.pub.publish(marker_array)


def main():
    rclpy.init()
    node = BBoxVisualizerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()