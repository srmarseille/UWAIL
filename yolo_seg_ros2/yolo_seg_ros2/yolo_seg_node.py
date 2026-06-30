import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image, CompressedImage
from vision_msgs.msg import Detection2DArray

from .yolo_seg_core import YoloSegCore


class YoloSegNode(Node):
    def __init__(self):
        super().__init__("yolo_seg_node")

        self.declare_parameter("image_topic", "/camera/color/image_raw/compressed")
        self.declare_parameter("detections_topic", "/yolo/detections")
        self.declare_parameter("mask_topic", "/yolo/mask")
        self.declare_parameter("annotated_topic", "/yolo/annotated")
        self.declare_parameter("model_path", "/home/smarseille/Thesis/ros2_thesis_ws/best_0605.pt")

        self.declare_parameter("detection_confidence", 0.75)
        self.declare_parameter("publish_annotated", True) # if true, publish annotated image on yolo/annotated topic 
        self.declare_parameter("annotate_line_width", 2) # thickness of bbox line

        self.declare_parameter("debug", False)

        self.declare_parameter("device", "cuda:0")
        requested_device = self.get_parameter("device").value

        self.image_topic = self.get_parameter("image_topic").value
        self.detections_topic = self.get_parameter("detections_topic").value
        self.mask_topic = self.get_parameter("mask_topic").value
        self.annotated_topic = self.get_parameter("annotated_topic").value

        # Keep the model path configurable so the repo is not tied to one machine.
        self.model_path = self.get_parameter("model_path").value
        self.conf = float(self.get_parameter("detection_confidence").value)
        self.publish_annotated = bool(self.get_parameter("publish_annotated").value)
        self.annotate_line_width = int(self.get_parameter("annotate_line_width").value)

        self.debug = bool(self.get_parameter("debug").value)

        # make Core class object
        self.core = YoloSegCore(model_path=self.model_path,
                                device=requested_device,
                                debug=self.debug,
                                annotate_line_width=self.annotate_line_width,
                                publish_annotated=self.publish_annotated,
                                conf=self.conf)

        # Store device (cuda or cpu)
        self.device = self.core.device

        # Create publishers
        self.detection_pub = self.create_publisher(Detection2DArray, self.detections_topic, 10)
        self.mask_pub = self.create_publisher(Image, self.mask_topic, 10)

        # Create publisher for annotated image if enabled
        self.annotation_pub = None
        if self.publish_annotated:
            self.annotation_pub = self.create_publisher(Image, self.annotated_topic, 10)

        # Subscribe to RGB images
        self.sub = self.create_subscription(CompressedImage, self.image_topic, self.on_image, qos_profile_sensor_data)


        # loggs for when starting with launch file
        self.get_logger().info("yolo_seg_node started")
        if requested_device.startswith("cuda") and self.device == "cpu":
            self.get_logger().warn("CUDA not available, using CPU")
        self.get_logger().info("model: " + self.model_path + " (" + self.device + ")")


    def on_image(self, image_msg):
        """callback for image messages"""
        # Convert compressed image to bgr
        image_bgr = self.core.msg_to_bgr(image_msg)
        if image_bgr is None:
            if self.debug:
                self.get_logger().warn("compressed cv_bridge conversion failed")
            return

        # Run inference in core (file)
        results = self.core.predict(image_bgr)

        # Convert results to messages
        detections_msg, mask_msg, annotation_msg = self.core.results_to_messages(image_msg.header, results)

        self.detection_pub.publish(detections_msg)
        self.mask_pub.publish(mask_msg)

        # Publish annotated image if enabled
        if self.annotation_pub is not None and annotation_msg is not None:
            self.annotation_pub.publish(annotation_msg)


def main(args=None):
    rclpy.init(args=args)
    node = YoloSegNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()