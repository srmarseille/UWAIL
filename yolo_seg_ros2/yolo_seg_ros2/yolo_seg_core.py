import numpy as np
from cv_bridge import CvBridge

from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, BoundingBox2D
from vision_msgs.msg import ObjectHypothesisWithPose
import torch
from ultralytics import YOLO


class YoloSegCore:
    def __init__(self, model_path, device="cuda:0", debug=False, annotate_line_width=2, publish_annotated=True, conf=0.5):
        self.device = device
        self.model_path = model_path
        self.publish_annotated = publish_annotated
        self.annotate_line_width = annotate_line_width
        self.conf = conf
        self.debug = debug

        self.bridge = CvBridge()
        self.model = YOLO(self.model_path)

        if self.device.startswith("cuda") and not torch.cuda.is_available():
            print("CUDA not available. Using CPU device.")
            self.device = "cpu"

    def msg_to_bgr(self, image_msg):
        # Node receives compressed image, covert to bgr
        try:
            return self.bridge.compressed_imgmsg_to_cv2(image_msg, desired_encoding="bgr8")
        except Exception:
            return None

    def predict(self, image_bgr):
        # Run inference
        return self.model.predict(source=image_bgr,
                                  device=self.device,
                                  verbose=self.debug,
                                  conf=self.conf,
                                  agnostic_nms=True, # avoids overlapping masks from different classes same object
                                  retina_masks=True) # keeps masks at image resolution

    def results_to_messages(self, header, results):
        detections_array = Detection2DArray()
        detections_array.header = header

        # If yolo returns no results, return empty messages
        if not results or len(results) == 0:
            return detections_array, self.empty_mask(header), None

        result = results[0]
        height = int(result.orig_shape[0])
        width = int(result.orig_shape[1])

        boxes = getattr(result, "boxes", None)
        masks = getattr(result, "masks", None)

        # Pixel value 0 is background. Object masks use detection index + 1.
        instance_id_mask = np.zeros((height, width), dtype=np.int32)

        detection_count = 0
        if boxes is not None and boxes.xyxy is not None:
            detection_count = int(boxes.xyxy.shape[0])

        for index in range(detection_count):
            # YOLO stores on gpu, so move to cpu beforme building messages
            xyxy = boxes.xyxy[index].detach().cpu().numpy().astype(np.float32) # type: ignore[attr-defined]
            x1, y1, x2, y2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])

            class_id = -1
            score = 0.0
            if boxes.cls is not None: # type: ignore[attr-defined]
                class_id = int(boxes.cls[index].detach().cpu().item()) # type: ignore[attr-defined]
            if boxes.conf is not None: # type: ignore[attr-defined]
                score = float(boxes.conf[index].detach().cpu().item()) # type: ignore[attr-defined]

            # make 2D box message
            detection = Detection2D()
            detection.header = header

            bbox = BoundingBox2D()
            bbox.center.position.x = (x1 + x2) * 0.5
            bbox.center.position.y = (y1 + y2) * 0.5
            bbox.size_x = max(0.0, x2 - x1)
            bbox.size_y = max(0.0, y2 - y1)
            detection.bbox = bbox

            # class id and confidence are stored as detection hypothesis
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = str(class_id)
            hypothesis.hypothesis.score = score
            detection.results.append(hypothesis)

            detections_array.detections.append(detection)

            if masks is not None and masks.data is not None:
                mask = masks.data[index].detach().cpu().numpy().astype(np.uint8)
                instance_id_mask[mask > 0] = index + 1

        # Create mask message from segmented frame
        mask_msg = self.bridge.cv2_to_imgmsg(instance_id_mask, encoding="32SC1")
        mask_msg.header = header

        # Create annotated image, if enabled
        annotation_msg = None
        if self.publish_annotated:
            try:
                plotted = result.plot(line_width=self.annotate_line_width)
                annotation_msg = self.bridge.cv2_to_imgmsg(plotted, encoding="bgr8")
                annotation_msg.header = header
            except Exception:
                annotation_msg = None

        return detections_array, mask_msg, annotation_msg

    def empty_mask(self, header):
        # if no detections, make empty mask
        m = np.zeros((1, 1), dtype=np.int32)
        mask_msg = self.bridge.cv2_to_imgmsg(m, encoding="32SC1")
        mask_msg.header = header
        return mask_msg