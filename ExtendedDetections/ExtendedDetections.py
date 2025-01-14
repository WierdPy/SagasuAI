import numpy as np
import torch
from supervision import Detections as BaseDetections
from supervision.config import CLASS_NAME_DATA_FIELD


class ExtendedDetections(BaseDetections):
    @classmethod
    def from_yolov9(cls, yolov9_results) -> 'ExtendedDetections':
        """
        Creates a Detections instance from YOLOv9 inference results.

        Args:
            yolov9_results (yolov9.models.common.Detections):
                The output Detections instance from YOLOv9.

        Returns:
            ExtendedDetections: A new Detections object that includes YOLOv9 detections.

        Example:
            results = model(image)
            detections = ExtendedDetections.from_yolov9(results)
        """
        xyxy, confidences, class_ids = [], [], []

        for det in yolov9_results.pred:
            for *xyxy_coords, conf, cls_id in reversed(det):
                xyxy.append(torch.stack(xyxy_coords).cpu().numpy())
                confidences.append(float(conf))
                class_ids.append(int(cls_id))

        class_names = np.array([yolov9_results.names[i] for i in class_ids])

        if not xyxy:
            return cls.empty()

        return cls(
            xyxy=np.vstack(xyxy),
            confidence=np.array(confidences),
            class_id=np.array(class_ids),
            data={CLASS_NAME_DATA_FIELD: class_names},
        )