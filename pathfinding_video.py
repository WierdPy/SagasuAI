import sys

sys.path.append(r'C:\Users\burha\Codes\SagasuAI\ExtendedDetections')
from ExtendedDetections import ExtendedDetections



if 'yolov9' not in sys.path:
    sys.path.append('yolov9')

# ML/DL
import numpy as np
import torch

# CV
import cv2
import supervision as sv

# YOLOv9
from models.common import DetectMultiBackend, AutoShape
from utils.general import set_logging

# Video Demonstration
from IPython.display import HTML
from base64 import b64encode

from supervision import Detections as BaseDetections
from supervision.config import CLASS_NAME_DATA_FIELD

enter_counter  = 0  # Counter for tracking middle-line crossings
person_positions = {}


def draw_vertical_line(frame, frame_width):
    """Draw a vertical line in the middle of the frame."""
    middle_x = frame_width // 2
    color = (0, 255, 0)  # Green line
    thickness = 2
    frame = cv2.line(frame, (middle_x, 0), (middle_x, frame.shape[0]), color, thickness)
    return frame, middle_x

def check_crossing(tracker_id, current_x, middle_x):
    """Check if a person with a specific ID crosses the vertical middle line."""
    global enter_counter, person_positions
    if tracker_id in person_positions:
        last_x = person_positions[tracker_id]
        # Check if the person crosses from right to left
        if last_x > middle_x and current_x <= middle_x:
            enter_counter += 1
            print(f"Person {tracker_id} entered! Total count: {enter_counter}")
        # Check if the person crosses from left to right
        elif last_x < middle_x and current_x >= middle_x:
            enter_counter -= 1
            print(f"Person {tracker_id} exited! Total count: {enter_counter}")
    # Update the last known position
    person_positions[tracker_id] = current_x

def setup_model_and_video_info(model, config, source_path):
    # Initialize and configure YOLOv9 model
    model = prepare_yolov9(model, **config)
    # Retrieve video information
    video_info = sv.VideoInfo.from_video_path(source_path)
    return model, video_info

def display_counter(frame, counter, frame_width, frame_height):
    """Overlay the enter/exit counter text in the middle of the screen."""
    text = f"Net Enter: {counter}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    color = (255, 255, 255)  # White text
    thickness = 2

    # Calculate text size and position it at the center of the frame
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = (frame_width - text_size[0]) // 2
    text_y = (frame_height - text_size[1]) // 2
    cv2.putText(frame, text, (text_x, text_y), font, font_scale, color, thickness)
    return frame


def prepare_yolov9(model, conf=0.2, iou=0.7, classes=None, agnostic_nms=False, max_det=1000):
    model.conf = conf
    model.iou = iou
    model.classes = classes
    model.agnostic = agnostic_nms
    model.max_det = max_det
    return model


def create_byte_tracker(video_info):
    # Setup BYTETracker with video information
    return sv.ByteTrack(track_thresh=0.25, track_buffer=250, match_thresh=0.95, frame_rate=video_info.fps)


def setup_annotators():
    c = sv.ColorLookup.TRACK  # Colorize based on the TRACK id, as opposed to INDEX or CLASS
    # Initialize various annotators for bounding boxes, traces, and labels
    bounding_box_annotator = sv.BoundingBoxAnnotator(thickness=2, color_lookup=c)
    round_box_annotator = sv.RoundBoxAnnotator(thickness=2, color_lookup=c)
    corner_annotator = sv.BoxCornerAnnotator(thickness=2, color_lookup=c)
    trace_annotator = sv.TraceAnnotator(thickness=2, trace_length=50, color_lookup=c)
    label_annotator = sv.LabelAnnotator(text_scale=0.5, color_lookup=c)
    return [bounding_box_annotator, round_box_annotator, corner_annotator], trace_annotator, label_annotator


def setup_counting_zone(counting_zone, video_info):
    # Configure counting zone based on provided parameters
    max_width = video_info.width - 1
    max_height = video_info.height - 1
    if counting_zone == 'whole_frame':
        polygon = np.array([[0, 0], [max_width, 0], [max_width, max_height], [0, max_height]])
    else:
        polygon = np.clip(counting_zone, a_min=[0, 0], a_max=[max_width, max_height])
    polygon_zone = sv.PolygonZone(polygon=polygon, frame_resolution_wh=(video_info.width, video_info.height),
                                  triggering_position=sv.Position.CENTER)
    polygon_zone_annotator = sv.PolygonZoneAnnotator(polygon_zone, sv.Color.ROBOFLOW,
                                                     thickness=4 * (2 if counting_zone == 'whole_frame' else 1),
                                                     text_thickness=2, text_scale=2)
    return polygon_zone, polygon_zone_annotator

def annotate_frame(frame, index, video_info, detections, byte_tracker, counting_zone, polygon_zone,
                   polygon_zone_annotator, trace_annotator, annotators_list, label_annotator, show_labels, model):
    """Annotate the frame with lines, counters, and detections."""
    # Apply tracking to detections
    detections = byte_tracker.update_with_detections(detections)
    annotated_frame = frame.copy()

    # Draw vertical line
    annotated_frame, middle_x = draw_vertical_line(annotated_frame, frame.shape[1])

    # Process tracked persons
    for tracker_id, (xmin, ymin, xmax, ymax) in zip(detections.tracker_id, detections.xyxy):
        if tracker_id is not None:  # Ensure valid tracker ID
            current_x = int((xmin + xmax) / 2)  # Calculate the horizontal midpoint of the bounding box
            check_crossing(tracker_id, current_x, middle_x)

    # Overlay the counter on the frame
    annotated_frame = display_counter(annotated_frame, enter_counter, frame.shape[1], frame.shape[0])

    # Annotate frame with traces
    annotated_frame = trace_annotator.annotate(scene=annotated_frame, detections=detections)

    # Annotate frame with various bounding boxes
    section_index = int(index / (video_info.total_frames / len(annotators_list)))
    annotated_frame = annotators_list[section_index].annotate(scene=annotated_frame, detections=detections)

    # Optionally, add labels to the annotations
    if show_labels:
        annotated_frame = add_labels_to_frame(label_annotator, annotated_frame, detections, model)

    return annotated_frame


def add_labels_to_frame(annotator, frame, detections, model):
    labels = [f"#{tracker_id} {model.model.names[class_id]} {confidence:0.2f}" for confidence, class_id, tracker_id in
              zip(detections.confidence, detections.class_id, detections.tracker_id)]
    return annotator.annotate(scene=frame, detections=detections, labels=labels)


def process_video(model, config=dict(conf=0.1, iou=0.45, classes=None, ), counting_zone=None, show_labels=False,
                  source_path='input.mp4', target_path='output.mp4'):
    model, video_info = setup_model_and_video_info(model, config, source_path)
    byte_tracker = create_byte_tracker(video_info)
    annotators_list, trace_annotator, label_annotator = setup_annotators()

    def callback(frame: np.ndarray, index: int) -> np.ndarray:
        frame_rgb = frame[..., ::-1]  # Convert BGR to RGB
        results = model(frame_rgb, size=(video_info.height, video_info.width), augment=False)
        detections = ExtendedDetections.from_yolov9(results)
        return annotate_frame(frame, index, video_info, detections, byte_tracker, counting_zone, None, None,
                              trace_annotator, annotators_list, label_annotator, show_labels, model)

    sv.process_video(source_path=source_path, target_path=target_path, callback=callback)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = DetectMultiBackend(weights='yolov9-c-converted.pt', device=device, data='data\coco.yaml', fuse=True)
model = AutoShape(model)

process_video(
    model,
    config=dict(conf=0.2, iou=0.6, classes=0),
    show_labels=True,
    source_path='Videos\Labor\l1.MOV',
    target_path='Output\l1X.mp4'
)
#