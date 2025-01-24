import sys
sys.path.append(r'C:\Users\burha\Codes\SagasuAI\ExtendedDetections')
from ExtendedDetections import ExtendedDetections
from api import notify_person_event
from datetime import datetime

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

from supervision import Detections as BaseDetections
from supervision.config import CLASS_NAME_DATA_FIELD

enter_counter = 0  # Counter for tracking middle-line crossings
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
            original_count = enter_counter
            enter_counter += 1
            notify_person_event("Enter", datetime.now(), original_count, enter_counter)
        # Check if the person crosses from left to right
        elif last_x < middle_x and current_x >= middle_x:
            original_count = enter_counter
            enter_counter -= 1
            notify_person_event("Exit", datetime.now(), original_count, enter_counter)
    # Update the last known position
    person_positions[tracker_id] = current_x

def setup_model_and_video_info(model, config):
    # Configure YOLOv9 model
    return prepare_yolov9(model, **config)

def prepare_yolov9(model, conf=0.2, iou=0.7, classes=None, agnostic_nms=False, max_det=1000):
    model.conf = conf
    model.iou = iou
    model.classes = classes
    model.agnostic = agnostic_nms
    model.max_det = max_det
    return model

def create_byte_tracker(frame_rate):
    # Setup BYTETracker with a given frame rate
    return sv.ByteTrack(frame_rate=frame_rate)

def setup_annotators():
    c = sv.ColorLookup.TRACK  # Colorize based on the TRACK id
    bounding_box_annotator = sv.BoundingBoxAnnotator(thickness=2, color_lookup=c)
    round_box_annotator = sv.RoundBoxAnnotator(thickness=2, color_lookup=c)
    corner_annotator = sv.BoxCornerAnnotator(thickness=2, color_lookup=c)
    trace_annotator = sv.TraceAnnotator(thickness=2, trace_length=50, color_lookup=c)
    label_annotator = sv.LabelAnnotator(text_scale=0.5, color_lookup=c)
    return [bounding_box_annotator, round_box_annotator, corner_annotator], trace_annotator, label_annotator

def annotate_frame(frame, detections, byte_tracker, trace_annotator,
                   annotators_list, label_annotator, show_labels, model, middle_x):
    # Apply tracking to detections
    detections = byte_tracker.update_with_detections(detections)
    annotated_frame = frame.copy()

    # Process tracked persons
    for tracker_id, (xmin, ymin, xmax, ymax) in zip(detections.tracker_id, detections.xyxy):
        if tracker_id is not None:  # Ensure valid tracker ID
            current_x = int((xmin + xmax) / 2)  # Calculate the horizontal midpoint of the bounding box
            check_crossing(tracker_id, current_x, middle_x)

    # Overlay the counter on the frame
    annotated_frame = display_counter(annotated_frame, enter_counter, frame.shape[1], frame.shape[0])

    # Annotate frame with traces
    annotated_frame = trace_annotator.annotate(scene=annotated_frame, detections=detections)

    # Annotate frame with bounding boxes
    annotated_frame = annotators_list[0].annotate(scene=annotated_frame, detections=detections)

    # Optionally, add labels to the annotations
    if show_labels:
        annotated_frame = add_labels_to_frame(label_annotator, annotated_frame, detections, model)

    return annotated_frame

def add_labels_to_frame(annotator, frame, detections, model):
    labels = [f"#{tracker_id} {model.model.names[class_id]} {confidence:0.2f}" for confidence, class_id, tracker_id in
              zip(detections.confidence, detections.class_id, detections.tracker_id)]
    return annotator.annotate(scene=frame, detections=detections, labels=labels)

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

def process_webcam(model, config=dict(conf=0.2, iou=0.6, classes=0), show_labels=True, webcam_index=0, save_video=False):
    model = setup_model_and_video_info(model, config)
    byte_tracker = create_byte_tracker(frame_rate=30)  # Approximate webcam frame rate
    annotators_list, trace_annotator, label_annotator = setup_annotators()

    # Access webcam
    cap = cv2.VideoCapture(webcam_index)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    _, frame_temp = cap.read()
    frame_height, frame_width = frame_temp.shape[:2]

    # Set up video writer if save_video is enabled
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
        out = cv2.VideoWriter('output_video.mp4', fourcc, 30, (frame_width, frame_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Draw vertical line
        frame, middle_x = draw_vertical_line(frame, frame_width)

        frame_rgb = frame[..., ::-1]  # Convert BGR to RGB
        results = model(frame_rgb, augment=False)
        detections = ExtendedDetections.from_yolov9(results)
        annotated_frame = annotate_frame(frame, detections, byte_tracker,
                                         trace_annotator, annotators_list,
                                         label_annotator, show_labels, model, middle_x)

        # Save the frame to the video file
        if save_video:
            out.write(annotated_frame)

        # Display the frame
        cv2.imshow('Sagasu', annotated_frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if save_video:
        out.release()
    cv2.destroyAllWindows()

# Model setup
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = DetectMultiBackend(weights='yolov9-c-converted.pt',
                            device=device, data='yolov9/data/coco.yaml', fuse=True)
model = AutoShape(model)

# Process webcam stream
process_webcam(
    model,
    config=dict(conf=0.2, iou=0.6, classes=0),
    show_labels=True,
    webcam_index=0,  # Default webcam index
    save_video=False  # Enable video saving
)
#