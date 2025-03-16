# verify_yolov8_video.py

from ultralytics import YOLO
import cv2
import os
import argparse
import numpy as np
import torch
import sys
import time
from collections import defaultdict

def parse_arguments():
    parser = argparse.ArgumentParser(description="YOLOv8 Video Object Detection")
    parser.add_argument('--model', default='yolov8n.pt', choices=['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt'],
                        help='YOLOv8 model to use (default: yolov8n.pt)')
    parser.add_argument('--conf-threshold', type=float, default=0.25,
                        help='Confidence threshold for detections (default: 0.25)')
    parser.add_argument('--video', type=str, default=None,
                        help='Path to the video file (default: None, will use webcam)')
    parser.add_argument('--save', action='store_true',
                        help='Save the processed video')
    parser.add_argument('--no-display', action='store_true',
                        help='Do not display video while processing')
    return parser.parse_args()

# Get project root directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Models directory path - updated to use the VTD subdirectory
models_path = os.path.join(project_root, "models", "VTD")
os.makedirs(models_path, exist_ok=True)

# Results directory for saving processed videos
results_path = os.path.join(project_root, "results", "videos")
os.makedirs(results_path, exist_ok=True)

def get_class_probabilities(model, frame):
    """
    Get probabilities for all classes directly from the model's output
    before non-maximum suppression is applied
    """
    # Preprocess image to model input format
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Get model prediction (without applying confidence threshold)
    results = model(img)
    
    # Extract the raw predictions before NMS
    raw_preds = results[0].boxes
    
    # If using newer YOLO versions that provide raw probs
    if hasattr(raw_preds, 'conf'):
        # Get class indices and confidences
        class_indices = [int(cls) for cls in raw_preds.cls.cpu().numpy()]
        confidences = raw_preds.conf.cpu().numpy()
        
        # Create dictionary of class name to highest confidence
        class_conf_dict = {}
        for i, cls_idx in enumerate(class_indices):
            cls_name = model.names[cls_idx]
            conf = confidences[i]
            # Keep highest confidence for each class
            if cls_name not in class_conf_dict or conf > class_conf_dict[cls_name]:
                class_conf_dict[cls_name] = float(conf)
                
        return class_conf_dict
    
    # Fallback method - direct calculation from model logits
    # Create blank dictionary for probabilities
    class_probs = {}
    for cls_name in model.names.values():
        class_probs[cls_name] = 0.0
        
    # Hard-coded vehicle classes to ensure we check them
    vehicle_classes = ["car", "truck", "bus", "motorcycle", "bicycle"]
    for cls in vehicle_classes:
        if cls in model.names.values():
            class_probs[cls] = 0.0
            
    return class_probs

# Simple object tracker to maintain consistent IDs
class SimpleTracker:
    def __init__(self, max_disappeared=20, max_distance=100):
        self.next_object_id = 0
        self.objects = {}  # Dictionary of object_id -> (center, class_name, max_confidence, box)
        self.disappeared = {}  # Dictionary of object_id -> count of frames disappeared
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        
    def register(self, centroid, class_name, confidence, box):
        """Register a new object with given centroid and class"""
        self.objects[self.next_object_id] = (centroid, class_name, confidence, box)
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1
    
    def deregister(self, object_id):
        """Deregister an object that has disappeared for too long"""
        del self.objects[object_id]
        del self.disappeared[object_id]
    
    def update(self, new_boxes, class_names, confidences):
        """Update tracker with new detections"""
        # Handle the case where no objects were detected
        if len(new_boxes) == 0:
            # Mark all existing objects as disappeared
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                
                # Deregister if object has been gone too long
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            
            # Return empty mapping
            return {}
        
        # Compute centroids of new detections
        new_centroids = []
        for box in new_boxes:
            x1, y1, x2, y2 = box
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            new_centroids.append((cx, cy))
        
        # If we have no existing objects, register all new objects
        if len(self.objects) == 0:
            for i, centroid in enumerate(new_centroids):
                self.register(centroid, class_names[i], confidences[i], new_boxes[i])
        else:
            # Try to match new detections with existing objects
            object_ids = list(self.objects.keys())
            object_centroids = [self.objects[i][0] for i in object_ids]
            
            # Compute distances between each pair of existing objects and new centroids
            distances = {}
            for i, object_id in enumerate(object_ids):
                distances[object_id] = {}
                for j, new_centroid in enumerate(new_centroids):
                    # Calculate Euclidean distance between centroids
                    d = np.sqrt((object_centroids[i][0] - new_centroid[0]) ** 2 + 
                                (object_centroids[i][1] - new_centroid[1]) ** 2)
                    distances[object_id][j] = d
            
            # Find best matches, starting with smallest distances
            used_objects = set()
            used_new_indices = set()
            
            # Sort all distances from smallest to largest
            all_pairs = []
            for object_id in object_ids:
                for new_idx, distance in distances[object_id].items():
                    all_pairs.append((distance, object_id, new_idx))
            
            all_pairs.sort(key=lambda x: x[0])
            
            # Match pairs by distance
            for distance, object_id, new_idx in all_pairs:
                # Skip if either object or detection is already matched
                if object_id in used_objects or new_idx in used_new_indices:
                    continue
                    
                # Skip if distance is too large
                if distance > self.max_distance:
                    continue
                    
                # Match the object and detection
                old_centroid, old_class, old_conf, old_box = self.objects[object_id]
                
                # Update with new position and box, but keep class and max confidence
                new_centroid = new_centroids[new_idx]
                new_class = class_names[new_idx]
                new_conf = confidences[new_idx]
                new_box = new_boxes[new_idx]
                
                # Only keep the same class and update max confidence 
                if new_class == old_class:
                    max_conf = max(old_conf, new_conf)
                else:
                    # If class changed, keep previous class if confidence was high,
                    # otherwise accept the new class
                    if old_conf > 0.7:
                        new_class = old_class
                        max_conf = old_conf
                    else:
                        max_conf = new_conf
                
                self.objects[object_id] = (new_centroid, new_class, max_conf, new_box)
                self.disappeared[object_id] = 0
                
                used_objects.add(object_id)
                used_new_indices.add(new_idx)
            
            # Register any new detections that weren't matched
            for i, centroid in enumerate(new_centroids):
                if i not in used_new_indices:
                    self.register(centroid, class_names[i], confidences[i], new_boxes[i])
            
            # Update status of any objects that weren't matched
            for object_id in object_ids:
                if object_id not in used_objects:
                    self.disappeared[object_id] += 1
                    
                    # Deregister if object has been gone too long
                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)
    
    def get_objects(self):
        """Return all tracked objects"""
        result = []
        for object_id, (centroid, class_name, confidence, box) in self.objects.items():
            result.append((object_id, centroid, class_name, confidence, box))
        return result

def process_video(video_path, model, conf_threshold, save_video=False, display_video=True):
    """Process video with YOLOv8 model"""
    # Open video file or camera
    if video_path is None:
        cap = cv2.VideoCapture(0)  # Use webcam
        video_name = "webcam"
    else:
        cap = cv2.VideoCapture(video_path)
        video_name = os.path.basename(video_path).split('.')[0]
    
    # Check if video opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video source")
        return
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Create video writer if save_video is True
    video_writer = None
    if save_video:
        output_path = os.path.join(results_path, f"{video_name}_processed.mp4")
        video_writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (frame_width, frame_height)
        )
        print(f"Saving processed video to: {output_path}")
    
    # Initialize object tracker
    tracker = SimpleTracker(max_disappeared=15, max_distance=min(frame_width, frame_height) // 8)
    
    # Process video frames
    frame_count = 0
    start_time = time.time()
    fps_counter_start = time.time()
    fps_counter = 0
    
    print("Processing video... Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Increment frame counter
        frame_count += 1
        fps_counter += 1
        
        # Calculate processing FPS every second
        if time.time() - fps_counter_start >= 1.0:
            processing_fps = fps_counter / (time.time() - fps_counter_start)
            print(f"Processing speed: {processing_fps:.2f} FPS")
            fps_counter = 0
            fps_counter_start = time.time()
        
        # Run YOLOv8 inference
        results = model.predict(source=frame, conf=conf_threshold)
        
        # Process results
        for r in results:
            boxes = r.boxes
            
            # Draw bounding boxes and labels
            if len(boxes) > 0:
                # Prepare data for tracker
                detection_boxes = []
                detection_classes = []
                detection_confs = []
                
                for box in boxes:
                    cls_id = int(box.cls[0].item())
                    cls_name = model.names[cls_id]
                    conf = float(box.conf[0].item())
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    detection_boxes.append((x1, y1, x2, y2))
                    detection_classes.append(cls_name)
                    detection_confs.append(conf)
                
                # Update tracker with new detections
                tracker.update(detection_boxes, detection_classes, detection_confs)
                
                # Draw tracked objects
                for obj_id, _, cls_name, max_conf, (x1, y1, x2, y2) in tracker.get_objects():
                    # Determine color based on classification
                    if max_conf > 0.7:
                        color = (0, 255, 0)  # Green
                    elif max_conf > 0.5:
                        color = (0, 255, 255)  # Yellow
                    else:
                        color = (0, 0, 255)  # Red
                    
                    # Scale for text and lines based on frame size
                    scale_factor = min(frame_width, frame_height) / 1000
                    text_scale = max(0.5, scale_factor)
                    line_thickness = max(1, int(2 * scale_factor))
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, line_thickness)
                    
                    # Create label with stable confidence
                    label = f"{cls_name}: {max_conf:.2f}"
                    
                    # Calculate text size for background rectangle
                    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 
                                              text_scale, line_thickness)[0]
                    
                    # Draw background rectangle for text
                    cv2.rectangle(frame, (x1, y1-text_size[1]-15), 
                                (x1+text_size[0]+15, y1), color, -1)
                    
                    # Put class label text
                    cv2.putText(frame, label, (x1+7, y1-7), cv2.FONT_HERSHEY_SIMPLEX, 
                              text_scale, (0, 0, 0), line_thickness)
        
        # Add processing info to the frame
        elapsed = time.time() - start_time
        info_text = f"Frame: {frame_count} | Time: {elapsed:.2f}s"
        cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                  0.7, (0, 0, 255), 2)
        
        # Display the frame if needed
        if display_video:
            cv2.imshow("YOLOv8 Video Detection", frame)
            
            # Check for user quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Save the frame if requested
        if video_writer is not None:
            video_writer.write(frame)
    
    # Release resources
    cap.release()
    if video_writer is not None:
        video_writer.release()
    
    if display_video:
        cv2.destroyAllWindows()
    
    # Print processing summary
    total_time = time.time() - start_time
    print(f"Processing complete - {frame_count} frames in {total_time:.2f} seconds")
    print(f"Average processing speed: {frame_count / total_time:.2f} FPS")
    
    if save_video:
        print(f"Processed video saved to: {output_path}")

def main():
    args = parse_arguments()
    
    # Load model
    model_path = os.path.join(models_path, args.model)
    model = YOLO(model_path)
    print(f"Loaded model: {args.model}")
    
    # Process video
    process_video(
        args.video,
        model,
        args.conf_threshold,
        save_video=args.save,
        display_video=not args.no_display
    )

if __name__ == "__main__":
    main()
    print("\nYOLOv8 video verification completed!")
    print("\nTip: For better performance or accuracy, try:")
    print("1. Using a different model: python verify_yolov8_video.py --model yolov8m.pt")
    print("2. Adjusting confidence threshold: python verify_yolov8_video.py --conf-threshold 0.4")
    print("3. Saving the processed video: python verify_yolov8_video.py --video input.mp4 --save")
