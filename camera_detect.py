
import cv2
import torch
import numpy as np
import time
from pathlib import Path
import argparse
import sys

sys.path.append(str(Path(__file__).parent))

from ultralytics import YOLO

class BottleDetector:
    def __init__(self, model_path, conf_threshold=0.5, iou_threshold=0.45):

        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        try:
            self.model = YOLO(model_path)
        except Exception as e:
            sys.exit(1)
        
        self.class_names = ['bottle']
        
        self.colors = {
            'bottle': (0, 255, 0),  
        }
        
        self.frame_count = 0
        self.fps = 0
        self.last_time = time.time()
        
    def draw_detections(self, frame, results):
        annotated_frame = frame.copy()
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    
                    if cls < len(self.class_names):
                        class_name = self.class_names[cls]
                        color = self.colors.get(class_name, (0, 255, 0))
                        
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                        
                        label = f"{class_name}: {conf:.2f}"
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                        cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), 
                                    (x1 + label_size[0], y1), color, -1)
                        cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return annotated_frame
    
    def draw_stats(self, frame):
        self.frame_count += 1
        current_time = time.time()
        if current_time - self.last_time >= 1.0:
            self.fps = self.frame_count / (current_time - self.last_time)
            self.frame_count = 0
            self.last_time = current_time
        
        fps_text = f"FPS: {self.fps:.1f}"
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        instructions = [
            "'q' quit",
            "'s' save",
            "'r' reset"
        ]
        
        for i, instruction in enumerate(instructions):
            y_pos = frame.shape[0] - 80 + i * 25
            cv2.putText(frame, instruction, (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def run_camera(self, camera_id=0, save_dir="saved_frames"):
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            return
        
        print("success!")
        
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                results = self.model(frame, conf=self.conf_threshold, iou=self.iou_threshold, verbose=False)
                
                annotated_frame = self.draw_detections(frame, results)
                
                annotated_frame = self.draw_stats(annotated_frame)
                
                cv2.imshow('Bottle Detection', annotated_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("quit")
                    break
                elif key == ord('s'):
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = save_path / f"bottle_detection_{timestamp}_{frame_count:06d}.jpg"
                    cv2.imwrite(str(filename), annotated_frame)
                    print(f"saved as: {filename}")
                elif key == ord('r'):
                    self.frame_count = 0
                    self.last_time = time.time()
                
                frame_count += 1
                
        except KeyboardInterrupt:
            print("\ninterped by keyboard")
        finally:
            cap.release()
            cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="real-time bottle detection")
    parser.add_argument("--model", type=str, default="yolov10s.pt")
    parser.add_argument("--conf", type=float, default=0.7)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--save-dir", type=str, default="saved_frames")
    
    args = parser.parse_args()
    
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"error model not exit: {model_path}")
        return
    
    detector = BottleDetector(
        model_path=str(model_path),
        conf_threshold=args.conf,
        iou_threshold=args.iou
    )
    
    detector.run_camera(
        camera_id=args.camera,
        save_dir=args.save_dir
    )

if __name__ == "__main__":
    main() 