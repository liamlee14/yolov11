import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('runs/train/bottle_yolov11/weights/best.pt')

    metrics = model.val(
        data='bottles_yolo_dataset/data.yaml',
        batch=16,
        imgsz=640,
        device=0,
        split='val'
    )
    print(f"Precision: {metrics.box.mp:.3f}")
    print(f"Recall: {metrics.box.mr:.3f}")
    print(f"mAP50: {metrics.box.map50:.3f}")
    print(f"mAP50-95: {metrics.box.map:.3f}")
    print(f"Speed (inference): {metrics.speed['inference']:.2f} ms/image")