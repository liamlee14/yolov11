from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('D:/yolo/runs/train/bottle_yolov11/weights/best.pt')

    results = model.predict(
        source='D:/yolo/bottles_yolo_dataset/images/test',
        imgsz=640,
        conf=0.25,
        save=True,
        save_txt=True
    )
