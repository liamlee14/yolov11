import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if __name__ == '__main__':
    from ultralytics import YOLO
 
    model = YOLO('yolo11m.pt')
  
    results = model.train(
        data='bottles_yolo_dataset/data.yaml', 
        epochs=60,                             
        imgsz=640,                          
        batch=12,                             
        device=0,                             
        lr0=0.001,                           
        lrf=0.01,                           
        cos_lr=True,                         
        project='runs/train',                
        name='bottle_yolov11',              
        save=True,                          
        exist_ok=True,                        
        verbose=True,                         
    )
    metrics = model.val(
        data='bottles_yolo_dataset/data.yaml',
        batch=16,
        imgsz=640,
        device=0,
        split='val'
    )
    model.predict(source='bottles_yolo_dataset/images/val', save=True, imgsz=640, conf=0.25)
    print("best model: runs/train/bottle_yolov11")
