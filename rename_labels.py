import os

def rename_labels(img_dir, label_dir):
    img_files = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
    img_bases = [os.path.splitext(f)[0] for f in img_files]

    for base in img_bases:
        if base.endswith('_rgb'):
            label_base = base  
            old_label_base = base.replace('_rgb', '')
            old_label = os.path.join(label_dir, old_label_base + '.txt')
            new_label = os.path.join(label_dir, label_base + '.txt')
            if os.path.exists(old_label):
                os.rename(old_label, new_label)
                print(f'Renamed: {old_label} -> {new_label}')

if __name__ == "__main__":
 
    train_img_dir = r'D:\YOLO\bottles_yolo_dataset\images\train'
    train_label_dir = r'D:\YOLO\bottles_yolo_dataset\labels\train'
    val_img_dir = r'D:\YOLO\bottles_yolo_dataset\images\val'
    val_label_dir = r'D:\YOLO\bottles_yolo_dataset\labels\val'

    print("Renaming train labels...")
    rename_labels(train_img_dir, train_label_dir)
    print("Renaming val labels...")
    rename_labels(val_img_dir, val_label_dir)
    print("Done.")
