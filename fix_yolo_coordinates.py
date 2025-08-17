import os
import glob

def normalize_coordinates(value):
    return max(0, min(0.999, float(value)))

def fix_label_file(label_path):
    modified = False
    with open(label_path, 'r') as f:
        lines = f.readlines()

    fixed_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 5:  # class_id x_center y_center width height
            class_id = parts[0]
            x_center = normalize_coordinates(parts[1])
            y_center = normalize_coordinates(parts[2])
            width = normalize_coordinates(parts[3])
            height = normalize_coordinates(parts[4])

            if float(parts[1]) > 1 or float(parts[2]) > 1 or float(parts[3]) > 1 or float(parts[4]) > 1:
                modified = True

            fixed_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        else:
            fixed_lines.append(line.strip())

    if modified:
        with open(label_path, 'w') as f:
            f.write('\n'.join(fixed_lines))
        return True
    return False

def fix_all_label_files(dataset_dir):
    train_labels_dir = os.path.join(dataset_dir, 'labels/train')
    val_labels_dir = os.path.join(dataset_dir, 'labels/val')
    
    train_files = glob.glob(os.path.join(train_labels_dir, '*.txt'))
    fixed_train_count = sum(fix_label_file(file) for file in train_files)
    
    val_files = glob.glob(os.path.join(val_labels_dir, '*.txt'))
    fixed_val_count = sum(fix_label_file(file) for file in val_files)
    
    
    train_cache = os.path.join(dataset_dir, 'labels/train.cache')
    val_cache = os.path.join(dataset_dir, 'labels/val.cache')
    
    if os.path.exists(train_cache):
        os.remove(train_cache)
        print(f"deleted: {train_cache}")
    
    if os.path.exists(val_cache):
        os.remove(val_cache)

if __name__ == "__main__":
    dataset_dir = 'bottles_yolo_dataset'
    fix_all_label_files(dataset_dir)