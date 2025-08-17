import os
import shutil
import random
glob = __import__('glob')

bottles_003_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'bottles_003'))
bottles_003_val_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'bottles_003_val'))
bottles_003_yolo_labels = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'bottles_003_yolo_labels'))
bottles_003_val_yolo_labels = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'bottles_003_val_yolo_labels'))
dataset_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'bottles_yolo_dataset'))

for sub in ['images/train', 'images/val', 'labels/train', 'labels/val']:
    os.makedirs(os.path.join(dataset_dir, sub), exist_ok=True)

def find_matching_files(folder, extension, pattern=None):

    if pattern:
        return glob.glob(os.path.join(folder, f"*{pattern}*.{extension}"))
    else:
        return glob.glob(os.path.join(folder, f"*.{extension}"))

def extract_base_name(filename):
    base = os.path.basename(filename)
    base = os.path.splitext(base)[0]
    if base.endswith('_rgb'):
        base = base[:-4] 
    return base

def prepare_dataset():
    print("scanning")
    
    
    bottles_003_rgb_files = find_matching_files(bottles_003_dir, "jpg", "_rgb")
    bottles_003_val_rgb_files = find_matching_files(bottles_003_val_dir, "jpg", "_rgb")
    
   
    bottles_003_labels = find_matching_files(bottles_003_yolo_labels, "txt")
    bottles_003_val_labels = find_matching_files(bottles_003_val_yolo_labels, "txt")
    
    bottles_003_rgb_dict = {extract_base_name(f): f for f in bottles_003_rgb_files}
    bottles_003_val_rgb_dict = {extract_base_name(f): f for f in bottles_003_val_rgb_files}
    bottles_003_labels_dict = {extract_base_name(f): f for f in bottles_003_labels}
    bottles_003_val_labels_dict = {extract_base_name(f): f for f in bottles_003_val_labels}
    
    all_val_pairs = []
    for base_name in bottles_003_val_rgb_dict:
        if base_name in bottles_003_val_labels_dict:
            all_val_pairs.append((
                bottles_003_val_rgb_dict[base_name],
                bottles_003_val_labels_dict[base_name]
            ))

    random.seed(42)
    test_size = int(len(all_val_pairs) * 0.1)
    test_pairs = random.sample(all_val_pairs, test_size)
    train_pairs = [p for p in all_val_pairs if p not in test_pairs]

    val_pairs = []
    for base_name in bottles_003_rgb_dict:
        if base_name in bottles_003_labels_dict:
            val_pairs.append((
                bottles_003_rgb_dict[base_name],
                bottles_003_labels_dict[base_name]
            ))

    if len(val_pairs) > 200:
        random.seed(42)  
        val_pairs = random.sample(val_pairs, 200)
    
    if len(train_pairs) < len(val_pairs):
        train_pairs, val_pairs = val_pairs, train_pairs

    for img_path, lbl_path in train_pairs:
        img_basename = os.path.basename(img_path)
        lbl_basename = os.path.basename(lbl_path)
        shutil.copy(img_path, os.path.join(dataset_dir, 'images/train', img_basename))
        shutil.copy(lbl_path, os.path.join(dataset_dir, 'labels/train', lbl_basename))
    
    for img_path, lbl_path in val_pairs:
        img_basename = os.path.basename(img_path)
        lbl_basename = os.path.basename(lbl_path)
        shutil.copy(img_path, os.path.join(dataset_dir, 'images/val', img_basename))
        shutil.copy(lbl_path, os.path.join(dataset_dir, 'labels/val', lbl_basename))

    for sub in ['images/test', 'labels/test']:
        os.makedirs(os.path.join(dataset_dir, sub), exist_ok=True)

    for img_path, lbl_path in test_pairs:
        img_basename = os.path.basename(img_path)
        lbl_basename = os.path.basename(lbl_path)
        shutil.copy(img_path, os.path.join(dataset_dir, 'images/test', img_basename))
        shutil.copy(lbl_path, os.path.join(dataset_dir, 'labels/test', lbl_basename))

    train_path = os.path.join(dataset_dir, 'images/train').replace('\\', '/')
    val_path = os.path.join(dataset_dir, 'images/val').replace('\\', '/')
    yaml_path = os.path.join(dataset_dir, 'data.yaml')
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write("""
train: {}
val: {}
nc: 1
names: ['bottle']
""".format(train_path, val_path))
    print(f"traing set/val_set: {len(train_pairs)/len(val_pairs) if len(val_pairs) else 0:.2f}")
    
    return len(train_pairs), len(val_pairs)

if __name__ == "__main__":
    train_count, val_count = prepare_dataset() 