import os
import json
import glob
import numpy as np
import yaml

train_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'bottles_003'))
val_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'bottles_003_val'))
test_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '2022-01-10_flaschen_labeled_Markus'))
yaml_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '2022-01-10_flaschen_labeled_Gebauer', '1641821010560883045.yaml'))
output_train = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'bottles_003_yolo_labels'))
output_val = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'bottles_003_val_yolo_labels'))
output_test = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'bottles_003_test_yolo_labels'))
os.makedirs(output_train, exist_ok=True)
os.makedirs(output_val, exist_ok=True)
os.makedirs(output_test, exist_ok=True)

yml = yaml.safe_load(open(yaml_file, 'r', encoding='utf-8'))
K = np.array(yml['K']).reshape(3, 3)
img_width = yml['width']
img_height = yml['height']

def project_point(pt3d, K):
    pt3d = np.array(pt3d).reshape(3, 1)
    pt2d = K @ pt3d
    pt2d = pt2d[:2, 0] / pt2d[2, 0]
    return pt2d

def process_folder(folder, output_dir):
    json_files = glob.glob(os.path.join(folder, '*_gt.json'))
    total_instances = 0
    filtered_instances = 0
    for json_path in json_files:
        base = os.path.basename(json_path).replace('_gt.json', '')
        img_path = os.path.join(folder, base + '_rgb.jpg')
        if not os.path.exists(img_path):
            continue
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        txt_lines = []
        for inst in data.get('instances', []):
            total_instances += 1
            Toc = np.array(inst['Toc'])
            center3d = Toc[:3, 3]  
     
            x2d, y2d = project_point(center3d, K)
            x_center = x2d / img_width
            y_center = y2d / img_height
            r = inst['r']
            fx, fy = K[0,0], K[1,1]
            z = center3d[2] if center3d[2] != 0 else 1.0
            pixel_radius = (fx + fy) / 2 * r / z
            width = height = 2 * pixel_radius / img_width
            if 0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 <= width <= 1 and 0 <= height <= 1:
                txt_lines.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
            else:
                filtered_instances += 1
        txt_path = os.path.join(output_dir, base + '.txt')
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(txt_lines))
    print(f"processed: {folder} -> {output_dir}，total:{total_instances}，{filtered_instances}useless object")

process_folder(train_dir, output_train)
process_folder(val_dir, output_val)
process_folder(test_dir, output_test)
print("Complete！") 