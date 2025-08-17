import os
import glob

def remove_duplicate_labels(labels_dir):
    txt_files = glob.glob(os.path.join(labels_dir, '*.txt'))
    total_files = 0
    files_modified = 0
    for txt_file in txt_files:
        with open(txt_file, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        unique_lines = list(dict.fromkeys(lines)) 
        if len(unique_lines) < len(lines):
            with open(txt_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(unique_lines) + '\n')
            files_modified += 1
        total_files += 1
    print(f"checked {total_files} {files_modified} mal labels duplicated_labels removed。")

if __name__ == "__main__":
    remove_duplicate_labels(os.path.join('bottles_yolo_dataset', 'labels', 'train'))
    remove_duplicate_labels(os.path.join('bottles_yolo_dataset', 'labels', 'val'))