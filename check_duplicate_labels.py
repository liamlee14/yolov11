import os
import glob
from collections import Counter

def check_duplicate_labels(labels_dir):
    txt_files = glob.glob(os.path.join(labels_dir, '*.txt'))
    total_files = 0
    files_with_duplicates = 0
    for txt_file in txt_files:
        with open(txt_file, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        counter = Counter(lines)
        duplicates = [item for item, count in counter.items() if count > 1]
        if duplicates:
            files_with_duplicates += 1
            for dup in duplicates:
                print(f"  repeat: {dup} (for {counter[dup]} times)")
        total_files += 1
    print(f"\nchecked {total_files} files， {files_with_duplicates} files repeated")

if __name__ == "__main__":
    check_duplicate_labels(os.path.join('bottles_yolo_dataset', 'labels', 'train'))
    check_duplicate_labels(os.path.join('bottles_yolo_dataset', 'labels', 'val')) 