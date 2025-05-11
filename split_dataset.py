import os
import shutil
import random
from pathlib import Path


source_dir = Path('archive/images')  
target_dir = Path('dataset')


for split in ['train', 'val']:
    (target_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
    (target_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)


all_images = list(source_dir.glob('*.jpg'))
random.shuffle(all_images)

# %80 eğitim, %20 validasyon
split_idx = int(len(all_images) * 0.8)
train_images = all_images[:split_idx]
val_images = all_images[split_idx:]

def move_files(images, split):
    for img_path in images:
        label_path = img_path.with_suffix('.txt')
        if not label_path.exists():
            continue 

        
        img_target = target_dir / 'images' / split / img_path.name
        label_target = target_dir / 'labels' / split / label_path.name

        shutil.copy(img_path, img_target)
        shutil.copy(label_path, label_target)

move_files(train_images, 'train')
move_files(val_images, 'val')
