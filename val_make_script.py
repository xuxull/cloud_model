import os
import shutil
import random

source_dir = "data/clouds_train"

train_dir = "data/train"
val_dir = "data/val"

split_ratio = 0.8

for class_name in os.listdir(source_dir):

    class_path = os.path.join(source_dir, class_name)

    if not os.path.isdir(class_path):
        continue

    images = os.listdir(class_path)

    random.shuffle(images)

    split_index = int(len(images) * split_ratio)

    train_images = images[:split_index]
    val_images = images[split_index:]

    os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)

    for image in train_images:
        shutil.copy(
            os.path.join(class_path, image),
            os.path.join(train_dir, class_name, image)
        )

    for image in val_images:
        shutil.copy(
            os.path.join(class_path, image),
            os.path.join(val_dir, class_name, image)
        )

print("Done!")