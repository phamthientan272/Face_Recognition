import os
from PIL import Image
import numpy as np
import pickle
import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "faces_gray_crop")

current_id = 0
label_ids = {}
y_labels = []
x_train = []
size = (100, 100)

for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root, file)
            label = os.path.basename(root).replace(" ", "-").lower()
            if label in label_ids:
                pass
            else:
                label_ids[label] = current_id
                current_id += 1
            id_ = label_ids[label]
            # print(label_ids)

            image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, size)
            image_array = np.array(image, "uint8")

            x_train.append(image_array)
            y_labels.append(id_)

data = (x_train, y_labels)
print(np.array(x_train).shape)
with open("data.pickle", "wb") as f:
    pickle.dump(data, f)

print(label_ids)

with open("labels.pickle", 'wb') as f:
    pickle.dump(label_ids, f)
