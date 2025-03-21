import cv2
import os

dataset_path = r"C:\Users\DAVID ABUGA\Desktop\DATASETT"  # Change if needed

for person in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person)
    if not os.path.isdir(person_path):
        continue

    for img in os.listdir(person_path):
        img_path = os.path.join(person_path, img)

        image = cv2.imread(img_path)
        if image is None:
            print(f"Removing corrupted image: {img_path}")
            os.remove(img_path)

print("✅ Corrupt images removed successfully!")
