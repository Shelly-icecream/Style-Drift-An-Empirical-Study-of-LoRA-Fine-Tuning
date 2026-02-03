import cv2
import os

def center_crop_square(img):
    h, w = img.shape[:2]
    side = min(h, w)
    y = (h - side) // 2
    x = (w - side) // 2
    return img[y:y+side, x:x+side]

input_dir = "raw_images"
output_dir = "clean_images"
os.makedirs(output_dir, exist_ok=True)

for name in os.listdir(input_dir):
    if not name.lower().endswith((".jpg", ".png", ".jpeg", ".webp")):
        continue

    path = os.path.join(input_dir, name)
    img = cv2.imread(path)

    if img is None:
        continue

    img = center_crop_square(img)
    img = cv2.resize(img, (512, 512))

    out_name = os.path.splitext(name)[0] + ".png"
    cv2.imwrite(os.path.join(output_dir, out_name), img)

print("Done!")

