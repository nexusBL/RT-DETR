import cv2, os, glob, random, shutil

TRAIN_IMG = r'C:\Users\icroc\Documents\RT-DETR\MERGED\images\train'
TRAIN_LBL = r'C:\Users\icroc\Documents\RT-DETR\MERGED\labels\train'

# Only pick ORIGINAL images — skip scale/boost/gray/extra copies
images = [f for f in (glob.glob(TRAIN_IMG + '/*.jpg') + glob.glob(TRAIN_IMG + '/*.png'))
          if not any(tag in os.path.basename(f) 
                     for tag in ['_gray', 'scale', 'boost_', 'xtra_'])]

# Skip if grayscale already exists
to_process = [f for f in images 
              if not os.path.exists(
                  os.path.join(TRAIN_IMG, 
                               os.path.splitext(os.path.basename(f))[0] + '_gray' + 
                               os.path.splitext(f)[1]))]

sample = random.sample(to_process, int(len(to_process) * 0.30))
print(f"Original images       : {len(images)}")
print(f"Already have grayscale: {len(images) - len(to_process)}")
print(f"Adding grayscale for  : {len(sample)} new images (30%)")

added = 0
for img_path in sample:
    img = cv2.imread(img_path)
    if img is None:
        continue

    gray     = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    base    = os.path.splitext(os.path.basename(img_path))[0]
    ext     = os.path.splitext(img_path)[1]
    new_img = os.path.join(TRAIN_IMG, base + '_gray' + ext)
    cv2.imwrite(new_img, gray_bgr)

    lbl_path = os.path.join(TRAIN_LBL, base + '.txt')
    new_lbl  = os.path.join(TRAIN_LBL, base + '_gray.txt')
    if os.path.exists(lbl_path):
        shutil.copy2(lbl_path, new_lbl)
        added += 1

print(f"\n✅ Added {added} grayscale images!")