from ultralytics import RTDETR
import glob, os, shutil
from datetime import datetime

def backup_model(src_path, version_name):
    """Backup best.pt with timestamp so it never gets overwritten"""
    if not os.path.exists(src_path):
        return
    backup_dir = r'C:\Users\icroc\Documents\RT-DETR\backups'
    os.makedirs(backup_dir, exist_ok=True)
    ts  = datetime.now().strftime('%Y%m%d_%H%M%S')
    dst = os.path.join(backup_dir, f'{version_name}_best_{ts}.pt')
    shutil.copy2(src_path, dst)
    print(f'  Backed up to: {dst}')

if __name__ == '__main__':

    DATASET_DIR = r'C:\Users\icroc\Documents\RT-DETR\QCAR2_300_CLEAN'

    train_imgs = (glob.glob(rf'{DATASET_DIR}\train\images\*.jpg') +
                  glob.glob(rf'{DATASET_DIR}\train\images\*.jpeg') +
                  glob.glob(rf'{DATASET_DIR}\train\images\*.png'))

    valid_imgs = (glob.glob(rf'{DATASET_DIR}\valid\images\*.jpg') +
                  glob.glob(rf'{DATASET_DIR}\valid\images\*.jpeg') +
                  glob.glob(rf'{DATASET_DIR}\valid\images\*.png'))

    print("=" * 60)
    print("  RT-DETR XL — QCar2 Traffic Sign Detection")
    print("=" * 60)
    print(f"  Dataset  : QCAR2_300_CLEAN")
    print(f"  Train    : {len(train_imgs)} images")
    print(f"  Valid    : {len(valid_imgs)} images")
    print(f"  Split    : 60% train / 40% valid")
    print(f"  Classes  : 13")
    print(f"  Model    : rtdetr-x.pt (XL — largest RT-DETR)")
    print(f"  Epochs   : 350")
    print(f"  Device   : RTX 4070 Laptop GPU")
    print("=" * 60)

    # ── Start fresh from pretrained rtdetr-x (XL model) ──
    # Downloads automatically on first run (~160MB)
    model = RTDETR('rtdetr-x.pt')

    results = model.train(
        data    = rf'{DATASET_DIR}\data.yaml',
        epochs  = 350,
        imgsz   = 640,
        batch   = 4,           # XL model is heavier than L — reduce batch
        device  = 0,
        workers = 2,
        project = r'C:\Users\icroc\Documents\RT-DETR\runs',
        name    = 'qcar2_xl',
        save    = True,
        plots   = True,

        # Optimizer — fresh head on clean small dataset
        amp           = True,
        optimizer     = 'AdamW',
        lr0           = 0.0001,
        lrf           = 0.01,
        warmup_epochs = 5,
        patience      = 50,    # small dataset — needs more patience
        save_period   = 20,

        # Color augmentation
        hsv_h = 0.05,
        hsv_s = 0.7,
        hsv_v = 0.4,

        # ── NO horizontal flip ──
        # Turn Left / Turn Right signs — flipping causes confusion!
        fliplr = 0.0,
        flipud = 0.0,

        # Geometric augmentation
        scale     = 0.5,
        translate = 0.1,
        degrees   = 10.0,
        shear     = 2.0,

        # Advanced augmentation — critical for small dataset
        mosaic     = 1.0,
        mixup      = 0.05,
        copy_paste = 0.05,
        erasing    = 0.3,
    )

    # ── Auto backup best.pt immediately after training ──
    best_path = r'C:\Users\icroc\Documents\RT-DETR\runs\qcar2_xl\weights\best.pt'

    print('\n' + '=' * 60)
    print('  ✅  QCar2 XL Training Complete!')
    print('=' * 60)

    if os.path.exists(best_path):
        print('\n  Backing up best model...')
        backup_model(best_path, 'qcar2_xl')

        # Easy access copy — separate from old BEST_MODEL.pt
        easy_path = r'C:\Users\icroc\Documents\RT-DETR\BEST_MODEL_QCAR2.pt'
        shutil.copy2(best_path, easy_path)
        print(f'  Easy access copy: {easy_path}')

    print(f'\n  All models saved:')
    print(f'  Old V1   : runs/traffic_v15/weights/best.pt     (91.7% mAP)')
    print(f'  Old V2   : runs/traffic_final/weights/best.pt   (92.6% mAP)')
    print(f'  Old V2b  : runs/traffic_v2/weights/best.pt      (93.4% mAP)')
    print(f'  Old V3   : runs/traffic_v3/weights/best.pt      (old dataset)')
    print(f'  NEW QCar2: runs/qcar2_xl/weights/best.pt        (current)')
    print(f'  BEST OLD : BEST_MODEL.pt                        (old 31 class)')
    print(f'  BEST NEW : BEST_MODEL_QCAR2.pt                  (13 class QCar2)')
    print(f'\n  Backups folder: C:\\Users\\icroc\\Documents\\RT-DETR\\backups\\')
    print(f'\n  Update webcam:')
    print(f'    MODEL_PATH = r"C:\\Users\\icroc\\Documents\\RT-DETR\\BEST_MODEL_QCAR2.pt"')
    print('=' * 60)