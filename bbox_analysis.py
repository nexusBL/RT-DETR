import glob, numpy as np

LABELS = r'C:\Users\icroc\Documents\RT-DETR\MERGED\labels\train'

all_widths  = []
all_heights = []
all_areas   = []

for lbl in glob.glob(LABELS + '/*.txt'):
    for line in open(lbl).readlines():
        p = line.strip().split()
        if len(p) >= 5:
            w = float(p[3])  # normalized width
            h = float(p[4])  # normalized height
            all_widths.append(w)
            all_heights.append(h)
            all_areas.append(w * h)

all_widths  = np.array(all_widths)
all_heights = np.array(all_heights)
all_areas   = np.array(all_areas)

print("="*60)
print("BOUNDING BOX SIZE ANALYSIS")
print("(all values are % of image size — 1.0 = full image)")
print("="*60)

print(f"\nBox WIDTH:")
print(f"  Smallest  : {all_widths.min():.3f}  ({all_widths.min()*100:.1f}% of image)")
print(f"  Largest   : {all_widths.max():.3f}  ({all_widths.max()*100:.1f}% of image)")
print(f"  Average   : {all_widths.mean():.3f}  ({all_widths.mean()*100:.1f}% of image)")
print(f"  Median    : {np.median(all_widths):.3f}  ({np.median(all_widths)*100:.1f}% of image)")

print(f"\nBox HEIGHT:")
print(f"  Smallest  : {all_heights.min():.3f}  ({all_heights.min()*100:.1f}% of image)")
print(f"  Largest   : {all_heights.max():.3f}  ({all_heights.max()*100:.1f}% of image)")
print(f"  Average   : {all_heights.mean():.3f}  ({all_heights.mean()*100:.1f}% of image)")
print(f"  Median    : {np.median(all_heights):.3f}  ({np.median(all_heights)*100:.1f}% of image)")

print(f"\nBox AREA (w x h):")
print(f"  Smallest  : {all_areas.min():.5f}  ({all_areas.min()*100:.2f}% of image)")
print(f"  Largest   : {all_areas.max():.3f}  ({all_areas.max()*100:.1f}% of image)")
print(f"  Average   : {all_areas.mean():.4f}  ({all_areas.mean()*100:.2f}% of image)")
print(f"  Median    : {np.median(all_areas):.4f}  ({np.median(all_areas)*100:.2f}% of image)")

print(f"\nTotal boxes analysed: {len(all_areas):,}")

print(f"\nSIZE DISTRIBUTION:")
tiny   = (all_areas < 0.01).sum()
small  = ((all_areas >= 0.01) & (all_areas < 0.05)).sum()
medium = ((all_areas >= 0.05) & (all_areas < 0.15)).sum()
large  = (all_areas >= 0.15).sum()
total  = len(all_areas)
print(f"  Tiny   (<1% of image)  : {tiny:>6} boxes  ({tiny/total*100:.1f}%)")
print(f"  Small  (1-5% of image) : {small:>6} boxes  ({small/total*100:.1f}%)")
print(f"  Medium (5-15% of image): {medium:>6} boxes  ({medium/total*100:.1f}%)")
print(f"  Large  (>15% of image) : {large:>6} boxes  ({large/total*100:.1f}%)")

print(f"\nCONCLUSION:")
med = np.median(all_areas)
if med < 0.02:
    print("  Signs are SMALL in dataset = model detects far away signs")
    print("  May STRUGGLE with close-up signs on webcam!")
elif med < 0.08:
    print("  Signs are MEDIUM in dataset = model detects medium distance")
    print("  Works OK at 1-3 meters distance")
else:
    print("  Signs are LARGE in dataset = model detects close-up signs")
    print("  May STRUGGLE with far away signs!")
