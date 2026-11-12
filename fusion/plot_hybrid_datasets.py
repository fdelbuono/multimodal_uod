import matplotlib
matplotlib.use("Agg")
from pathlib import Path
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HYBRID_ROOT = Path("dataset/test_croatia/hybrid_datasets")

CAMERA_HYBRID = HYBRID_ROOT / "camera_hybrid"
SONAR_HYBRID = HYBRID_ROOT / "sonar_hybrid"

CAMERA_ORIGINAL_ROOT = Path("dataset/test_croatia/camera-test")
SONAR_ORIGINAL_ROOT = Path("dataset/test_croatia/sonar-test-right")

NUM_PAIRS_TO_PLOT = 2
RANDOM_SEED = 67
SAVE_FIGURE = True
OUTPUT_FIGURE = HYBRID_ROOT / "hybrid_dataset_examples.png"

CLASS_NAMES = ["rov", "plant", "animal", "trash"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# Colors are RGB for Matplotlib.
ORIGINAL_COLOR = (0.1, 0.85, 0.25)   # green
PROJECTED_COLOR = (1.0, 0.55, 0.05)  # orange


def image_map(root):
    images_dir = Path(root) / "images"
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    result = {}
    for p in sorted(images_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            if p.stem in result:
                raise ValueError(f"Duplicate image stem '{p.stem}' under {images_dir}")
            result[p.stem] = p
    return result


def label_path(root, stem):
    p = Path(root) / "labels" / f"{stem}.txt"
    if p.exists():
        return p
    # Support nested image/label folders if dataset has nested structure.
    matches = list((Path(root) / "labels").rglob(f"{stem}.txt"))
    return matches[0] if len(matches) == 1 else None


def read_yolo(path, image_shape):
    if path is None or not path.exists():
        return []
    h, w = image_shape[:2]
    boxes = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                cls_id = int(float(parts[0]))
                xc, yc, bw, bh = map(float, parts[1:5])
            except ValueError:
                continue
            x1 = (xc - bw / 2) * w
            y1 = (yc - bh / 2) * h
            x2 = (xc + bw / 2) * w
            y2 = (yc + bh / 2) * h
            boxes.append({"cls": cls_id, "xyxy": np.array([x1, y1, x2, y2], dtype=float)})
    return boxes


def box_iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def is_original(label, original_labels, iou_threshold=0.98):
    # Labels in the hybrid set preserve primary-modality boxes, so near-exact
    # same-class overlap identifies original boxes; others are projected labels.
    for orig in original_labels:
        if label["cls"] == orig["cls"] and box_iou(label["xyxy"], orig["xyxy"]) >= iou_threshold:
            return True
    return False


def draw_labels(ax, image_rgb, hybrid_labels, original_labels, title):
    ax.imshow(image_rgb)
    ax.set_title(title, fontsize=11)
    ax.axis("off")
    for label in hybrid_labels:
        x1, y1, x2, y2 = label["xyxy"]
        original = is_original(label, original_labels)
        color = ORIGINAL_COLOR if original else PROJECTED_COLOR
        rect = Rectangle((x1, y1), max(1, x2 - x1), max(1, y2 - y1),
                         linewidth=2, edgecolor=color, facecolor="none")
        ax.add_patch(rect)
        cls_id = label["cls"]
        cls_name = CLASS_NAMES[cls_id] if 0 <= cls_id < len(CLASS_NAMES) else str(cls_id)
        tag = f"{cls_name} | {'original' if original else 'projected'}"
        ax.text(x1, max(0, y1 - 4), tag, color="black", fontsize=7,
                bbox=dict(facecolor=color, alpha=0.9, pad=1, edgecolor="none"))


if __name__ == "__main__":

    
    cam_map = image_map(CAMERA_HYBRID)
    sonar_map = image_map(SONAR_HYBRID)
    common = sorted(set(cam_map) & set(sonar_map))
    if not common:
        raise RuntimeError(
            "No matching image stems found between camera_hybrid/images and "
            "sonar_hybrid/images. Check pairing and output paths."
        )

    count = min(NUM_PAIRS_TO_PLOT, len(common))
    rng = random.Random(RANDOM_SEED)
    chosen = sorted(rng.sample(common, count)) if count < len(common) else common

    fig, axes = plt.subplots(count, 2, figsize=(16, max(4, count * 4.5)), squeeze=False)
    for row, stem in enumerate(chosen):
        cam_bgr = cv2.imread(str(cam_map[stem]))
        sonar_bgr = cv2.imread(str(sonar_map[stem]))
        if cam_bgr is None or sonar_bgr is None:
            print(f"Skipping unreadable pair: {stem}")
            continue
        cam_rgb = cv2.cvtColor(cam_bgr, cv2.COLOR_BGR2RGB)
        sonar_rgb = cv2.cvtColor(sonar_bgr, cv2.COLOR_BGR2RGB)

        cam_hybrid_labels = read_yolo(label_path(CAMERA_HYBRID, stem), cam_bgr.shape)
        sonar_hybrid_labels = read_yolo(label_path(SONAR_HYBRID, stem), sonar_bgr.shape)
        cam_original_labels = read_yolo(label_path(CAMERA_ORIGINAL_ROOT, stem), cam_bgr.shape)
        sonar_original_labels = read_yolo(label_path(SONAR_ORIGINAL_ROOT, stem), sonar_bgr.shape)

        draw_labels(axes[row, 0], cam_rgb, cam_hybrid_labels, cam_original_labels,
                    f"Camera hybrid — {stem}")
        draw_labels(axes[row, 1], sonar_rgb, sonar_hybrid_labels, sonar_original_labels,
                    f"Sonar hybrid — {stem}")

    # Figure-level legend.
    handles = [
        Rectangle((0, 0), 1, 1, facecolor=ORIGINAL_COLOR, edgecolor=ORIGINAL_COLOR,
                  label="Original modality label"),
        Rectangle((0, 0), 1, 1, facecolor=PROJECTED_COLOR, edgecolor=PROJECTED_COLOR,
                  label="Added projected label"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("Hybrid ground-truth examples", y=1.015, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.98])

    if SAVE_FIGURE:
        OUTPUT_FIGURE.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUTPUT_FIGURE, dpi=180, bbox_inches="tight")
        print(f"Saved figure to: {OUTPUT_FIGURE}")
    plt.show()

