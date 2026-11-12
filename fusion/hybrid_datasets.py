"""Create camera_hybrid and sonar_hybrid YOLO datasets from paired camera/sonar data.

camera_hybrid GT:
  - keep camera GT boxes
  - project sonar GT boxes into the camera image
  - add projected sonar boxes unless they overlap a same-class camera GT box

sonar_hybrid GT:
  - keep sonar GT boxes
  - project each camera GT box center into sonar coordinates using the configured
    depth heuristic, then add a square whose side length is the camera box width
    in pixels, unless it overlaps a same-class sonar GT box

IMPORTANT: a camera image alone does not provide metric depth. The default
heuristic below uses the camera bounding-box width (in pixels) as the numeric
camera-space depth. This is a heuristic, not a physically calibrated depth.
Adjust CAMERA_WIDTH_TO_DEPTH_SCALE if you want to change that assumption.
"""
from pathlib import Path
import shutil
import warnings
import cv2
import numpy as np

from fusion.camera_sonar_fusion_pipeline import get_sonar_configuration, sonar_box_to_camera_box,load_calibration, FusionParameters
from calibration.calibration_config import Calibration

CAMERA_ROOT = Path("dataset/test_croatia/camera-test")
SONAR_ROOT = Path("dataset/test_croatia/sonar-test-right")
CAMERA_IMAGES_DIR = CAMERA_ROOT / "images"
CAMERA_LABELS_DIR = CAMERA_ROOT / "labels"
SONAR_IMAGES_DIR = SONAR_ROOT / "images"
SONAR_LABELS_DIR = SONAR_ROOT / "labels"

OUTPUT_ROOT = Path("dataset/test_croatia/hybrid_datasets")
CAMERA_OUTPUT = OUTPUT_ROOT / "camera_hybrid"
SONAR_OUTPUT = OUTPUT_ROOT / "sonar_hybrid"

DEDUP_IOU_THRESHOLD = 0.1
CAMERA_WIDTH_TO_DEPTH_SCALE = 1.0
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

calibration = Calibration("./calibration/calibration_croatia_2.json")


def build_image_map(directory):
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    result = {}
    duplicates = []
    for p in sorted(directory.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            if p.stem in result:
                duplicates.append(p.stem)
            result[p.stem] = p
    if duplicates:
        raise ValueError(
            f"Duplicate image filename stems in {directory}: {duplicates[:10]}. "
            "Pairing by stem would be ambiguous; rename files or pair by relative path."
        )
    return result


def find_label(label_root, image_path):
    matches = list(Path(label_root).rglob(f"{image_path.stem}.txt"))
    if len(matches) > 1:
        raise ValueError(f"Multiple labels found for image {image_path.stem} in {label_root}")
    return matches[0] if matches else None


def read_yolo_labels(label_path, image_shape):
    """Return labels as pixel xyxy dictionaries: {'cls': int, 'xyxy': np.array}."""
    if label_path is None or not label_path.exists():
        return []
    h, w = image_shape[:2]
    labels = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) < 5:
                warnings.warn(f"Skipping malformed label {label_path}:{line_no}")
                continue
            cls_f, xc, yc, bw, bh = map(float, parts[:5])
            cls_id = int(cls_f)
            if not (0 <= cls_id < FusionParameters.NUM_CLASSES):
                warnings.warn(f"Skipping class {cls_id} outside 0..{FusionParameters.NUM_CLASSES - 1} in {label_path}")
                continue
            box = np.array([
                (xc - bw / 2) * w,
                (yc - bh / 2) * h,
                (xc + bw / 2) * w,
                (yc + bh / 2) * h,
            ], dtype=np.float32)
            box = clip_box(box, w, h)
            if box[2] > box[0] and box[3] > box[1]:
                labels.append({"cls": cls_id, "xyxy": box})
    return labels


def clip_box(box, width, height):
    x1, y1, x2, y2 = map(float, box)
    return np.array([
        np.clip(x1, 0, width - 1), np.clip(y1, 0, height - 1),
        np.clip(x2, 0, width - 1), np.clip(y2, 0, height - 1),
    ], dtype=np.float32)


def box_iou(a, b):
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def same_class_duplicate(box, cls_id, primary_labels, threshold):
    return any(
        int(item["cls"]) == int(cls_id)
        and box_iou(box, item["xyxy"]) >= threshold
        for item in primary_labels
    )


def xyxy_to_yolo(label, image_shape):
    h, w = image_shape[:2]
    x1, y1, x2, y2 = map(float, label["xyxy"])
    xc = ((x1 + x2) / 2) / w
    yc = ((y1 + y2) / 2) / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    # Guard against tiny floating-point excursions beyond [0, 1].
    return int(label["cls"]), np.clip(xc, 0, 1), np.clip(yc, 0, 1), np.clip(bw, 0, 1), np.clip(bh, 0, 1)


def write_yolo_labels(path, labels, image_shape):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for label in labels:
            cls_id, xc, yc, bw, bh = xyxy_to_yolo(label, image_shape)
            f.write(f"{cls_id} {xc:.8f} {yc:.8f} {bw:.8f} {bh:.8f}\n")


def copy_image_and_write_labels(image_path, source_image_root, output_root, labels):
    rel = image_path.relative_to(source_image_root)
    out_image = output_root / "images" / rel
    out_label = output_root / "labels" / rel.with_suffix(".txt")
    out_image.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image_path, out_image)
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Could not read image: {image_path}")
    write_yolo_labels(out_label, labels, image.shape)


def project_sonar_gt_to_camera(sonar_labels, sonar_image, camera_image, R, T, f):
    sonar_config = get_sonar_configuration(sonar_image)
    camera_h, camera_w = camera_image.shape[:2]
    projected = []
    for label in sonar_labels:
        result = sonar_box_to_camera_box(
            label["xyxy"], R, T, f, sonar_config, camera_image.shape
        )
        if result is None:
            continue
        box = clip_box(result[0] if isinstance(result, tuple) else result["xyxy"],
               camera_w, camera_h)
        if box[2] > box[0] and box[3] > box[1]:
            projected.append({"cls": int(label["cls"]), "xyxy": box})
    return projected


def project_camera_gt_to_sonar(camera_labels, camera_image, sonar_image, R, T, f):
    """Project camera box centers into sonar; use a square with side = camera box width.

    Camera center is back-projected at assumed depth Z = camera box width in pixels
    times CAMERA_WIDTH_TO_DEPTH_SCALE. The transform is inverted as Ps = R.T @ (Pc-T).
    The resulting sonar azimuth/range is mapped into the sonar image. Since the sonar
    image encodes azimuth horizontally and range vertically, the square side is the
    original camera box width in pixels, as requested.
    """
    sonar_h, sonar_w = sonar_image.shape[:2]
    sonar_config = get_sonar_configuration(sonar_image)
    tip_x, tip_y = sonar_config["sonar_tip_pixel_coord"]
    scale = sonar_config["sonar_scaling_factor"]
    cam_h, cam_w = camera_image.shape[:2]
    cu, cv = cam_w / 2.0, cam_h / 2.0
    projected = []

    for label in camera_labels:
        x1, y1, x2, y2 = map(float, label["xyxy"])
        box_width = x2 - x1
        if box_width <= 0:
            continue
        u = (x1 + x2) / 2.0
        v = (y1 + y2) / 2.0
        assumed_depth = max(box_width * CAMERA_WIDTH_TO_DEPTH_SCALE, 1e-3)

        # Camera pinhole back-projection at assumed depth.
        P_c = np.array([
            (u - cu) * assumed_depth / f,
            (v - cv) * assumed_depth / f,
            assumed_depth,
        ], dtype=np.float64)
        # Invert P_c = R @ P_s + T.
        P_s = R.T @ (P_c - T)
        if not np.all(np.isfinite(P_s)):
            continue

        # The sonar image uses horizontal azimuth and vertical range.
        rho = float(np.hypot(P_s[0], P_s[1]))
        theta = float(np.arctan2(P_s[0], P_s[1]))
        if rho <= 0:
            continue
        center_u = tip_x + rho * np.sin(theta) * scale
        center_v = tip_y - rho * np.cos(theta) * scale

        half = box_width / 2.0
        square = clip_box(
            [center_u - half, center_v - half, center_u + half, center_v + half],
            sonar_w, sonar_h,
        )
        if square[2] > square[0] and square[3] > square[1]:
            projected.append({"cls": int(label["cls"]), "xyxy": square})
    return projected

    
if __name__ == "__main__":

    camera_images = build_image_map(CAMERA_IMAGES_DIR)
    sonar_images = build_image_map(SONAR_IMAGES_DIR)
    shared_stems = sorted(set(camera_images) & set(sonar_images))
    camera_only = sorted(set(camera_images) - set(sonar_images))
    sonar_only = sorted(set(sonar_images) - set(camera_images))
    if not shared_stems:
        raise RuntimeError("No camera/sonar image pairs found with matching filename stems.")
    if camera_only or sonar_only:
        print(f"Warning: {len(camera_only)} camera-only and {len(sonar_only)} sonar-only images will not be used.")

    R, T, f = load_calibration(calibration.params["alpha"], calibration.params["beta"], calibration.params["gamma"], calibration.params["tx"], calibration.params["ty"], calibration.params["tz"], calibration.params["f"])
    CAMERA_OUTPUT.mkdir(parents=True, exist_ok=True)
    SONAR_OUTPUT.mkdir(parents=True, exist_ok=True)

    stats = {
        "pairs": 0,
        "camera_gt": 0,
        "sonar_gt": 0,
        "camera_hybrid_added_sonar": 0,
        "camera_hybrid_deduplicated_sonar": 0,
        "sonar_hybrid_added_camera": 0,
        "sonar_hybrid_deduplicated_camera": 0,
        "sonar_gt_projection_failed": 0,
        "camera_gt_projection_failed_or_outside": 0,
    }

    for i, stem in enumerate(shared_stems, 1):
        cam_path, sonar_path = camera_images[stem], sonar_images[stem]
        cam_img = cv2.imread(str(cam_path))
        sonar_img = cv2.imread(str(sonar_path))
        if cam_img is None or sonar_img is None:
            warnings.warn(f"Skipping unreadable image pair: {stem}")
            continue

        cam_gt = read_yolo_labels(find_label(CAMERA_LABELS_DIR, cam_path), cam_img.shape)
        sonar_gt = read_yolo_labels(find_label(SONAR_LABELS_DIR, sonar_path), sonar_img.shape)
        stats["camera_gt"] += len(cam_gt)
        stats["sonar_gt"] += len(sonar_gt)

        # Camera hybrid: camera GT is primary; add non-duplicate projected sonar GT.
        projected_sonar = project_sonar_gt_to_camera(sonar_gt, sonar_img, cam_img, R, T, f)
        camera_hybrid_gt = [dict(cls=x["cls"], xyxy=x["xyxy"].copy()) for x in cam_gt]
        for item in projected_sonar:
            if same_class_duplicate(item["xyxy"], item["cls"], cam_gt, DEDUP_IOU_THRESHOLD):
                stats["camera_hybrid_deduplicated_sonar"] += 1
            else:
                camera_hybrid_gt.append(item)
                stats["camera_hybrid_added_sonar"] += 1
        stats["sonar_gt_projection_failed"] += max(0, len(sonar_gt) - len(projected_sonar))

        # Sonar hybrid: sonar GT is primary; add non-duplicate projected camera squares.
        projected_camera = project_camera_gt_to_sonar(cam_gt, cam_img, sonar_img, R, T, f)
        sonar_hybrid_gt = [dict(cls=x["cls"], xyxy=x["xyxy"].copy()) for x in sonar_gt]
        for item in projected_camera:
            if same_class_duplicate(item["xyxy"], item["cls"], sonar_gt, DEDUP_IOU_THRESHOLD):
                stats["sonar_hybrid_deduplicated_camera"] += 1
            else:
                sonar_hybrid_gt.append(item)
                stats["sonar_hybrid_added_camera"] += 1
        stats["camera_gt_projection_failed_or_outside"] += max(0, len(cam_gt) - len(projected_camera))

        copy_image_and_write_labels(cam_path, CAMERA_IMAGES_DIR, CAMERA_OUTPUT, camera_hybrid_gt)
        copy_image_and_write_labels(sonar_path, SONAR_IMAGES_DIR, SONAR_OUTPUT, sonar_hybrid_gt)
        stats["pairs"] += 1
        if i % 50 == 0 or i == len(shared_stems):
            print(f"Processed {i}/{len(shared_stems)} pairs")

    # Write dataset YAML files suitable for Ultralytics validation.
    camera_yaml = CAMERA_OUTPUT / "camera_hybrid.yaml"
    sonar_yaml = SONAR_OUTPUT / "sonar_hybrid.yaml"
    camera_yaml.write_text(
        "path: " + str(CAMERA_OUTPUT.resolve()) + "\n"
        "train: images\nval: images\ntest: images\n"
        "names:\n" + "\n".join(f"  {i}: {name}" for i, name in enumerate(FusionParameters.CLASS_NAMES)) + "\n",
        encoding="utf-8",
    )
    sonar_yaml.write_text(
        "path: " + str(SONAR_OUTPUT.resolve()) + "\n"
        "train: images\nval: images\ntest: images\n"
        "names:\n" + "\n".join(f"  {i}: {name}" for i, name in enumerate(FusionParameters.CLASS_NAMES)) + "\n",
        encoding="utf-8",
    )

    print("\nHybrid dataset creation complete")
    print(f"Camera hybrid: {CAMERA_OUTPUT.resolve()}")
    print(f"  YAML: {camera_yaml.resolve()}")
    print(f"Sonar hybrid:  {SONAR_OUTPUT.resolve()}")
    print(f"  YAML: {sonar_yaml.resolve()}")
    print("\nCounts:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print("\nNote: camera->sonar projection uses the configured width-based depth heuristic, not measured metric depth.")


